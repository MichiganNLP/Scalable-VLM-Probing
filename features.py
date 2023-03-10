from __future__ import annotations

import ast
import itertools
import json
import string
from collections import Counter, defaultdict
from typing import Any, Callable, Collection, Iterable, Literal, Mapping, MutableSequence, Sequence, Tuple, get_args

import numpy as np
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn_pandas import DataFrameMapper

NegType = Literal["s", "v", "o"]
Triplet = Tuple[str, str, str]

VALID_NEG_TYPES = get_args(NegType)

PATH_LEVIN_VERBS = "data/levin_verbs.txt"
PATH_LEVIN_SEMANTIC_BROAD = "data/levin_semantic_broad.json"
PATH_LIWC = "data/LIWC.2015.all.txt"
PATH_CONCRETENESS = "data/concreteness.txt"
PATH_WORD_FREQUENCIES = "data/words_counter_LAION.json"

text_model = SentenceTransformer("all-MiniLM-L6-v2")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def _parse_triplets(triplets: str) -> Sequence[Triplet]:
    if triplets.startswith("["):
        return [triplet.split(",") for triplet in ast.literal_eval(triplets)]
    else:
        return [triplets.split(",")]  # noqa


def _lemmatize(word: str) -> str:
    return "person" if word == "people" else lemmatizer.lemmatize(word)


def _stem(word: str) -> str:
    return stemmer.stem(word, to_lowercase=False)


def _get_sentence_match_triplet(triplets: Sequence[Triplet], sentence: str) -> Triplet:
    if len(triplets) == 1:
        return triplets[0]
    else:
        words = sentence.split()
        lemmatized_words = (_lemmatize(word) for word in words)
        stemmed_words = (_stem(word) for word in words)
        all_words = set(itertools.chain(words, lemmatized_words, stemmed_words))
        return next((triplet for triplet in triplets if set(triplet) <= all_words), triplets[0])


def _preprocess_sentences(sentences: pd.Series) -> pd.Series:
    return sentences.str.lower().str.translate(str.maketrans("", "", string.punctuation))


def _load_clip_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["sentence", "neg_sentence", "pos_triplet", "neg_triplet", "neg_type",
                                    "clip prediction", "clip_score_diff", "pos_clip_score",
                                    "neg_clip_score"]).sort_index()

    df.sentence = _preprocess_sentences(df.sentence)
    df.neg_sentence = _preprocess_sentences(df.neg_sentence)

    df.pos_triplet = df.pos_triplet.apply(_parse_triplets)
    df.neg_triplet = df.neg_triplet.apply(_parse_triplets)

    df = df[((df.pos_triplet.str.len() > 0)  # The triplets are list, but we can use this `str` function.
             & (df.neg_triplet.str.len() > 0)
             & (df.sentence.str.len() > 0)
             & (df.neg_sentence.str.len() > 0))]

    df.loc[:, "pos_triplet"] = df.apply(lambda row: _get_sentence_match_triplet(row.pos_triplet, row.sentence), axis=1)
    df.loc[:, "neg_triplet"] = df.apply(lambda row: _get_sentence_match_triplet(row.neg_triplet, row.neg_sentence),
                                        axis=1)

    df.neg_type = df.neg_type.str.get(0)
    assert df.neg_type.isin(VALID_NEG_TYPES).all()

    df["clip prediction"] = df["clip prediction"] == "pos"

    return df


def _parse_levin_file(path: str = PATH_LEVIN_VERBS, path_semantic_broad: str = PATH_LEVIN_SEMANTIC_BROAD,
                      return_mode: Literal["alternation", "semantic_broad", "semantic_fine_grained", "all"] = "all",
                      verbose: bool = True) -> Mapping[str, Collection[str]]:
    content = ""
    map_class_name_to_words = {}
    with open(path) as file:
        for line in file:
            line = line.lstrip()
            if line and line[0].isnumeric():
                class_name = " ".join(line.split())
            else:
                if line:
                    content += line.replace("\r\n", "").rstrip()
                    content += " "
                else:
                    if "-*-" not in content:
                        map_class_name_to_words[class_name] = {w.lower() for w in content.split()}
                    content = ""
        if "-*-" not in content:
            map_class_name_to_words[class_name] = {w.lower() for w in content.split()}

    with open(path_semantic_broad) as file:
        map_class_number_to_broad_name = {int(number_str): name for number_str, name in json.load(file).items()}

    map_word_to_semantic_broad_class_names = defaultdict(set)
    map_word_to_semantic_fine_grained_class_names = defaultdict(set)
    map_word_to_alternation_class_names = defaultdict(set)
    map_word_to_class_names = defaultdict(set)

    semantic_broad_classes = set()
    semantic_fine_grained_class_count = 0
    alternation_class_count = 0

    for class_name, words in map_class_name_to_words.items():
        for word in words:
            map_word_to_class_names[word].add(class_name)

        if (class_number := int(class_name.split(" ", maxsplit=1)[0].split(".", maxsplit=1)[0])) <= 8:
            alternation_class_count += 1

            for word in words:
                map_word_to_alternation_class_names[word].add(class_name)
        else:
            semantic_fine_grained_class_count += 1

            broad_class_name = map_class_number_to_broad_name[class_number]
            semantic_broad_classes.add(broad_class_name)

            for word in words:
                map_word_to_semantic_fine_grained_class_names[word].add(class_name)
                map_word_to_semantic_broad_class_names[broad_class_name].update(words)

    if verbose:
        print(f"--Levin semantic broad nb classes:", len(semantic_broad_classes))
        print(f"--Levin semantic fine-grained nb classes:", semantic_fine_grained_class_count)
        print(f"--Levin alternations nb classes:", alternation_class_count)
        print(f"--Levin total nb classes:", len(map_class_name_to_words))

    if return_mode == "alternation":
        return map_word_to_alternation_class_names
    elif return_mode == "semantic_broad":
        return map_word_to_semantic_broad_class_names
    elif return_mode == "semantic_fine_grained":
        return map_word_to_semantic_fine_grained_class_names
    elif return_mode == "all":
        return map_word_to_class_names
    else:
        raise ValueError(f"Invalid return mode: {return_mode}")


def _get_levin_category(word: str, dict_levin: Mapping[str, Collection[str]], neg_type: NegType) -> Collection[str]:
    return dict_levin.get(word, []) if neg_type == "v" else []


def _get_nb_synsets(word: str, neg_type: NegType) -> int:  # noqa
    # pos = _neg_type_to_pos(neg_type)
    # synsets = wordnet.synsets(word, pos=pos)  # FIXME: why not using the POS?
    synsets = wordnet.synsets(word)
    return len(synsets)


def _get_hypernyms(word: str, neg_type: NegType) -> Sequence[str]:
    if synsets := wordnet.synsets(word, pos=_neg_type_to_pos(neg_type)):
        hypernym_synsets = synsets[0].hypernyms() or synsets  # FIXME: why not using all the synsets?
        return [hypernym_synsets[0]._lemma_names[0]]  # FIXME: why not returning all lemma names?
    else:
        return [word]


def _parse_liwc_file(path: str = PATH_LIWC, verbose: bool = True) -> Mapping[str, Sequence[str]]:
    dict_liwc = defaultdict(list)
    liwc_categories = set()

    with open(path) as file:
        for line in file:
            word, category = (w.strip() for w in line.strip().split(","))
            dict_liwc[word].append(category)
            liwc_categories.add(category)

    if verbose:
        print(f"Total number of LIWC categories:", len(liwc_categories))

    return dict_liwc


def _get_liwc_category(word: str, dict_liwc: Mapping[str, Sequence[str]]) -> Collection[str]:
    # The shortest word in LIWC with a wildcard is 2 characters long.
    return {category
            for categories in itertools.chain([dict_liwc.get(word, [])],
                                              (dict_liwc.get(word[:i] + "*", []) for i in range(2, len(word) + 1)))
            for category in categories}


def _parse_concreteness_file(path: str = PATH_CONCRETENESS) -> Mapping[str, float]:
    dict_concreteness = {}
    with open(path) as file:
        next(file)  # Skip the first line.
        for line in file:
            word, _, concreteness_m, _, _, _, _, _, _ = line.split("	")
            dict_concreteness[word] = float(concreteness_m)
    return dict_concreteness


def _get_concreteness_score(word: str, dict_concreteness: Mapping[str, float]) -> float:
    return dict_concreteness.get(word, float("nan"))


def _neg_type_to_pos(neg_type: NegType) -> Literal["n", "v"]:
    return "v" if neg_type == "v" else "n"  # noqa


def _compute_wup_similarity(word_original: str, word_replacement: str, neg_type: NegType) -> float:
    pos = _neg_type_to_pos(neg_type)
    return max((synset_original.wup_similarity(synset_replacement)
                for synset_original in wordnet.synsets(word_original, pos=pos)
                for synset_replacement in wordnet.synsets(word_replacement, pos=pos)),
               default=float("nan"))


def _compute_lch_similarity(word_original: str, word_replacement: str, neg_type: NegType) -> float:
    pos = _neg_type_to_pos(neg_type)
    return max((synset_original.lch_similarity(synset_replacement)
                for synset_original in wordnet.synsets(word_original, pos=pos)
                for synset_replacement in wordnet.synsets(word_replacement, pos=pos)),
               default=float("nan"))


def _compute_path_similarity(word_original: str, word_replacement: str, neg_type: NegType) -> float:
    pos = _neg_type_to_pos(neg_type)
    return max((synset_original.path_similarity(synset_replacement)
                for synset_original in wordnet.synsets(word_original, pos=pos)
                for synset_replacement in wordnet.synsets(word_replacement, pos=pos)),
               default=float("nan"))


def _neg_type_name_to_index(neg_type: NegType) -> int:
    return VALID_NEG_TYPES.index(neg_type)


def _get_changed_word(triplet: Triplet, neg_type: NegType) -> str:
    return triplet[_neg_type_name_to_index(neg_type)]


def _get_common_words(triplet: Triplet, neg_type: NegType) -> Collection[str]:
    return [t for t, other_neg_type in zip(triplet, VALID_NEG_TYPES) if other_neg_type != neg_type]


def _compute_feature_for_each_word(df: pd.DataFrame, prefix: str, func: Callable[[str, pd.Series], Any],
                                   compute_neg_features: bool = True) -> None:
    if compute_neg_features:
        df[f"{prefix}-original"] = df.apply(lambda row: func(row["word-original"], row), axis=1)
        df[f"{prefix}-replacement"] = df.apply(lambda row: func(row["word-replacement"], row), axis=1)

    placeholder_result = pd.Series([func("dog", df.iloc[0])])

    common_results = df.apply(lambda row: [func(w, row) for w in row["words-common"]], axis=1)

    if np.issubdtype(placeholder_result.dtype, np.number):
        df[f"{prefix}-common"] = common_results.map(lambda r: sum(r) / len(r))
        if compute_neg_features:
            df[f"{prefix}-change"] = df[f"{prefix}-original"] - df[f"{prefix}-replacement"]
    elif is_feature_string(placeholder_result):
        df[f"{prefix}-common"] = common_results.map(set)
    elif is_feature_multi_label(placeholder_result):
        df[f"{prefix}-common"] = common_results.map(lambda r: {label for labels in r for label in labels})
    else:
        raise ValueError(f"Unknown feature type {type(placeholder_result)}")

    common_results_df = pd.DataFrame(common_results.values.tolist(), index=common_results.index)
    for i, column_name in enumerate(common_results_df.columns):
        df[f"{prefix}-common-{i}"] = common_results_df[column_name]


def _compute_features(clip_results: pd.DataFrame, feature_deny_list: Collection[str] = frozenset(),
                      max_feature_count: int | None = None, compute_neg_features: bool = True,
                      compute_similarity_features: bool = True) -> pd.DataFrame:
    print("Computing all the features…")

    if max_feature_count:
        clip_results = clip_results.sample(max_feature_count)

    df = clip_results.copy()

    if compute_neg_features:
        df["word-original"] = df.apply(lambda row: _get_changed_word(row.pos_triplet, row.neg_type), axis=1)
        df["word-replacement"] = df.apply(lambda row: _get_changed_word(row.neg_triplet, row.neg_type), axis=1)
        df["words-common"] = df.apply(lambda row: _get_common_words(row.pos_triplet, row.neg_type), axis=1)
    else:
        df["words-common"] = df.pos_triplet

    for i in range(len(df["words-common"].iloc[0])):
        df[f"words-common-{i}"] = df["words-common"].str[i]

    if "Levin" not in feature_deny_list:
        dict_levin = _parse_levin_file()
        _compute_feature_for_each_word(df, "Levin",
                                       lambda w, row: _get_levin_category(w, dict_levin, row.neg_type),
                                       compute_neg_features=compute_neg_features)

    if "LIWC" not in feature_deny_list:
        dict_liwc = _parse_liwc_file()
        _compute_feature_for_each_word(df, "LIWC", lambda w, _: _get_liwc_category(w, dict_liwc),
                                       compute_neg_features=compute_neg_features)

    if "hypernym" not in feature_deny_list:
        print("Computing the hypernyms…", end="")
        _compute_feature_for_each_word(df, "hypernym", lambda w, row: _get_hypernyms(w, row.neg_type),
                                       compute_neg_features=compute_neg_features)
        print(" ✓")

    if "frequency" not in feature_deny_list:
        with open(PATH_WORD_FREQUENCIES) as json_file:
            word_frequencies = json.load(json_file)
        _compute_feature_for_each_word(df, "frequency", lambda w, _: word_frequencies.get(w, 0),
                                       compute_neg_features=compute_neg_features)

    if "concreteness" not in feature_deny_list:
        dict_concreteness = _parse_concreteness_file()
        _compute_feature_for_each_word(df, "concreteness", lambda w, _: _get_concreteness_score(w, dict_concreteness),
                                       compute_neg_features=compute_neg_features)

    if "nb-synsets" not in feature_deny_list:
        print("Computing the number of synsets…", end="")
        _compute_feature_for_each_word(df, "nb-synsets", lambda w, row: _get_nb_synsets(w, row.neg_type),
                                       compute_neg_features=compute_neg_features)
        print(" ✓")

    if compute_similarity_features:
        if "text-similarity" not in feature_deny_list and compute_neg_features:
            print("Computing the text similarity…")

            embedded_sentences = text_model.encode(df.sentence.array, show_progress_bar=True)
            embedded_neg_sentences = text_model.encode(df.neg_sentence.array, show_progress_bar=True)

            df["text-similarity"] = util.pairwise_cos_sim(embedded_sentences, embedded_neg_sentences)
            # We set the similarity to NaN for empty sentences:
            df.loc[[s == "" for s in df.neg_sentence], "text-similarity"] = float("nan")

        if "word-similarity" not in feature_deny_list and compute_neg_features:
            print("Computing the word similarity…")

            embedded_original_words = text_model.encode(df["word-original"].array, show_progress_bar=True)
            embedded_replacement_words = text_model.encode(df["word-replacement"].array, show_progress_bar=True)

            df["word-similarity"] = util.pairwise_cos_sim(embedded_original_words, embedded_replacement_words)

        if "wup-similarity" not in feature_deny_list and compute_neg_features:
            print("Computing the Wu-Palmer similarity…", end="")
            df["wup-similarity"] = df.apply(
                lambda row: _compute_wup_similarity(row["word-original"], row["word-replacement"], row.neg_type),
                axis=1)
            print(" ✓")

        if "lch-similarity" not in feature_deny_list and compute_neg_features:
            print("Computing the Leacock-Chodorow similarity…", end="")
            df["lch-similarity"] = df.apply(
                lambda row: _compute_lch_similarity(row["word-original"], row["word-replacement"], row.neg_type),
                axis=1)
            print(" ✓")

        if "path-similarity" not in feature_deny_list and compute_neg_features:
            print("Computing the Path similarity…", end="")
            df["path-similarity"] = df.apply(
                lambda row: _compute_path_similarity(row["word-original"], row["word-replacement"], row.neg_type), axis=1)
            print(" ✓")

    print("Feature computation done.")

    return df


# sklearn-pandas doesn't support the new way (scikit-learn >= 1.1) some transformers output the features.
# See https://github.com/scikit-learn-contrib/sklearn-pandas/pull/248
def _fix_column_names(df: pd.DataFrame, mapper: DataFrameMapper) -> pd.DataFrame:
    for columns, transformer, kwargs in mapper.built_features:
        if (isinstance(transformer, OneHotEncoder)
                or (isinstance(transformer, Pipeline) and any(isinstance(t, OneHotEncoder) for t in transformer))):
            assert isinstance(columns, Iterable) and not isinstance(columns, str)

            new_names = transformer.get_feature_names_out(columns)

            old_name_prefix = kwargs.get("alias", "_".join(str(c) for c in columns))
            old_names = [f"{old_name_prefix}_{i}" for i in range(len(new_names))]

            df = df.rename(columns=dict(zip(old_names, new_names)))
        elif isinstance(transformer, Pipeline) and isinstance(transformer[0], MultiLabelBinarizer):
            # The way sklearn-pandas infers the names is by iterating the transformers and getting the names and trying
            # to get the features names that are available from the last one that has them. Then, it checks if their
            # length matches the output number of features. However, if the binarizer is followed by feature selection,
            # this process fails as the previous condition is not met. So we handle it manually here.
            assert isinstance(columns, str)

            # MultiLabelBinarizer doesn't implement `get_feature_names_out`.
            new_names = [f"{columns}_{c}" for c in transformer[0].classes_]

            # We slice as an iterator and not by passing a slice to `__getitem__` because if the transformer is of type
            # `TransformerPipeline` then it fails.
            for t in itertools.islice(transformer, 1, None):
                new_names = t.get_feature_names_out(new_names)

            old_name_prefix = kwargs.get("alias", columns)
            old_names = [f"{old_name_prefix}_{i}" for i in range(len(new_names))]

            df = df.rename(columns=dict(zip(old_names, new_names)))

    return df


class SelectMinNonZero(SelectorMixin, BaseEstimator):
    def __init__(self, feature_min_non_zero_values: int = 50) -> None:
        self.feature_min_non_zero_values = feature_min_non_zero_values

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> SelectMinNonZero:  # noqa
        assert not np.issubdtype(X.dtype, np.floating)
        self.non_zero_counts_ = (X != 0).sum(axis=0)  # noqa
        if isinstance(self.non_zero_counts_, np.matrix):  # Can happen with CSR matrices.
            self.non_zero_counts_ = self.non_zero_counts_.A1  # noqa
        return self

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self)
        return self.non_zero_counts_ >= self.feature_min_non_zero_values  # noqa


def _infer_transformer(feature: np.ndarray, impute_missing_values: bool = True,
                       standardize_binary_features: bool = True, feature_min_non_zero_values: int = 50) -> Any:
    transformers = None

    dtype = feature.dtype
    if is_feature_binary(feature):
        transformers = [SelectMinNonZero(feature_min_non_zero_values)]
    elif np.issubdtype(dtype, np.number):
        transformers = ([SimpleImputer()] if impute_missing_values else []) + [StandardScaler()]  # noqa
    elif is_feature_string(feature):
        transformers = [OneHotEncoder(dtype=bool, sparse_output=not standardize_binary_features),
                        SelectMinNonZero(feature_min_non_zero_values)]
    elif is_feature_multi_label(feature):
        transformers = [MultiLabelBinarizer(), SelectMinNonZero(feature_min_non_zero_values)]

    if standardize_binary_features and (is_feature_binary(feature) or is_feature_multi_label(feature)
                                        or is_feature_string(feature)):
        transformers.append(StandardScaler())

    return transformers


def _transform_features_to_numbers(
        df: pd.DataFrame, dependent_variable_name: str, standardize_dependent_variable: bool = True,
        standardize_binary_features: bool = True, feature_min_non_zero_values: int = 50,
        merge_original_and_replacement_features: bool = True, verbose: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    if not standardize_dependent_variable:
        dependent_variable = df.pop(dependent_variable_name)

    columns_to_drop = (list({"sentence", "neg_sentence", "pos_triplet", "neg_triplet", "clip prediction",
                            "clip_score_diff", "pos_clip_score", "neg_clip_score"} - {dependent_variable_name})
                       + [c for c in df.columns if "-common-" in c])
    df = df.drop(columns=list(columns_to_drop))

    if verbose:
        feature_count = len(df.columns)
        if standardize_dependent_variable:
            feature_count -= 1
        print("Number of features before transforming them into numerical:", feature_count)

    transformers: MutableSequence[Tuple] = []

    for column_name in df.columns:
        column = df[column_name]
        impute_missing_values = column_name != dependent_variable_name
        if transformer := _infer_transformer(column, impute_missing_values=impute_missing_values,
                                             standardize_binary_features=standardize_binary_features,
                                             feature_min_non_zero_values=feature_min_non_zero_values):
            selector = column_name if is_feature_multi_label(column) else [column_name]
            transformers.append((selector, transformer))

    considered_column_names = {c for t in transformers for c in (t[0] if isinstance(t[0], list) else [t[0]])}
    if ignored_column_names := set(df.columns) - considered_column_names:
        print("Columns ignored because their type is unsupported:", ignored_column_names)

    mapper = DataFrameMapper(transformers, df_out=True)

    print("Transforming the features into numbers…", end="")
    new_df = mapper.fit_transform(df)
    new_df = _fix_column_names(new_df, mapper)
    print(" ✓")

    if standardize_dependent_variable:
        dependent_variable = new_df.pop(dependent_variable_name)

    if merge_original_and_replacement_features:
        new_columns = {}
        columns_to_remove = []

        multi_label_original_word_feature_names = [t[0]
                                                   for t in transformers
                                                   if isinstance(t[0], str) and t[0].endswith("-original")]

        for column in new_df.columns:
            if column.startswith(multi_label_original_word_feature_names):
                prefix = column.split("-", maxsplit=1)[0]
                suffix = column.split("_", maxsplit=1)[1]

                replacement_column_name = f"{prefix}-replacement_{suffix}"
                if replacement_column_name in new_df.columns:
                    # FIXME: this calculation should be done before standardization, otherwise it's wrong.
                    new_columns[f"{prefix}-change_{suffix}"] = new_df[column] - new_df[replacement_column_name]
                    columns_to_remove.append(column)
                    columns_to_remove.append(replacement_column_name)

        # FIXME: these "change" columns should also be standardized if the other ones also were standardized.

        # Change them all together to avoid DataFrame fragmentation.
        new_df = new_df.drop(columns_to_remove, axis="columns")
        new_df = pd.concat((new_df, pd.DataFrame.from_dict(new_columns)), axis="columns")

    return new_df, dependent_variable  # noqa


def _describe_features(features: pd.DataFrame, dependent_variable: np.ndarray) -> None:
    # FIXME: the main feature names doesn't work well. It should only split the binarized ones.
    main_feature_names = [feature_name.split("_")[0] for feature_name in features.columns]
    print(f"Features size:", len(features.columns), "--", Counter(main_feature_names))
    print(f"Features shape:", features.shape)

    if not np.issubdtype(dependent_variable.dtype, np.floating):
        print(f"Dependent variable value counts:", Counter(dependent_variable))


def _compute_numeric_features(clip_results: pd.DataFrame, dependent_variable_name: str,
                              max_feature_count: int | None = None, feature_deny_list: Collection[str] = frozenset(),
                              compute_neg_features: bool = True, compute_similarity_features: bool = True,
                              merge_original_and_replacement_features: bool = True,
                              feature_min_non_zero_values: int = 50, standardize_dependent_variable: bool = True,
                              standardize_binary_features: bool = True,
                              verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    raw_features = _compute_features(clip_results, feature_deny_list=feature_deny_list,
                                     max_feature_count=max_feature_count, compute_neg_features=compute_neg_features,
                                     compute_similarity_features=compute_similarity_features)
    features, dependent_variable = _transform_features_to_numbers(
        raw_features, dependent_variable_name, standardize_dependent_variable=standardize_dependent_variable,
        standardize_binary_features=standardize_binary_features,
        feature_min_non_zero_values=feature_min_non_zero_values,
        merge_original_and_replacement_features=merge_original_and_replacement_features)

    if verbose:
        _describe_features(features, dependent_variable)

    return raw_features, features, dependent_variable


def load_features(path: str, dependent_variable_name: str, max_feature_count: int | None = None,
                  feature_deny_list: Collection[str] = frozenset(), standardize_dependent_variable: bool = True,
                  standardize_binary_features: bool = True, compute_neg_features: bool = True,
                  compute_similarity_features: bool = True, merge_original_and_replacement_features: bool = True,
                  feature_min_non_zero_values: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    clip_results = _load_clip_results(path)
    return _compute_numeric_features(
        clip_results, dependent_variable_name, max_feature_count=max_feature_count, feature_deny_list=feature_deny_list,
        standardize_dependent_variable=standardize_dependent_variable,
        standardize_binary_features=standardize_binary_features, compute_neg_features=compute_neg_features,
        compute_similarity_features=compute_similarity_features,
        merge_original_and_replacement_features=merge_original_and_replacement_features,
        feature_min_non_zero_values=feature_min_non_zero_values)


def is_feature_binary(feature: np.ndarray | pd.Series) -> bool:
    return feature.dtype == bool or (np.issubdtype(feature.dtype, np.integer) and set(np.unique(feature)) == {0, 1})


def is_feature_multi_label(feature: np.ndarray | pd.Series) -> bool:
    return all(issubclass(type(x), Iterable) and not issubclass(type(x), str) for x in feature)


def is_feature_string(feature: np.ndarray | pd.Series) -> bool:
    return all(issubclass(type(x), str) for x in feature)
