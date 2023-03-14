from __future__ import annotations

import ast
import itertools
import json
import string
import warnings
from collections import Counter, defaultdict
from typing import Any, Callable, Collection, Iterable, Literal, Mapping, MutableSequence, Sequence, Tuple, get_args

import numpy as np
import pandas as pd
import statsmodels.api as sm
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn_pandas import DataFrameMapper
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm.auto import trange

NegType = Literal["s", "v", "o"]
Pos = Literal["n", "v"]
Triplet = Tuple[str, str, str]

LevinReturnMode = Literal["alternation", "semantic_broad", "semantic_fine_grained", "all"]

VALID_NEG_TYPES = get_args(NegType)
VALID_LEVIN_RETURN_MODES = get_args(LevinReturnMode)

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


def _neg_type_to_pos(neg_type: NegType) -> Pos:
    return "v" if neg_type == "v" else "n"  # noqa


def _parse_levin_file(path: str = PATH_LEVIN_VERBS, path_semantic_broad: str = PATH_LEVIN_SEMANTIC_BROAD,
                      return_mode: LevinReturnMode = "all", verbose: bool = True) -> Mapping[str, Collection[str]]:
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
        print(f"--Levin semantic broad number of classes:", len(semantic_broad_classes))
        print(f"--Levin semantic fine-grained number of classes:", semantic_fine_grained_class_count)
        print(f"--Levin alternations number of class:", alternation_class_count)
        print(f"Total number of Levin classes:", len(map_class_name_to_words))

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


def _get_levin_category(word: str, dict_levin: Mapping[str, Collection[str]], pos: Pos) -> Collection[str]:
    return dict_levin.get(word, []) if pos == "v" else []


def _get_nb_synsets(word: str, pos: Pos) -> int:  # noqa
    # We don't use the POS information because we're using this as a proxy of ambiguity.
    return len(wn.synsets(word))


def _get_hypernyms(word: str, pos: Pos) -> Collection[str]:
    if synsets := wn.synsets(word, pos=pos):
        # The first synset is the most likely definition of the word.
        return {hypernym_synset.name() for hypernym_synset in synsets[0].hypernyms()}
    else:
        return []


warnings.filterwarnings("ignore", message="Discarded redundant search for Synset.*")


def _get_indirect_hypernyms(word: str, pos: Pos) -> Collection[str]:
    if synsets := wn.synsets(word, pos=pos):
        # The first synset is the most likely definition of the word.
        return {s.name()
                for hypernym_synset in synsets[0].hypernyms()  # We skip the direct hypernyms.
                for s in hypernym_synset.closure(lambda s: s.hypernyms())}
    else:
        return []


def _get_frequency(word: str, word_frequencies: Mapping[str, int]) -> int:
    return word_frequencies.get(word, 0)


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


def _compute_wup_similarity(word_original: str, word_replacement: str, pos: Pos) -> float:
    return max((synset_original.wup_similarity(synset_replacement)
                for synset_original in wn.synsets(word_original, pos=pos)
                for synset_replacement in wn.synsets(word_replacement, pos=pos)),
               default=float("nan"))


def _compute_lch_similarity(word_original: str, word_replacement: str, pos: Pos) -> float:
    return max((synset_original.lch_similarity(synset_replacement)
                for synset_original in wn.synsets(word_original, pos=pos)
                for synset_replacement in wn.synsets(word_replacement, pos=pos)),
               default=float("nan"))


def _compute_path_similarity(word_original: str, word_replacement: str, pos: Pos) -> float:
    return max((synset_original.path_similarity(synset_replacement)
                for synset_original in wn.synsets(word_original, pos=pos)
                for synset_replacement in wn.synsets(word_replacement, pos=pos)),
               default=float("nan"))


def _neg_type_name_to_index(neg_type: NegType) -> int:
    return VALID_NEG_TYPES.index(neg_type)


def _get_changed_word(triplet: Triplet, neg_type: NegType) -> str:
    return triplet[_neg_type_name_to_index(neg_type)]


def _get_common_words(triplet: Triplet, neg_type: NegType) -> Collection[str]:
    return [t for t, other_neg_type in zip(triplet, VALID_NEG_TYPES) if other_neg_type != neg_type]


def _get_common_words_pos(neg_type: NegType) -> Collection[Pos]:
    return [_neg_type_to_pos(other_neg_type) for other_neg_type in VALID_NEG_TYPES if other_neg_type != neg_type]


def _compute_feature_for_each_word(df: pd.DataFrame, prefix: str, func: Callable[[str, Pos], Any],
                                   compute_neg_features: bool = True) -> None:
    if compute_neg_features:
        df[f"{prefix}-original"] = df.apply(lambda row: func(row["word-original"],
                                                             _neg_type_to_pos(row["neg-type"])), axis=1)
        df[f"{prefix}-replacement"] = df.apply(lambda row: func(row["word-replacement"],
                                                                _neg_type_to_pos(row["neg-type"])), axis=1)

    placeholder_result = pd.Series([func("dog", "n")])

    common_results = df.apply(lambda row: [func(w, pos)
                                           for w, pos in zip(row["words-common"], row["words-common-pos"])], axis=1)

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
                      max_data_count: int | None = None, compute_neg_features: bool = True,
                      levin_return_mode: LevinReturnMode = "all",
                      compute_similarity_features: bool = True) -> pd.DataFrame:
    print("Computing all the features…")

    if max_data_count:
        clip_results = clip_results.sample(min(max_data_count, len(clip_results)))

    df = clip_results.copy()

    # We use the underscore to separate a feature name from its value if it's binarized.
    df = df.rename(columns={"neg_type": "neg-type"})

    if compute_neg_features:
        df["word-original"] = df.apply(lambda row: _get_changed_word(row.pos_triplet, row["neg-type"]), axis=1)
        df["word-replacement"] = df.apply(lambda row: _get_changed_word(row.neg_triplet, row["neg-type"]), axis=1)
        df["words-common"] = df.apply(lambda row: _get_common_words(row.pos_triplet, row["neg-type"]), axis=1)
        df["words-common-pos"] = df["neg-type"].map(_get_common_words_pos)
    else:
        df["words-common"] = df.pos_triplet
        df["words-common-pos"] = [[_neg_type_to_pos(neg_type) for neg_type in VALID_NEG_TYPES]] * len(df)

    for i in range(len(df["words-common"].iloc[0])):
        df[f"words-common-{i}"] = df["words-common"].str[i]

    if "Levin" not in feature_deny_list:
        dict_levin = _parse_levin_file(return_mode=levin_return_mode)
        _compute_feature_for_each_word(df, "Levin", lambda w, pos: _get_levin_category(w, dict_levin, pos),
                                       compute_neg_features=compute_neg_features)

    if "LIWC" not in feature_deny_list:
        dict_liwc = _parse_liwc_file()
        _compute_feature_for_each_word(df, "LIWC", lambda w, _: _get_liwc_category(w, dict_liwc),
                                       compute_neg_features=compute_neg_features)

    if "hypernym" not in feature_deny_list:
        print("Computing the hypernyms…", end="")
        _compute_feature_for_each_word(df, "hypernym", _get_hypernyms, compute_neg_features=compute_neg_features)
        print(" ✓")

    if "hypernym/indirect" not in feature_deny_list:
        print("Computing the indirect hypernyms…", end="")
        _compute_feature_for_each_word(df, "hypernym/indirect", _get_indirect_hypernyms,
                                       compute_neg_features=compute_neg_features)
        print(" ✓")

    if "frequency" not in feature_deny_list:
        with open(PATH_WORD_FREQUENCIES) as json_file:
            word_frequencies = json.load(json_file)
        _compute_feature_for_each_word(df, "frequency", lambda w, _: _get_frequency(w, word_frequencies),
                                       compute_neg_features=compute_neg_features)

    if "concreteness" not in feature_deny_list:
        dict_concreteness = _parse_concreteness_file()
        _compute_feature_for_each_word(df, "concreteness", lambda w, _: _get_concreteness_score(w, dict_concreteness),
                                       compute_neg_features=compute_neg_features)

    if "nb-synsets" not in feature_deny_list:
        print("Computing the number of synsets…", end="")
        _compute_feature_for_each_word(df, "nb-synsets", _get_nb_synsets, compute_neg_features=compute_neg_features)
        print(" ✓")

    if compute_similarity_features and compute_neg_features:
        if "text-similarity" not in feature_deny_list:
            print("Computing the text similarity…")

            embedded_sentences = text_model.encode(df.sentence.array, show_progress_bar=True)
            embedded_neg_sentences = text_model.encode(df.neg_sentence.array, show_progress_bar=True)

            df["text-similarity"] = util.pairwise_cos_sim(embedded_sentences, embedded_neg_sentences)
            # We set the similarity to NaN for empty sentences:
            df.loc[[s == "" for s in df.neg_sentence], "text-similarity"] = float("nan")

        if "word-similarity" not in feature_deny_list:
            print("Computing the word similarity…")

            embedded_original_words = text_model.encode(df["word-original"].array, show_progress_bar=True)
            embedded_replacement_words = text_model.encode(df["word-replacement"].array, show_progress_bar=True)

            df["word-similarity"] = util.pairwise_cos_sim(embedded_original_words, embedded_replacement_words)

        if "wup-similarity" not in feature_deny_list:
            print("Computing the Wu-Palmer similarity…", end="")
            df["wup-similarity"] = df.apply(
                lambda row: _compute_wup_similarity(row["word-original"], row["word-replacement"], row["neg-type"]),
                axis=1)
            print(" ✓")

        if "lch-similarity" not in feature_deny_list:
            print("Computing the Leacock-Chodorow similarity…", end="")
            df["lch-similarity"] = df.apply(
                lambda row: _compute_lch_similarity(row["word-original"], row["word-replacement"], row["neg-type"]),
                axis=1)
            print(" ✓")

        if "path-similarity" not in feature_deny_list:
            print("Computing the Path similarity…", end="")
            df["path-similarity"] = df.apply(
                lambda row: _compute_path_similarity(row["word-original"], row["word-replacement"], row["neg-type"]),
                axis=1)
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
    transformers = None  # `None` in this context means to ignore the feature. Note it's different from `[]`.

    if is_feature_binary(feature):
        transformers = []
    elif np.issubdtype(feature.dtype, np.number):
        transformers = ([SimpleImputer()] if impute_missing_values else []) + [StandardScaler()]  # noqa
    elif is_feature_string(feature):
        transformers = [OneHotEncoder(dtype=bool, sparse_output=not standardize_binary_features)]
    elif is_feature_multi_label(feature):
        transformers = [MultiLabelBinarizer()]

    if transformers is not None:
        transformers.append(VarianceThreshold())  # We remove the constant features.

    if is_feature_binary(feature) or is_feature_multi_label(feature) or is_feature_string(feature):
        transformers.append(SelectMinNonZero(feature_min_non_zero_values))

        if standardize_binary_features:
            transformers.append(StandardScaler())

    return transformers


def _transform_features_to_numbers(
        df: pd.DataFrame, dependent_variable_name: str, standardize_dependent_variable: bool = True,
        standardize_binary_features: bool = True, feature_min_non_zero_values: int = 50,
        compute_neg_features: bool = True, merge_original_and_replacement_features: bool = True,
        add_constant_feature: bool = False, verbose: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    if not standardize_dependent_variable:
        dependent_variable = df.pop(dependent_variable_name)

    columns_to_drop = (list({"sentence", "neg_sentence", "neg-type", "pos_triplet", "neg_triplet", "clip prediction",
                             "clip_score_diff", "pos_clip_score", "neg_clip_score"} - {dependent_variable_name}))
    if compute_neg_features:
        columns_to_drop += [c for c in df.columns if "-common-" in c]  # These don't make sense for the negatives.
    df = df.drop(columns=list(columns_to_drop))

    if verbose:
        feature_count = len(df.columns)
        if standardize_dependent_variable:
            feature_count -= 1
        print("Number of features before the transformation:", feature_count)

    # We have to manually remove the multi-label constant features because otherwise it fails in `VarianceThreshold`
    # if used in the transformer mapper as some binarized multi-label features may have no sub-features left.
    # We can't use `VarianceThreshold` directly here because it only supports numerical features.
    # We don't check for all types of features here because it's a bit complicated to do it in a generic way.
    if empty_multi_label_feature_names := [c
                                           for c in df
                                           if (is_feature_multi_label(df[c])
                                               and (df[c].transform(tuple) == tuple(df[c][0])).all())]:
        print("Multi-label features ignored because they are constant:", empty_multi_label_feature_names)
        df = df.drop(columns=empty_multi_label_feature_names)

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

    if add_constant_feature:
        new_df = sm.add_constant(new_df)

    print("Number of features after the transformation:", len(new_df.columns))

    return new_df, dependent_variable  # noqa


def _describe_features(features: pd.DataFrame, dependent_variable: pd.Series) -> None:
    main_feature_names = [feature_name.split("_", maxsplit=1)[0] for feature_name in features.columns]
    print(f"Features size:", len(features.columns), "--", Counter(main_feature_names))
    print(f"Features shape:", features.shape)

    if not np.issubdtype(dependent_variable.dtype, np.floating):
        print(f"Dependent variable value counts:", Counter(dependent_variable))


def _compute_numeric_features(clip_results: pd.DataFrame, dependent_variable_name: str,
                              max_data_count: int | None = None, feature_deny_list: Collection[str] = frozenset(),
                              compute_neg_features: bool = True, levin_return_mode: LevinReturnMode = "all",
                              compute_similarity_features: bool = True,
                              merge_original_and_replacement_features: bool = True,
                              add_constant_feature: bool = False, feature_min_non_zero_values: int = 50,
                              standardize_dependent_variable: bool = True, standardize_binary_features: bool = True,
                              verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw_features = _compute_features(clip_results, feature_deny_list=feature_deny_list,
                                     max_data_count=max_data_count, compute_neg_features=compute_neg_features,
                                     levin_return_mode=levin_return_mode,
                                     compute_similarity_features=compute_similarity_features)
    features, dependent_variable = _transform_features_to_numbers(
        raw_features, dependent_variable_name, standardize_dependent_variable=standardize_dependent_variable,
        standardize_binary_features=standardize_binary_features,
        feature_min_non_zero_values=feature_min_non_zero_values, compute_neg_features=compute_neg_features,
        merge_original_and_replacement_features=merge_original_and_replacement_features,
        add_constant_feature=add_constant_feature, verbose=verbose)

    return raw_features, features, dependent_variable


def load_features(path: str, dependent_variable_name: str, max_data_count: int | None = None,
                  feature_deny_list: Collection[str] = frozenset(), standardize_dependent_variable: bool = True,
                  standardize_binary_features: bool = True, compute_neg_features: bool = True,
                  levin_return_mode: LevinReturnMode = "all", compute_similarity_features: bool = True,
                  merge_original_and_replacement_features: bool = True, add_constant_feature: bool = False,
                  remove_correlated_features: bool = True, feature_correlation_keep_threshold: float = .8,
                  do_vif: bool = False, feature_min_non_zero_values: int = 50,
                  verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    clip_results = _load_clip_results(path)
    raw_features, features, dependent_variable = _compute_numeric_features(
        clip_results, dependent_variable_name, max_data_count=max_data_count, feature_deny_list=feature_deny_list,
        standardize_dependent_variable=standardize_dependent_variable,
        standardize_binary_features=standardize_binary_features, compute_neg_features=compute_neg_features,
        levin_return_mode=levin_return_mode, compute_similarity_features=compute_similarity_features,
        merge_original_and_replacement_features=merge_original_and_replacement_features,
        add_constant_feature=add_constant_feature, feature_min_non_zero_values=feature_min_non_zero_values,
        verbose=verbose)

    if remove_correlated_features:
        print("Computing the feature correlation matrix…", end="")
        # TODO: a chi-squared test would be better for binary data. But it should be done before standardization.
        # From: https://stackoverflow.com/a/52509954/1165181
        corr_matrix = features.corr().abs()
        print(" ✓")
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= feature_correlation_keep_threshold)]
        print("The following", len(to_drop), "features are correlated and will be removed:", to_drop)
        features.drop(to_drop, axis="columns", inplace=True)
        print("Number of features after the removal of correlated features:", len(features.columns))

        if do_vif:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide.*")
                with trange(len(features.columns), desc="Removing correlated features") as progress_bar:
                    for _ in progress_bar:
                        c_max_vif, max_vif = max(((c, variance_inflation_factor(new_df.values, i))
                                                  for i, c in enumerate(new_df.columns)),
                                                 key=lambda t: t[1])
                        if max_vif < 5:
                            break
                        new_df = new_df.drop(c_max_vif, axis="columns")
                        progress_bar.set_postfix_str(f"Removed {c_max_vif} (VIF={max_vif:.2f})")
            print(f"Final largest VIF ({max_vif}) comes from {c_max_vif}.")
            print("Number of features after the removal of correlated features based on VIF:", len(new_df.columns))

    if verbose:
        _describe_features(features, dependent_variable)

    return raw_features, features, dependent_variable


def is_feature_binary(feature: np.ndarray | pd.Series) -> bool:
    return feature.dtype == bool or (np.issubdtype(feature.dtype, np.number) and set(np.unique(feature)) == {0, 1})


def is_feature_multi_label(feature: np.ndarray | pd.Series) -> bool:
    x = feature[0]  # We suppose the first one is representative to make it faster.
    return issubclass(type(x), Iterable) and not issubclass(type(x), str)


def is_feature_string(feature: np.ndarray | pd.Series) -> bool:
    x = feature[0]  # We suppose the first one is representative to make it faster.
    return issubclass(type(x), str)
