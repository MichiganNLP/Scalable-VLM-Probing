import ast
import functools
import itertools
import json
import string
from collections import Counter, defaultdict
from typing import Collection, Iterable, Literal, Mapping, MutableSequence, Optional, Sequence, Tuple, get_args

import numpy as np
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
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
                                    "clip prediction", "clip_score_diff"]).sort_index()

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


@functools.lru_cache
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


def _get_levin_category(word: str, dict_levin: Mapping[str, Collection[str]]) -> Collection[str]:
    return dict_levin.get(word, [])


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


@functools.lru_cache
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


@functools.lru_cache
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


def _get_changed_word(triplet: Triplet, neg_type: NegType) -> str:
    return triplet[VALID_NEG_TYPES.index(neg_type)]


def _compute_features(clip_results: pd.DataFrame, do_regression: bool = True,
                      feature_deny_list: Collection[str] = frozenset(),
                      max_feature_count: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    print("Computing all the features…")

    if max_feature_count:
        clip_results = clip_results[:max_feature_count]

    df = clip_results.copy()

    df["word_original"] = df.apply(lambda row: _get_changed_word(row.pos_triplet, row.neg_type), axis=1)
    df["word_replacement"] = df.apply(lambda row: _get_changed_word(row.neg_triplet, row.neg_type), axis=1)

    if "text_similarity" not in feature_deny_list:
        print("Computing the text similarity…")

        embedded_sentences = text_model.encode(df.sentence.array, show_progress_bar=True)
        embedded_neg_sentences = text_model.encode(df.neg_sentence.array, show_progress_bar=True)

        df["text_similarity"] = util.pairwise_cos_sim(embedded_sentences, embedded_neg_sentences)
        # We set the similarity to NaN for empty sentences:
        df.loc[[s == "" for s in df.neg_sentence], "text_similarity"] = float("nan")

    if "word_similarity" not in feature_deny_list:
        print("Computing the word similarity…")

        embedded_original_words = text_model.encode(df.word_original.array, show_progress_bar=True)
        embedded_replacement_words = text_model.encode(df.word_replacement.array, show_progress_bar=True)

        df["word_similarity"] = util.pairwise_cos_sim(embedded_original_words, embedded_replacement_words)

    if "Levin" not in feature_deny_list:
        dict_levin = _parse_levin_file()

        df["levin-original"] = df.apply(
            lambda row: _get_levin_category(row.word_original, dict_levin) if row.neg_type == "v" else [], axis=1)
        df["levin-replacement"] = df.apply(
            lambda row: _get_levin_category(row.word_replacement, dict_levin) if row.neg_type == "v" else [], axis=1)

    if "LIWC" not in feature_deny_list:
        dict_liwc = _parse_liwc_file()

        df["LIWC-original"] = df.word_original.apply(lambda w: _get_liwc_category(w, dict_liwc))
        df["LIWC-replacement"] = df.word_replacement.apply(lambda w: _get_liwc_category(w, dict_liwc))

    if "hypernym" not in feature_deny_list:
        print("Computing the hypernyms…", end="")
        df["hypernym-original"] = df.apply(lambda row: _get_hypernyms(row.word_original, row.neg_type), axis=1)
        df["hypernym-replacement"] = df.apply(lambda row: _get_hypernyms(row.word_replacement, row.neg_type), axis=1)
        print(" ✓")

    if "frequency" not in feature_deny_list:
        with open(PATH_WORD_FREQUENCIES) as json_file:
            word_frequencies = json.load(json_file)

        df["frequency-original"] = df.word_original.apply(lambda w: word_frequencies.get(w, 0))
        df["frequency-replacement"] = df.word_replacement.apply(lambda w: word_frequencies.get(w, 0))

        df["frequency-change"] = df["frequency-original"] - df["frequency-replacement"]

    if "concreteness" not in feature_deny_list:
        dict_concreteness = _parse_concreteness_file()

        df["concreteness-original"] = df.word_original.apply(
            lambda w: _get_concreteness_score(w, dict_concreteness))
        df["concreteness-replacement"] = df.word_replacement.apply(
            lambda w: _get_concreteness_score(w, dict_concreteness))

        df["concreteness-change"] = df["concreteness-original"] - df["concreteness-replacement"]

    if "wup_similarity" not in feature_deny_list:
        print("Computing the Wu-Palmer similarity…", end="")
        df["wup_similarity"] = df.apply(
            lambda row: _compute_wup_similarity(row.word_original, row.word_replacement, row.neg_type), axis=1)
        print(" ✓")

    if "lch_similarity" not in feature_deny_list:
        print("Computing the Leacock-Chodorow similarity…", end="")
        df["lch_similarity"] = df.apply(
            lambda row: _compute_lch_similarity(row.word_original, row.word_replacement, row.neg_type), axis=1)
        print(" ✓")

    if "path_similarity" not in feature_deny_list:
        print("Computing the Path similarity…", end="")
        df["path_similarity"] = df.apply(
            lambda row: _compute_path_similarity(row.word_original, row.word_replacement, row.neg_type), axis=1)
        print(" ✓")

    if "nb_synsets" not in feature_deny_list:
        print("Computing the number of synsets…", end="")
        df["nb-synsets-original"] = df.apply(lambda row: _get_nb_synsets(row.word_original, row.neg_type), axis=1)
        df["nb-synsets-replacement"] = df.apply(lambda row: _get_nb_synsets(row.word_replacement, row.neg_type), axis=1)
        print(" ✓")

    print("Feature computation done.")

    return df, df["clip_score_diff"] if do_regression else df["clip prediction"]


# sklearn-pandas doesn't support the new way (scikit-learn >= 1.1) some transformers output the features.
# See https://github.com/scikit-learn-contrib/sklearn-pandas/pull/248
def _fix_one_hot_encoder_columns(df: pd.DataFrame, mapper: DataFrameMapper) -> pd.DataFrame:
    for columns, transformer, kwargs in mapper.built_features:
        if isinstance(transformer, OneHotEncoder):
            assert isinstance(columns, Iterable) and not isinstance(columns, str)

            new_names = transformer.get_feature_names_out(columns)

            old_name_prefix = kwargs.get("alias", "_".join(str(c) for c in columns))
            old_names = [f"{old_name_prefix}_{i}" for i in range(len(new_names))]

            df = df.rename(columns=dict(zip(old_names, new_names)))

    return df


def _transform_features_to_numbers(df: pd.DataFrame,
                                   merge_original_and_replacement_features: bool = True) -> pd.DataFrame:
    df = df.drop(columns=["sentence", "neg_sentence", "pos_triplet", "neg_triplet", "neg_type", "word_original",
                          "word_replacement", "clip prediction", "clip_score_diff"])

    transformers: MutableSequence[Tuple] = []

    for column_name in df.columns:
        column = df[column_name]
        dtype = column.dtype
        if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer):
            transformers.append(([column_name], [SimpleImputer(), StandardScaler()]))
        elif dtype == object:
            types = {type(x) for x in column}
            if all(issubclass(t, str) for t in types):
                transformers.append(([column_name], OneHotEncoder(dtype=bool)))
            elif all(issubclass(t, Iterable) and not issubclass(t, str) for t in types):
                transformers.append((column_name, MultiLabelBinarizer()))

    considered_column_names = {c for t in transformers for c in (t[0] if isinstance(t[0], list) else [t[0]])}
    if ignored_column_names := set(df.columns) - considered_column_names:
        print("Columns ignored because their type is unsupported:", ignored_column_names)

    mapper = DataFrameMapper(transformers, df_out=True)

    print("Transforming the features into numbers…", end="")
    new_df = mapper.fit_transform(df)
    new_df = _fix_one_hot_encoder_columns(new_df, mapper)
    print(" ✓")

    if merge_original_and_replacement_features:
        new_columns = {}
        columns_to_remove = []

        multi_label_original_word_feature_names = [t[0]
                                                   for t in transformers
                                                   if (isinstance(t[1], MultiLabelBinarizer)
                                                       and t[0].endswith("-original"))]

        for column in new_df.columns:
            if column.startswith(multi_label_original_word_feature_names):
                prefix = column.split("-", maxsplit=1)[0]
                suffix = column.split("_", maxsplit=1)[1]

                replacement_column_name = f"{prefix}-replacement_{suffix}"
                if replacement_column_name in new_df.columns:
                    new_columns[f"{prefix}-change_{suffix}"] = new_df[column] - new_df[replacement_column_name]
                    columns_to_remove.append(column)
                    columns_to_remove.append(replacement_column_name)

        # Change them all together to avoid DataFrame fragmentation.
        new_df = new_df.drop(columns_to_remove, axis="columns")
        new_df = pd.concat((new_df, pd.DataFrame.from_dict(new_columns)), axis="columns")

    return new_df


def _describe_features(features: pd.DataFrame, dependent_variable: np.ndarray) -> None:
    if not np.issubdtype(dependent_variable.dtype, np.floating):
        print(f"Dependent variable value counts:", Counter(dependent_variable))

    main_feature_names = [feature_name.split("_")[0] for feature_name in features.columns]
    print(f"Features size:", len(features.columns), "--", Counter(main_feature_names))
    print(f"Features shape:", features.shape)


def _compute_numeric_features(clip_results: pd.DataFrame, max_feature_count: Optional[int] = None,
                              feature_deny_list: Collection[str] = frozenset(),
                              merge_original_and_replacement_features: bool = True, do_regression: bool = True,
                              feature_min_non_zero_values: int = 50,
                              verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    raw_features, dependent_variable = _compute_features(clip_results, feature_deny_list=feature_deny_list,
                                                         do_regression=do_regression,
                                                         max_feature_count=max_feature_count)
    features = _transform_features_to_numbers(
        raw_features, merge_original_and_replacement_features=merge_original_and_replacement_features)

    features_mask = (features != 0).sum(0) >= feature_min_non_zero_values
    if (~features_mask).any():  # noqa
        print("The following features are removed because they have too few non-zero values:",
              features.columns[~features_mask])
    features = features.loc[:, features_mask]

    if verbose:
        _describe_features(features, dependent_variable)

    return raw_features, features, dependent_variable


def load_features(path: str, max_feature_count: Optional[int] = None, feature_deny_list: Collection[str] = frozenset(),
                  merge_original_and_replacement_features: bool = True, do_regression: bool = True,
                  feature_min_non_zero_values: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    clip_results = _load_clip_results(path)
    return _compute_numeric_features(
        clip_results, max_feature_count=max_feature_count, feature_deny_list=feature_deny_list,
        merge_original_and_replacement_features=merge_original_and_replacement_features, do_regression=do_regression,
        feature_min_non_zero_values=feature_min_non_zero_values)


def is_feature_binary(feature: np.ndarray) -> bool:
    return feature.dtype == bool or (np.issubdtype(feature.dtype, np.integer) and set(np.unique(feature)) == {0, 1})
