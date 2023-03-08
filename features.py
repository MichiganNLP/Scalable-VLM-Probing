import ast
import functools
import itertools
import json
import string
from collections import Counter, defaultdict
from typing import Any, Collection, Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple, get_args

import numpy as np
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn_pandas import DataFrameMapper
from tqdm.auto import tqdm

NegType = Literal["s", "v", "o"]
Triplet = Tuple[str, str, str]

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
    assert df.neg_type.isin(get_args(NegType)).all()

    df["clip prediction"] = df["clip prediction"] == "pos"

    return df


@functools.lru_cache
def _parse_levin_file(path: str = PATH_LEVIN_VERBS,
                      path_semantic_broad: str = PATH_LEVIN_SEMANTIC_BROAD,
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

    # map_word_to_semantic_broad_class_names = defaultdict(set)
    # map_word_to_semantic_fine_grained_class_names = defaultdict(set)
    # map_word_to_alternation_class_names = defaultdict(set)
    map_word_to_class_names = defaultdict(set)

    semantic_broad_classes = set()
    semantic_fine_grained_class_count = 0
    alternation_class_count = 0

    for class_name, words in map_class_name_to_words.items():
        for word in words:
            map_word_to_class_names[word].add(class_name)

        if (class_number := int(class_name.split(" ", maxsplit=1)[0].split(".", maxsplit=1)[0])) <= 8:
            alternation_class_count += 1

            # for word in words:
            #     map_word_to_alternation_class_names[word].add(class_name)
        else:
            semantic_fine_grained_class_count += 1

            broad_class_name = map_class_number_to_broad_name[class_number]
            semantic_broad_classes.add(broad_class_name)

            # for word in words:
            #     map_word_to_semantic_fine_grained_class_names[word].add(class_name)
            #     map_word_to_semantic_broad_class_names[broad_class_name].update(words)

    if verbose:
        print(f"--Levin semantic broad nb classes:", len(semantic_broad_classes))
        print(f"--Levin semantic fine-grained nb classes:", semantic_fine_grained_class_count)
        print(f"--Levin alternations nb classes:", alternation_class_count)
        print(f"--Levin total nb classes:", len(map_class_name_to_words))

    return map_word_to_class_names


def _get_levin_category(word: str, dict_levin: Mapping[str, Collection[str]]) -> Collection[str]:
    return dict_levin[word]


def _get_nb_synsets(word: str, neg_type: NegType) -> int:  # noqa
    # pos = _neg_type_to_pos(neg_type)
    # synsets = wordnet.synsets(word, pos=pos)
    synsets = wordnet.synsets(word)
    return len(synsets)


def _get_hypernyms(word: str, neg_type: NegType) -> Sequence[str]:
    pos = _neg_type_to_pos(neg_type)
    if not (synsets := wordnet.synsets(word, pos=pos)):
        return "nan"  # FIXME: or `return [word]`?
    synsets = synsets[0].hypernyms() or synsets
    broad_semantic_category = synsets[0]._lemma_names[0]
    return [broad_semantic_category]


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


def _get_original_word(pos_triplet: Triplet, neg_triplet: Triplet, neg_type: NegType) -> Tuple[str, str]:
    if neg_type == "s":
        return pos_triplet[0], neg_triplet[0]
    elif neg_type == "v":
        return pos_triplet[1], neg_triplet[1]
    elif neg_type == "o":
        return pos_triplet[2], neg_triplet[2]
    else:
        raise ValueError(f"Wrong neg_type: {neg_type}, needs to be one from {get_args(NegType)}")


def _compute_features(clip_results: pd.DataFrame,
                      max_feature_count: Optional[int] = None) -> pd.DataFrame:
    if max_feature_count:
        clip_results = clip_results[:max_feature_count]

    dict_features: Dict[str, Any] = {"word_original": [], "word_replacement": [],
                                     "Levin-original": [], "Levin-replacement": [],
                                     "LIWC-original": [], "LIWC-replacement": [],
                                     "hypernym-original": [], "hypernym-replacement": [],
                                     "frequency-original": [], "frequency-replacement": [],
                                     "concreteness-original": [], "concreteness-replacement": [],
                                     "nb-synsets-original": [], "nb-synsets-replacement": [], "text_similarity": [],
                                     "wup_similarity": [], "lch_similarity": [], "path_similarity": []}

    dict_levin = _parse_levin_file()
    dict_liwc = _parse_liwc_file()
    dict_concreteness = _parse_concreteness_file()

    with open(PATH_WORD_FREQUENCIES) as json_file:
        word_frequencies = json.load(json_file)

    sentences = clip_results.sentence.array
    negative_sentences = clip_results.neg_sentence.array

    dict_features["sent"] = sentences
    dict_features["n_sent"] = negative_sentences

    dict_features["neg_type"] = clip_results.neg_type

    dict_features["label"] = clip_results["clip prediction"]
    dict_features["clip-score-diff"] = clip_results.clip_score_diff

    embedded_sentences = text_model.encode(sentences, show_progress_bar=True)
    embedded_neg_sentences = text_model.encode(negative_sentences, show_progress_bar=True)

    dict_features["text_similarity"] = util.pairwise_cos_sim(embedded_sentences, embedded_neg_sentences)
    # We set the similarity to NaN for empty sentences:
    dict_features["text_similarity"][[s == "" for s in negative_sentences]] = float("nan")

    for _, row in tqdm(clip_results.iterrows(), desc="Computing the features", total=len(clip_results)):
        word_original, word_replacement = _get_original_word(row.pos_triplet, row.neg_triplet, row.neg_type)

        if not word_replacement or not word_original:
            raise ValueError(f"Found empty word original or word replacement")

        if row.neg_type == "v":
            levin_classes_w_original = _get_levin_category(word_original, dict_levin)
            levin_classes_w_replacement = _get_levin_category(word_replacement, dict_levin)
        else:
            levin_classes_w_original, levin_classes_w_replacement = [], []

        liwc_category_w_original = _get_liwc_category(word_original, dict_liwc)
        liwc_category_w_replacement = _get_liwc_category(word_replacement, dict_liwc)

        frequency_w_original = word_frequencies.get(word_original, 0)
        frequency_w_replacement = word_frequencies.get(word_replacement, 0)

        concreteness_w_original = _get_concreteness_score(word_original, dict_concreteness)
        concreteness_w_replacement = _get_concreteness_score(word_replacement, dict_concreteness)

        # wup_similarity = _compute_wup_similarity(word_original, word_replacement, neg_type=row.neg_type)
        lch_similarity = _compute_lch_similarity(word_original, word_replacement, neg_type=row.neg_type)
        # path_similarity = _compute_path_similarity(word_original, word_replacement, neg_type=row.neg_type)
        wup_similarity, path_similarity = float("nan"), float("nan")
        nb_synsets_word_original = _get_nb_synsets(word_original, row.neg_type)
        nb_synsets_word_replacement = _get_nb_synsets(word_replacement, row.neg_type)

        hypernym_original = _get_hypernyms(word_original, row.neg_type)
        hypernym_replacement = _get_hypernyms(word_replacement, row.neg_type)

        dict_features["word_original"].append(word_original)
        dict_features["word_replacement"].append(word_replacement)
        dict_features["Levin-original"].append(levin_classes_w_original)
        dict_features["Levin-replacement"].append(levin_classes_w_replacement)
        dict_features["LIWC-original"].append(liwc_category_w_original)
        dict_features["LIWC-replacement"].append(liwc_category_w_replacement)
        dict_features["hypernym-original"].append(hypernym_original)
        dict_features["hypernym-replacement"].append(hypernym_replacement)
        dict_features["frequency-original"].append(frequency_w_original)
        dict_features["frequency-replacement"].append(frequency_w_replacement)
        dict_features["concreteness-original"].append(concreteness_w_original)
        dict_features["concreteness-replacement"].append(concreteness_w_replacement)
        dict_features["nb-synsets-original"].append(nb_synsets_word_original)
        dict_features["nb-synsets-replacement"].append(nb_synsets_word_replacement)
        dict_features["wup_similarity"].append(wup_similarity)
        dict_features["lch_similarity"].append(lch_similarity)
        dict_features["path_similarity"].append(path_similarity)

    embedded_original_words = text_model.encode(dict_features["word_original"], show_progress_bar=True)
    embedded_replacement_words = text_model.encode(dict_features["word_replacement"], show_progress_bar=True)

    dict_features["word_similarity"] = util.pairwise_cos_sim(embedded_original_words, embedded_replacement_words)

    return pd.DataFrame.from_dict(dict_features)


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
    df["concreteness-change"] = df["concreteness-original"] - df["concreteness-replacement"]
    df["frequency-change"] = df["frequency-original"] - df["frequency-replacement"]

    mapper = DataFrameMapper([
        # (["neg_type"], OneHotEncoder(dtype=bool)),
        ("Levin-original", MultiLabelBinarizer()),
        ("Levin-replacement", MultiLabelBinarizer()),
        ("LIWC-original", MultiLabelBinarizer()),
        ("LIWC-replacement", MultiLabelBinarizer()),
        ("hypernym-original", MultiLabelBinarizer()),
        ("hypernym-replacement", MultiLabelBinarizer()),
        (["concreteness-change"], [SimpleImputer(), StandardScaler()]),
        (["concreteness-original"], [SimpleImputer(), StandardScaler()]),
        (["concreteness-replacement"], [SimpleImputer(), StandardScaler()]),
        (["frequency-change"], [SimpleImputer(), StandardScaler()]),
        (["frequency-original"], [SimpleImputer(), StandardScaler()]),
        (["frequency-replacement"], [SimpleImputer(), StandardScaler()]),
        (["text_similarity"], [SimpleImputer(), StandardScaler()]),
        (["word_similarity"], [SimpleImputer(), StandardScaler()]),
        (["nb-synsets-original"], [SimpleImputer(), StandardScaler()]),
        (["nb-synsets-replacement"], [SimpleImputer(), StandardScaler()]),
        # (["wup_similarity"], [SimpleImputer(), StandardScaler()]),
        (["lch_similarity"], [SimpleImputer(), StandardScaler()]),
        # (["path_similarity"], [SimpleImputer(), StandardScaler()]),
    ], df_out=True)

    new_df = mapper.fit_transform(df)
    new_df = _fix_one_hot_encoder_columns(new_df, mapper)

    if merge_original_and_replacement_features:
        new_columns = {}
        columns_to_remove = []

        for column in new_df.columns:
            if column.startswith(("Levin-original", "LIWC-original", "hypernym-original")):
                prefix = column.split("-", maxsplit=1)[0]
                category = column.split("_", maxsplit=1)[1]

                replacement_column_name = f"{prefix}-replacement_{category}"
                if replacement_column_name in new_df.columns:
                    new_columns[f"{prefix}_change_{category}"] = new_df[column] - new_df[replacement_column_name]
                    columns_to_remove.append(column)
                    columns_to_remove.append(replacement_column_name)

        # Change them all together to avoid fragmentation.
        new_df = new_df.drop(columns_to_remove, axis="columns")
        new_df = pd.concat((new_df, pd.DataFrame.from_dict(new_columns)), axis="columns")

    return new_df


def _describe_features(clip_results: pd.DataFrame, features: pd.DataFrame) -> None:
    main_feature_names = [feature_name.split("_")[0] for feature_name in features.columns]
    print(f"Counter all labels:", Counter(clip_results["clip prediction"].tolist()))
    print(f"Features size:", len(features.columns), "--", Counter(main_feature_names))
    print(f"Features shape:", features.shape)


def _compute_numeric_features(clip_results: pd.DataFrame, max_feature_count: Optional[int] = None,
                              merge_original_and_replacement_features: bool = True, do_regression: bool = True,
                              feature_min_non_zero_values: int = 50
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    raw_features = _compute_features(clip_results, max_feature_count=max_feature_count)
    features = _transform_features_to_numbers(
        raw_features, merge_original_and_replacement_features=merge_original_and_replacement_features)
    features = features.loc[:, ((features != 0).sum(0) >= feature_min_non_zero_values)]
    _describe_features(clip_results, features)
    dependent_variable = raw_features["clip-score-diff"] if do_regression else raw_features["label"]
    return raw_features, features, dependent_variable


def load_features(path: str, max_feature_count: Optional[int] = None,
                  merge_original_and_replacement_features: bool = True, do_regression: bool = True,
                  feature_min_non_zero_values: int = 50,
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    clip_results = _load_clip_results(path)
    return _compute_numeric_features(
        clip_results, max_feature_count=max_feature_count,
        merge_original_and_replacement_features=merge_original_and_replacement_features, do_regression=do_regression,
        feature_min_non_zero_values=feature_min_non_zero_values)


def is_feature_binary(feature: np.ndarray) -> bool:
    return feature.dtype == bool or (np.issubdtype(feature.dtype, np.integer) and set(np.unique(feature)) == {0, 1})
