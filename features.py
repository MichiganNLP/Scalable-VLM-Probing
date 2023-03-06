import ast
import functools
import itertools
import json
import string
from collections import Counter, defaultdict
from typing import Any, Container, Dict, Iterable, Literal, Mapping, Optional, Sequence, Set, Tuple, get_args

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
def _parse_liwc_file(path: str = "data/LIWC.2015.all.txt") -> Tuple[Mapping[str, Sequence[str]], Set[str]]:
    dict_liwc = defaultdict(list)
    liwc_categories = set()
    with open(path) as file:
        for line in file:
            word, category = (w.strip() for w in line.strip().split(","))
            dict_liwc[word].append(category)
            liwc_categories.add(category)
    return dict_liwc, liwc_categories


@functools.lru_cache
def _parse_concreteness_file(path: str = "data/concreteness.txt") -> Mapping[str, float]:
    dict_concreteness = {}
    with open(path) as file:
        next(file)  # Skip the first line.
        for line in file:
            word, _, concreteness_m, _, _, _, _, _, _ = line.split("	")
            dict_concreteness[word] = float(concreteness_m)
    return dict_concreteness


def _get_levin_category(word: str, dict_levin_semantic: Mapping[str, Container[str]]) -> Sequence[str]:
    return [category
            for category, category_words in dict_levin_semantic.items()
            if word in category_words]


def _get_liwc_category(word: str, dict_liwc: Mapping[str, Sequence[str]]) -> Sequence[str]:
    return [category
            for key_word, categories in dict_liwc.items()
            if key_word == word or (key_word[-1] == "*" and word.startswith(key_word[:-1]))
            for category in categories]


def _neg_type_to_pos(neg_type: NegType) -> Literal["n", "v"]:
    return "v" if neg_type == "v" else "n"  # noqa


def _compute_wup_similarity(word_original: str, word_replacement: str, neg_type: NegType) -> float:
    pos = _neg_type_to_pos(neg_type)
    return max((synset_original.wup_similarity(synset_replacement)
                for synset_original in wordnet.synsets(word_original, pos=pos)
                for synset_replacement in wordnet.synsets(word_replacement, pos=pos)),
               default=float("nan"))


def _get_concreteness_score(word: str, dict_concreteness: Mapping[str, float]) -> float:
    return dict_concreteness.get(word, float("nan"))


@functools.lru_cache
def _parse_levin_file(
        path: str = "data/levin_verbs.txt") -> Tuple[Mapping[str, Sequence[str]], Mapping[str, Sequence[str]]]:
    content = ""
    levin_dict = {}
    compressed_levin_dict = defaultdict(list)
    with open(path) as file:
        for line in file:
            line = line.lstrip()
            if line and line[0].isnumeric():
                key = " ".join(line.split())
                key_compressed = key.split(" ")[0].split(".")[0]
            else:
                if line:
                    content += line.replace("\r\n", "").rstrip()
                    content += " "
                else:
                    if "-*-" not in content:
                        levin_dict[key] = [x.lower() for x in content.split()]
                        compressed_levin_dict[key_compressed].extend(levin_dict[key])
                    content = ""
        if "-*-" not in content:
            levin_dict[key] = [x.lower() for x in content.split()]
            compressed_levin_dict[key_compressed].extend(levin_dict[key])
    return levin_dict, compressed_levin_dict


def _parse_levin_dict(
        levin_dict: Mapping[str, Sequence[str]],
        path: str = "data/levin_semantic_broad.json",
) -> Tuple[Mapping[str, Container[str]], Mapping[str, Sequence[str]], Mapping[str, Sequence[str]],
           Mapping[str, Sequence[str]]]:
    with open(path) as file:
        map_int_to_name = {int(k): v for k, v in json.load(file).items()}

    levin_semantic_broad = defaultdict(set)
    levin_semantic_fine_grained, levin_alternations, levin_all = {}, {}, {}
    for key, value in levin_dict.items():
        int_key = int(key.split(" ", maxsplit=1)[0].split(".", maxsplit=1)[0])
        levin_all[key] = value
        if int_key <= 8:
            levin_alternations[key] = value
        else:
            levin_semantic_fine_grained[key] = value
            name_key = map_int_to_name[int_key]
            levin_semantic_broad[name_key].update(value)
    return levin_semantic_broad, levin_semantic_fine_grained, levin_alternations, levin_all


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

    mapper = DataFrameMapper([
        # (["neg_type"], OneHotEncoder(dtype=bool)),
        ("Levin-original", MultiLabelBinarizer()),
        ("Levin-replacement", MultiLabelBinarizer()),
        ("LIWC-original", MultiLabelBinarizer()),
        ("LIWC-replacement", MultiLabelBinarizer()),
        (["concreteness-change"], [SimpleImputer(), StandardScaler()]),
        (["text_similarity"], [SimpleImputer(), StandardScaler()]),
        (["word_similarity"], [SimpleImputer(), StandardScaler()]),
        # (["wup_similarity"], [SimpleImputer(), StandardScaler()]),
    ], df_out=True)

    new_df = mapper.fit_transform(df)
    new_df = _fix_one_hot_encoder_columns(new_df, mapper)

    if merge_original_and_replacement_features:
        new_columns = {}
        columns_to_remove = []

        for column in new_df.columns:
            if column.startswith(("Levin-original", "LIWC-original")):
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
                      max_feature_count: Optional[int] = None) -> Tuple[pd.DataFrame, Sequence[int]]:
    if max_feature_count:
        clip_results = clip_results[:max_feature_count]

    dict_features: Dict[str, Any] = {"word_original": [], "word_replacement": [], "neg_type": [],
                                     "Levin-original": [], "Levin-replacement": [], "LIWC-original": [],
                                     "LIWC-replacement": [], "concreteness-original": [],
                                     "concreteness-replacement": [], "wup_similarity": []}

    levin_dict, _ = _parse_levin_file()
    _, _, _, levin_all = _parse_levin_dict(levin_dict)
    dict_liwc, _ = _parse_liwc_file()
    dict_concreteness = _parse_concreteness_file()

    sentences = clip_results.sentence.array
    negative_sentences = clip_results.neg_sentence.array

    dict_features["sent"] = sentences
    dict_features["n_sent"] = negative_sentences

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
            levin_classes_w_original = _get_levin_category(word_original, levin_all)
            levin_classes_w_replacement = _get_levin_category(word_replacement, levin_all)
        else:
            levin_classes_w_original, levin_classes_w_replacement = [], []

        liwc_category_w_original = _get_liwc_category(word_original, dict_liwc)
        liwc_category_w_replacement = _get_liwc_category(word_replacement, dict_liwc)

        concreteness_w_original = _get_concreteness_score(word_original, dict_concreteness)
        concreteness_w_replacement = _get_concreteness_score(word_replacement, dict_concreteness)

        # wup_similarity = compute_wup_similarity(word_original, word_replacement, neg_type=row.neg_type)
        wup_similarity = float("nan")

        dict_features["word_original"].append(word_original)
        dict_features["word_replacement"].append(word_replacement)
        dict_features["neg_type"].append(row.neg_type)
        dict_features["Levin-original"].append(levin_classes_w_original)
        dict_features["Levin-replacement"].append(levin_classes_w_replacement)
        dict_features["LIWC-original"].append(liwc_category_w_original)
        dict_features["LIWC-replacement"].append(liwc_category_w_replacement)
        dict_features["concreteness-original"].append(concreteness_w_original)
        dict_features["concreteness-replacement"].append(concreteness_w_replacement)
        dict_features["wup_similarity"].append(wup_similarity)

    embedded_original_words = text_model.encode(dict_features["word_original"], show_progress_bar=True)
    embedded_replacement_words = text_model.encode(dict_features["word_replacement"], show_progress_bar=True)

    dict_features["word_similarity"] = util.pairwise_cos_sim(embedded_original_words, embedded_replacement_words)

    levin_liwc = [item for sublist in dict_features["Levin-original"] + dict_features["Levin-replacement"] +
                  dict_features["LIWC-original"] + dict_features["LIWC-replacement"] for item in sublist]
    features_count = (levin_liwc + ["neg_type-" + v for v in dict_features["neg_type"]] +
                      ["text_similarity"] * len(dict_features["text_similarity"]) +
                      ["word_similarity"] * len(dict_features["word_similarity"]) +
                      ["wup_similarity"] * len(dict_features["wup_similarity"]) +
                      ["concreteness-original"] * len(dict_features["concreteness-original"]) +
                      ["concreteness-replacement"] * len(dict_features["concreteness-replacement"]))

    return pd.DataFrame.from_dict(dict_features), features_count


def _describe_features(clip_results: pd.DataFrame, features: pd.DataFrame) -> None:
    print("Total classes before filtering:")

    levin_dict, compressed_levin_dict = _parse_levin_file()
    levin_semantic_broad, levin_semantic_all, levin_alternations, levin_all = _parse_levin_dict(levin_dict)
    print(f"--Levin semantic_broad nb classes:", len(levin_semantic_broad.keys()))
    print(f"--Levin semantic_all nb classes:", len(levin_semantic_all.keys()))
    print(f"--Levin alternations nb classes:", len(levin_alternations.keys()))
    print(f"--Levin alternations + semantic_all nb classes:", len(levin_all.keys()))

    liwc_dict, liwc_categories = _parse_liwc_file()
    print(f"LIWC total number of classes:", len(liwc_categories))

    print()
    print("Counts after filtering:")

    main_feature_names = [feature_name.split("_")[0] for feature_name in features.columns]
    print(f"Counter all labels:", Counter(clip_results["clip prediction"].tolist()))
    print(f"Features size:", len(features.columns), "--", Counter(main_feature_names))
    print(f"Features shape:", features.shape)

    print()
    print()


def _compute_numeric_features(clip_results: pd.DataFrame, max_feature_count: Optional[int] = None,
                              merge_original_and_replacement_features: bool = True, do_regression: bool = True,
                              feature_min_non_zero_values: int = 50
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, Sequence[int], np.ndarray]:
    raw_features, features_count = _compute_features(clip_results, max_feature_count=max_feature_count)
    features = _transform_features_to_numbers(
        raw_features, merge_original_and_replacement_features=merge_original_and_replacement_features)
    features = features.loc[:, ((features != 0).sum(0) >= feature_min_non_zero_values)]
    _describe_features(clip_results, features)
    dependent_variable = raw_features["clip-score-diff"] if do_regression else raw_features["label"]
    return raw_features, features, features_count, dependent_variable


def load_features(path: str, max_feature_count: Optional[int] = None,
                  merge_original_and_replacement_features: bool = True, do_regression: bool = True,
                  feature_min_non_zero_values: int = 50,
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, Sequence[int], np.ndarray]:
    clip_results = _load_clip_results(path)
    return _compute_numeric_features(
        clip_results, max_feature_count=max_feature_count,
        merge_original_and_replacement_features=merge_original_and_replacement_features, do_regression=do_regression,
        feature_min_non_zero_values=feature_min_non_zero_values)


def is_feature_binary(feature: np.ndarray) -> bool:
    return feature.dtype == bool or (np.issubdtype(feature.dtype, np.integer) and set(np.unique(feature)) == {0, 1})
