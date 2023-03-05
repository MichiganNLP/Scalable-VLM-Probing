#!/usr/bin/env python
import argparse
import ast
import json
import string
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Any, Container, Dict, Literal, Mapping, Optional, Sequence, Set, Tuple, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn_pandas import DataFrameMapper
from tqdm.auto import tqdm

NegType = Literal["s", "v", "o"]
Triplet = Tuple[str, str, str]
Instance = Tuple[int, str, str, Triplet, Triplet, str, bool, float]

text_model = SentenceTransformer("all-MiniLM-L6-v2")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def parse_triplets(triplets: str) -> Sequence[Triplet]:
    if triplets.startswith("["):
        return [triplet.split(",") for triplet in ast.literal_eval(triplets)]
    else:
        return [triplets.split(",")]  # noqa


def get_sentence_match_triplet(triplets: Sequence[Triplet], sentence: str) -> Triplet:
    if "people" in sentence.split():
        triplets = [(s.replace("person", "people"), v.replace("person", "people"), o.replace("person", "people"))
                    for s, v, o in triplets]

    if len(triplets) == 1:
        return triplets[0]
    else:
        words_sentence = sentence.split()
        lemmatized_words_sentence = [lemmatizer.lemmatize(word) for word in sentence.split()]
        stemmed_words_sentence = [stemmer.stem(word) for word in sentence.split()]
        all_words = set(words_sentence + lemmatized_words_sentence + stemmed_words_sentence)
        for triplet in triplets:
            if triplet[0] in all_words and triplet[1] in all_words and triplet[2] in all_words:
                return triplet

    return triplets[0]


def pre_process_sentences(sentence: str) -> str:
    if type(sentence) == str:
        sentence = sentence.lower()
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    else:
        sentence = ""
    return sentence


def read_data(path: str) -> Sequence[Instance]:
    df = pd.read_csv(path, index_col=0, usecols=["Unnamed: 0", "sentence", "neg_sentence", "pos_triplet", "neg_triplet",
                                                 "neg_type", "clip prediction", "clip_score_diff"]).sort_index()

    # df.sentence = df.sentence.apply(pre_process_sentences)
    # df.neg_sentence = df.neg_sentence.apply(pre_process_sentences)

    results = []
    for index, sentence, neg_sentence, pos_triplet, neg_triplet, neg_type, clip_prediction, clip_score_diff in \
            zip(df.index, df["sentence"], df["neg_sentence"], df["pos_triplet"], df["neg_triplet"], df["neg_type"],
                df["clip prediction"], df["clip_score_diff"]):

        sentence = pre_process_sentences(sentence)  # remove punctuation
        neg_sentence = pre_process_sentences(neg_sentence)

        parsed_pos_triplet = parse_triplets(pos_triplet)
        parsed_neg_triplet = parse_triplets(neg_triplet)
        if not parsed_pos_triplet or not parsed_neg_triplet or not sentence:
            continue

        match_pos_triplet = get_sentence_match_triplet(parsed_pos_triplet, sentence)
        match_neg_triplet = get_sentence_match_triplet(parsed_neg_triplet, neg_sentence)

        results.append(
            (index, sentence, neg_sentence, match_pos_triplet, match_neg_triplet, neg_type[0], clip_prediction,
             clip_score_diff))

    return results


def parse_liwc_file(path: str = "data/LIWC.2015.all.txt") -> Tuple[Mapping[str, Sequence[str]], Set[str]]:
    dict_liwc = defaultdict(list)
    liwc_categories = set()
    with open(path) as file:
        for line in file:
            word, category = (w.strip() for w in line.strip().split(","))
            dict_liwc[word].append(category)
            liwc_categories.add(category)
    return dict_liwc, liwc_categories


def parse_concreteness_file(path: str = "data/concreteness.txt") -> Mapping[str, float]:
    dict_concreteness = {}
    with open(path) as file:
        next(file)  # Skip the first line.
        for line in file:
            word, _, concreteness_m, _, _, _, _, _, _ = line.split("	")
            dict_concreteness[word] = float(concreteness_m)
    return dict_concreteness


def get_levin_category(word: str, dict_levin_semantic: Mapping[str, Container[str]]) -> Sequence[str]:
    return [category
            for category, category_words in dict_levin_semantic.items()
            if word in category_words]


def get_liwc_category(word: str, dict_liwc: Mapping[str, Sequence[str]]) -> Sequence[str]:
    return [category
            for key_word, categories in dict_liwc.items()
            if key_word == word or (key_word[-1] == "*" and word.startswith(key_word[:-1]))
            for category in categories]


def get_wup_similarity(word_original: str, word_replacement: str, neg_type: NegType) -> float:
    pos = "v" if neg_type == "v" else "n"
    return max((synset_original.wup_similarity(synset_replacement)
                for synset_original in wordnet.synsets(word_original, pos=pos)
                for synset_replacement in wordnet.synsets(word_replacement, pos=pos)),
               default=float("nan"))


def get_concreteness_score(word: str, dict_concreteness: Mapping[str, float]) -> float:
    return dict_concreteness.get(word, float("nan"))


def parse_levin_file(
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


def parse_levin_dict(levin_dict: Mapping[str, Sequence[str]],
                     path: str = "data/levin_semantic_broad.json",
                     ) -> Tuple[Mapping[str, Container[str]], Mapping[str, Sequence[str]], Mapping[str, Sequence[str]]]:
    with open(path) as file:
        map_int_to_name = {int(k): v for k, v in json.load(file).items()}

    levin_semantic_broad = defaultdict(set)
    levin_semantic_fine_grained, levin_alternations = {}, {}
    for key, value in levin_dict.items():
        int_key = int(key.split(" ", maxsplit=1)[0].split(".", maxsplit=1)[0])
        if int_key <= 8:
            levin_alternations[key] = value
        else:
            levin_semantic_fine_grained[key] = value
            name_key = map_int_to_name[int_key]
            levin_semantic_broad[name_key].update(value)
    return levin_semantic_broad, levin_semantic_fine_grained, levin_alternations


# sklearn-pandas doesn't support the new way (scikit-learn >= 1.1) some transformers output the features.
# See https://github.com/scikit-learn-contrib/sklearn-pandas/pull/248
def _fix_one_hot_encoder_columns(df: pd.DataFrame, mapper: DataFrameMapper) -> pd.DataFrame:
    for columns, transformer, kwargs in mapper.built_features:
        if isinstance(transformer, OneHotEncoder):
            new_names = transformer.get_feature_names_out(columns)

            if "alias" in kwargs:
                old_name_prefix = kwargs.get("alias")
            elif isinstance(columns, list):
                old_name_prefix = "_".join(str(c) for c in columns)
            else:
                old_name_prefix = columns

            old_names = [f"{old_name_prefix}_{i}" for i in range(len(new_names))]

            df = df.rename(columns=dict(zip(old_names, new_names)))

    return df


def transform_features(df: pd.DataFrame, merge_original_and_replacement: bool = True) -> pd.DataFrame:
    df["concreteness-change"] = df["concreteness-original"] - df["concreteness-replacement"]

    mapper = DataFrameMapper([
        # (["neg_type"], OneHotEncoder()),
        ("Levin-original", MultiLabelBinarizer()),
        ("Levin-replacement", MultiLabelBinarizer()),
        ("LIWC-original", MultiLabelBinarizer()),
        ("LIWC-replacement", MultiLabelBinarizer()),
        (["concreteness-change"], [SimpleImputer(), StandardScaler()]),
        (["text_similarity"], [SimpleImputer(), StandardScaler()]),
        (["word_similarity"], [SimpleImputer(), StandardScaler()]),
        (["wup_similarity"], [SimpleImputer(), StandardScaler()]),
    ], df_out=True)

    new_df = mapper.fit_transform(df)
    new_df = _fix_one_hot_encoder_columns(new_df, mapper)

    if merge_original_and_replacement:
        for column in new_df.columns:
            if column.startswith(("Levin-original", "LIWC-original")):
                prefix = column.split("-", maxsplit=1)[0]
                category = column.split("_", maxsplit=1)[1]

                replacement_column_name = f"{prefix}-replacement_{category}"
                if replacement_column_name in new_df.columns:
                    new_df[f"{prefix}_change_{category}"] = new_df[column] - new_df[replacement_column_name]
                    new_df = new_df.drop([column, replacement_column_name], axis="columns")

    return new_df


def get_original_word(pos_triplet: Triplet, neg_triplet: Triplet, neg_type: NegType) -> Tuple[str, str]:
    if neg_type == "s":
        return pos_triplet[0], neg_triplet[0]
    elif neg_type == "v":
        return pos_triplet[1], neg_triplet[1]
    elif neg_type == "o":
        return pos_triplet[2], neg_triplet[2]
    else:
        raise ValueError(f"Wrong neg_type: {neg_type}, needs to be one from {get_args(NegType)}")


def get_features(clip_results: Sequence[Instance],
                 max_feature_count: Optional[int] = None) -> Tuple[pd.DataFrame, Sequence[int]]:
    if max_feature_count:
        clip_results = clip_results[:max_feature_count]

    dict_features: Dict[str, Any] = {"word_original": [], "word_replacement": [], "neg_type": [],
                                     "Levin-original": [], "Levin-replacement": [], "LIWC-original": [],
                                     "LIWC-replacement": [], "concreteness-original": [],
                                     "concreteness-replacement": [], "wup_similarity": []}

    levin_dict, _ = parse_levin_file()
    levin_semantic_broad, _, _ = parse_levin_dict(levin_dict)
    dict_liwc, _ = parse_liwc_file()
    dict_concreteness = parse_concreteness_file()

    sentences = [x[1] for x in clip_results]
    negative_sentences = [x[2] for x in clip_results]

    dict_features["index"] = [x[0] for x in clip_results]
    dict_features["sent"] = sentences
    dict_features["n_sent"] = negative_sentences

    dict_features["label"] = [int(x[-2] == "pos") for x in clip_results]
    dict_features["clip-score-diff"] = [x[-1] for x in clip_results]

    embedded_sentences = text_model.encode(sentences, show_progress_bar=True)
    embedded_neg_sentences = text_model.encode(negative_sentences, show_progress_bar=True)

    dict_features["text_similarity"] = util.pairwise_cos_sim(embedded_sentences, embedded_neg_sentences)
    # We set the similarity to NaN for empty sentences:
    dict_features["text_similarity"][[s == "" for s in negative_sentences]] = float("nan")

    for _, sentence, neg_sentence, pos_triplet, neg_triplet, neg_type, _, _ in tqdm(
            clip_results, desc="Computing the features"):
        word_original, word_replacement = get_original_word(pos_triplet, neg_triplet, neg_type)

        if not word_replacement or not word_original:
            raise ValueError(f"Found empty word original or word replacement")

        if neg_type == "v":
            levin_classes_w_original = get_levin_category(word_original, levin_semantic_broad)  # TODO: other Levin?
            levin_classes_w_replacement = get_levin_category(word_replacement, levin_semantic_broad)
        else:
            levin_classes_w_original, levin_classes_w_replacement = [], []

        liwc_category_w_original = get_liwc_category(word_original, dict_liwc)
        liwc_category_w_replacement = get_liwc_category(word_replacement, dict_liwc)

        concreteness_w_original = get_concreteness_score(word_original, dict_concreteness)
        concreteness_w_replacement = get_concreteness_score(word_replacement, dict_concreteness)

        wup_similarity = get_wup_similarity(word_original, word_replacement, neg_type=neg_type)

        dict_features["word_original"].append(word_original)
        dict_features["word_replacement"].append(word_replacement)
        dict_features["neg_type"].append(neg_type)
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
    df = pd.DataFrame.from_dict(dict_features)
    return df, features_count


def plot_coef_weights(coef_weights: np.ndarray, feature_names: Sequence[str],
                      path: str = "data/coef_importance.png", top_features: int = 5) -> None:
    coef = coef_weights.ravel()  # flatten array
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    fig = plt.figure(figsize=(18, 7))
    colors = ["red" if c < 0 else "green" for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(2 * top_features), feature_names[top_coefficients], rotation=45, ha="right")
    fig.savefig(path, bbox_inches="tight")


# https://www.kaggle.com/code/pierpaolo28/pima-indians-diabetes-database/notebook
def print_sorted_coef_weights(coef: np.ndarray, coef_significance: np.ndarray, coef_sign: np.ndarray,
                              feature_names: Sequence[str], features_count: Sequence[int],
                              output_path: str = "data/sorted_features.csv") -> None:
    sorted_coefficients_idx = np.argsort(coef)[::-1]  # in descending order
    sorted_coefficients = [np.round(weight, 2) for weight in coef[sorted_coefficients_idx]]

    feature_names = np.array(feature_names)
    sorted_feature_names = feature_names[sorted_coefficients_idx].tolist()
    sorted_feature_significance = coef_significance[sorted_coefficients_idx].tolist()
    sorted_feature_sign = coef_sign[sorted_coefficients_idx].tolist()
    sorted_feature_counts = [features_count.count(feature.split("_")[1]) for feature in sorted_feature_names]

    df = pd.DataFrame(
        zip(sorted_feature_names, sorted_feature_significance, sorted_feature_counts, sorted_coefficients,
            sorted_feature_sign),
        columns=["Feature", "Significance", "Data Count", "Weight (abs)", "Weight sign"])
    df.to_csv(output_path, index=False)


def evaluate(method_name: str, labels_test: np.ndarray, predicted: Sequence[int]) -> None:
    accuracy = accuracy_score(labels_test, predicted) * 100
    precision = precision_score(labels_test, predicted) * 100
    recall = recall_score(labels_test, predicted) * 100
    f1 = f1_score(labels_test, predicted) * 100
    roc_auc = roc_auc_score(labels_test, predicted) * 100
    print(f"Method {method_name}, A: {accuracy:.2f}, P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f},"
          f" ROC-AUC: {roc_auc:.2f}")
    print(f"Counter predicted: {Counter(predicted)}")
    print(f"Counter GT: {Counter(labels_test)}")


def print_metrics(df: pd.DataFrame, feature_names: Sequence[str], features: np.ndarray) -> None:
    main_feature_names = [feature_name.split("_")[0] for feature_name in feature_names]
    print(f"Counter all labels: {Counter(df['label'].tolist())}")
    print(f"Data size: {len(df['index'].tolist())}")
    print(f"Features size: {len(feature_names)}, {Counter(main_feature_names)}")
    print(f"Features shape: {features.shape}")

    levin_dict, compressed_levin_dict = parse_levin_file()
    levin_semantic_broad, levin_semantic_all, levin_alternations = parse_levin_dict(levin_dict)
    print(f"--Levin semantic_broad nb classes: {len(levin_semantic_broad.keys())}")
    print(f"--Levin semantic_all nb classes: {len(levin_semantic_all.keys())}")
    print(f"--Levin alternations nb classes: {len(levin_alternations.keys())}")

    liwc_dict, liwc_categories = parse_liwc_file()
    print(f"LIWC total number of classes: {len(liwc_categories)}")


def process_features(clip_results: Sequence[Instance],
                     max_feature_count: Optional[int] = None,
                     feature_min_non_zero_values: int = 50) -> Tuple[pd.DataFrame, Sequence[int], np.ndarray]:
    df, features_count = get_features(clip_results, max_feature_count=max_feature_count)

    features = transform_features(df)

    features = features.loc[:, ((features != 0).sum(0) >= feature_min_non_zero_values)]

    # print_metrics(df, feature_names, features)
    return features, features_count, df["clip-score-diff"]


def build_classifier() -> svm.LinearSVC:
    return svm.LinearSVC(class_weight="balanced", max_iter=1_000_000)


def classify_shuffled(features: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    features = rng.permuted(features, axis=0)

    clf = build_classifier()
    clf.fit(features, labels)

    return abs(clf.coef_.ravel())


def analyse_coef_weights(features: np.ndarray, labels: np.ndarray,
                         iterations: int = 10_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    clf = build_classifier()

    print("Computing the coefficients with the real features…")
    clf.fit(features, labels)
    print("Coefficients computed.")

    coef_weights = clf.coef_.ravel()
    coef_sign = np.sign(coef_weights)
    coef_weights = abs(coef_weights)

    with Pool() as pool:
        list_shuffled_coef_weights = list(tqdm(
            pool.imap_unordered(partial(classify_shuffled, features, labels), range(iterations)),
            total=iterations, desc="Computing the coefficients with shuffled columns"))

    coef_significance = np.array([sum(list_coef[i] <= coef for list_coef in list_shuffled_coef_weights)
                                  for i, coef in enumerate(coef_weights)])

    return coef_weights, coef_significance, coef_sign


def compute_ols_summary(features: pd.DataFrame, labels: np.ndarray) -> None:
    features = sm.add_constant(features)

    summary = sm.OLS(labels, features).fit().summary()
    print(summary)
    print()
    print()

    table_as_html = summary.tables[1].as_html()
    df = pd.read_html(table_as_html, header=0, index_col=0)[0]

    df = df[df["P>|t|"] <= .05]
    df = df.sort_values(by=["coef"], ascending=False)

    print("Significant features:")
    print(df.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--iterations", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    clip_results = read_data("data/merged.csv")
    features, features_count, labels = process_features(clip_results, max_feature_count=1000 if args.debug else None)
    # coef_weights, coef_significance, coef_sign = analyse_coef_weights(features, labels, args.iterations)
    compute_ols_summary(features, labels)

    # print_sorted_coef_weights(coef_weights, coef_significance, coef_sign, feature_names, features_count)

    # plot_coef_weights(coef_weights, feature_names)


if __name__ == "__main__":
    main()
