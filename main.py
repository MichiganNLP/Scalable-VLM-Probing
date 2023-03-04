#!/usr/bin/env python
import argparse
import ast
import itertools
import json
import string
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Container, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from word_forms.word_forms import get_word_forms

Triplet = Tuple[str, str, str]
Instance = Tuple[int, str, str, Triplet, Triplet, str, bool, float]

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_model = AutoModel.from_pretrained(model_name)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def parse_triplets(triplets: str) -> Sequence[Triplet]:
    if triplets.startswith("["):
        return [triplet.split(",") for triplet in ast.literal_eval(triplets)]
    else:
        return [triplets.split(",")]  # noqa


def get_first_triplet(triplets: Sequence[Triplet]) -> Triplet:
    return next(iter(triplets), ("", "", ""))


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


def read_data(path: str = "data/merged.csv") -> Sequence[Instance]:
    df = pd.read_csv(path, index_col=0)
    df = df.sort_index()
    df["index"] = df.index
    results = []
    for index, sentence, neg_sentence, pos_triplet, neg_triplet, neg_type, clip_prediction, clip_score_diff in \
            zip(df["index"], df["sentence"], df["neg_sentence"], df["pos_triplet"], df["neg_triplet"], df["neg_type"],
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
            word, category = [w.strip() for w in line.strip().split(",")]
            dict_liwc[word].append(category)
            liwc_categories.add(category)
    return dict_liwc, liwc_categories


def parse_concreteness_file(path: str = "data/concreteness.txt") -> Mapping[str, float]:
    dict_concreteness = {}
    with open(path) as file:
        next(file)  # Skip the first line.
        for line in file:
            word, _, concreteness_m, _, _, _, _, _, _ = line.split("	")
            dict_concreteness[word] = round(float(concreteness_m))
    return dict_concreteness


def get_levin_category(word: str, dict_levin_semantic: Mapping[str, Container[str]]) -> Sequence[str]:
    return [category
            for category, category_words in dict_levin_semantic.items()
            if word in category_words]


def get_liwc_category(word: str, dict_liwc: Mapping[str, Sequence[str]]) -> Sequence[str]:
    return [category
            for key_word, categories in dict_liwc.items()
            if key_word == word or (key_word[-1] == "*" and key_word[:-1] in word)
            for category in categories]


def get_wup_similarity(word_original: str, word_replacement: str, pos: str) -> float:
    if pos == "s" or pos == "o":
        pos = "n"  # TODO: it might have other types of words different from nouns.

    try:
        syn1 = wordnet.synset(".".join([word_original, pos, "01"]))
    except wordnet.WordNetError:
        syn1 = wordnet.synsets(word_original)[0]

    try:
        syn2 = wordnet.synset(".".join([word_replacement, pos, "01"]))
    except wordnet.WordNetError:
        syn2 = wordnet.synsets(word_replacement)[0]

    return syn1.wup_similarity(syn2)


def compute_embedding(word_type: str, sentence: str) -> np.ndarray:
    if sentence:
        inputs = tokenizer(sentence, return_tensors="pt")
        map_word_token_idx = {x: tokenizer.encode(x, add_special_tokens=False) for x in sentence.split()}
        stemmed_map_word_token_idx = {stemmer.stem(word): map_word_token_idx[word]
                                      for word in map_word_token_idx.keys()}
        token_ids = []
        if word_type in stemmed_map_word_token_idx:
            token_ids = stemmed_map_word_token_idx[word_type]
        elif word_type in map_word_token_idx:
            token_ids = map_word_token_idx[word_type]
        else:
            all_word_forms = get_word_forms(word_type)["v"]
            for word in all_word_forms:
                if word in map_word_token_idx:
                    token_ids = map_word_token_idx[word]
                    break
        if token_ids:
            index_word = [inputs["input_ids"].tolist()[0].index(token_id) for token_id in token_ids]
        else:  # treat as separate word / not part of a sentence
            inputs = tokenizer(word_type, return_tensors="pt")
            index_word = [1]
    else:  # treat as separate word / not part of a sentence
        inputs = tokenizer(word_type, return_tensors="pt")
        index_word = [1]

    with torch.inference_mode():
        outputs = text_model(**inputs)

    embedding_word = np.mean(outputs.last_hidden_state.detach().numpy()[0][index_word[0]:index_word[-1] + 1], axis=0)
    return np.expand_dims(embedding_word, axis=0)


def save_bert_embeddings(list_words: Sequence[str], path: str = "data/bert_embeddings.npy") -> None:
    word_embeddings = {}
    for word in list_words:
        inputs = tokenizer(word, return_tensors="pt")
        with torch.inference_mode():
            outputs = text_model(**inputs)
        embedding_word = outputs.last_hidden_state.detach().numpy()[0][1]
        embedding_word = np.expand_dims(embedding_word, axis=0)
        word_embeddings[word] = embedding_word

    np.save(path, word_embeddings)


def save_bert_embeddings_sentences(list_word_sentence: Sequence[Tuple[str, str]],
                                   path: str = "data/bert_embeddings_sentences.npy") -> None:
    dict_data = {}
    for word, sentence in tqdm(list_word_sentence, desc="Computing BERT embeddings"):
        dict_data[(word, sentence)] = compute_embedding(word, sentence)
    np.save(path, dict_data)


def get_cosine_similarity_sent(word_original: str, word_replacement: str, sentence: str, neg_sentence: str,
                               bert_embeddings: Mapping[Tuple[str, str], np.ndarray]) -> float:
    embedding_word_original = bert_embeddings[(word_original, sentence)]
    embedding_word_replacement = bert_embeddings[(word_replacement, neg_sentence)]
    return cosine_similarity(embedding_word_original, embedding_word_replacement)[0][0]


def get_cosine_similarity(word_original: str, word_replacement: str,
                          bert_embeddings: Mapping[str, np.ndarray]) -> float:
    embedding_word_original = bert_embeddings[word_original]
    embedding_word_replacement = bert_embeddings[word_replacement]
    return cosine_similarity(embedding_word_original, embedding_word_replacement)[0][0]


def get_concreteness_score(word: str, dict_concreteness: Mapping[str, float]) -> float:
    return dict_concreteness.get(word, 3)  # 3 is the mean of all the scores, to not influence the results.


def parse_levin_file(path: str = "data/levin_verbs.txt") -> Tuple[Mapping[str, Sequence[str]],
                                                                  Mapping[str, Sequence[str]]]:
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
                     path: str = "data/levin_semantic_broad.json") -> Tuple[Mapping[str, Container[str]],
                                                                            Mapping[str, Sequence[str]],
                                                                            Mapping[str, Sequence[str]]]:
    with open(path) as file:
        map_int_to_name = {int(k): v for k, v in json.load(file).items()}

    levin_semantic_broad = defaultdict(set)
    levin_semantic_all, levin_alternations = {}, {}
    for key, value in levin_dict.items():
        int_key = int(key.split(" ", maxsplit=1)[0].split(".", maxsplit=1)[0])
        if int_key <= 8:
            levin_alternations[key] = value
        else:
            levin_semantic_all[key] = value
            name_key = map_int_to_name[int_key]
            levin_semantic_broad[name_key].update(value)
    return levin_semantic_broad, levin_semantic_all, levin_alternations


def transform_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                  Sequence[str], Sequence[str]]:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
    transformed = one_hot_encoder.fit_transform(df[["POS"]])
    one_hot_pos = transformed.toarray()
    pos_category = one_hot_encoder.categories_[0].tolist()

    binarizer = MultiLabelBinarizer()
    df_concat = pd.concat([df["Levin-original"], df["Levin-replacement"]])
    one_hot_levin = binarizer.fit_transform(df_concat)
    one_hot_levin_original, one_hot_levin_replacement = np.split(one_hot_levin, 2)
    one_hot_levin_change = one_hot_levin_original - one_hot_levin_replacement
    # one_hot_levin_change[one_hot_levin_change == 2] = 0  # only represent change
    levin_classes = binarizer.classes_.tolist()

    binarizer = MultiLabelBinarizer()
    df_concat = pd.concat([df["LIWC-original"], df["LIWC-replacement"]])
    one_hot_liwc = binarizer.fit_transform(df_concat)
    one_hot_liwc_original, one_hot_liwc_replacement = np.split(one_hot_liwc, 2)
    one_hot_liwc_change = one_hot_liwc_original - one_hot_liwc_replacement
    # one_hot_liwc_change[one_hot_liwc_change == 2] = 0  # only represent change
    liwc_classes = binarizer.classes_.tolist()

    concreteness_w_original = df["concreteness-original"].to_numpy()
    # concreteness_w_original = np.expand_dims(concreteness_w_original, axis=1)
    concreteness_w_replacement = df["concreteness-replacement"].to_numpy()
    concreteness_w_change = concreteness_w_original - concreteness_w_replacement
    concreteness_w_change = np.expand_dims(concreteness_w_change, axis=1)

    cosine_sim = df["cosine-sim"].to_numpy()
    cosine_sim = np.expand_dims(cosine_sim, axis=1)

    feat_categorical_names = ["POS_" + str(i) for i in pos_category] + \
                             ["Levin_" + str(i) for i in levin_classes] + \
                             ["LIWC_" + str(i) for i in liwc_classes]
    feat_continuous_names = ["concreteness_", "cosine-sim_"]

    return (one_hot_pos, one_hot_levin_change, one_hot_liwc_change, concreteness_w_change, cosine_sim,
            feat_categorical_names, feat_continuous_names)


def get_original_word(pos_triplet: Triplet, neg_triplet: Triplet, neg_type: str) -> Tuple[str, str]:
    if neg_type == "s":
        return pos_triplet[0], neg_triplet[0]
    elif neg_type == "v":
        return pos_triplet[1], neg_triplet[1]
    elif neg_type == "o":
        return pos_triplet[2], neg_triplet[2]
    else:
        raise ValueError(f"Wrong neg_type: {neg_type}, needs to be from s,v,o")


def get_bert_data(clip_results: Sequence[Instance]) -> None:
    list_word_sentence, set_words = [], set()
    for _, sentence, neg_sentence, pos_triplet, neg_triplet, neg_type, _, _ in clip_results:
        word_original, word_replacement = get_original_word(pos_triplet, neg_triplet, neg_type)
        if not word_replacement or not word_original:
            continue
        if (word_original, sentence) not in list_word_sentence:
            list_word_sentence.append((word_original, sentence))
        if (word_replacement, neg_sentence) not in list_word_sentence:
            list_word_sentence.append((word_replacement, neg_sentence))
        set_words.add(word_replacement)
        set_words.add(word_original)
    print(len(list_word_sentence))
    save_bert_embeddings_sentences(list_word_sentence)


def get_features(clip_results: Sequence[Instance], bert_embeddings_path: str = "data/bert_embeddings.npy",  # noqa
                 max_feature_count: Optional[int] = None) -> Tuple[pd.DataFrame, Sequence[int]]:
    dict_features = {"index": [], "sent": [], "n_sent": [], "word_original": [], "word_replacement": [], "POS": [],
                     "Levin-original": [], "Levin-replacement": [], "LIWC-original": [], "LIWC-replacement": [],
                     "concreteness-original": [], "concreteness-replacement": [], "cosine-sim": [], "label": [],
                     "clip-score-diff": []}

    levin_dict, compressed_levin_dict = parse_levin_file()
    levin_semantic_broad, levin_semantic_all, levin_alternations = parse_levin_dict(levin_dict)
    dict_liwc, _ = parse_liwc_file()
    dict_concreteness = parse_concreteness_file()
    # bert_embeddings = np.load(path, allow_pickle=True).item()  # transform from ndarray to dict

    if max_feature_count:
        clip_results = clip_results[:max_feature_count]

    for index, sentence, neg_sentence, pos_triplet, neg_triplet, neg_type, clip_prediction, clip_score_diff in tqdm(
            clip_results, desc="Computing the features"):
        word_original, word_replacement = get_original_word(pos_triplet, neg_triplet, neg_type)

        if not word_replacement or not word_original:
            print(f"Found empty word original or word replacement in index {index} not processing data and continue…")
            continue

        cosine_sim = np.random.randn()  # TODO: fix the BERT embeddings file to have all words.
        #                                  Also, use something that actually supports cosine similarity, such as S-BERT.
        # cosine_sim = get_cosine_similarity(word_original, word_replacement, bert_embeddings)  # noqa

        if neg_type == "v":  # TODO: How []/ No Levin or LIWC class influence results
            levin_classes_w_original = get_levin_category(word_original, levin_semantic_broad)
            levin_classes_w_replacement = get_levin_category(word_replacement, levin_semantic_broad)
        else:
            levin_classes_w_original, levin_classes_w_replacement = [], []

        liwc_category_w_original = get_liwc_category(word_original, dict_liwc)
        liwc_category_w_replacement = get_liwc_category(word_replacement, dict_liwc)

        concreteness_w_original = get_concreteness_score(word_original, dict_concreteness)
        concreteness_w_replacement = get_concreteness_score(word_replacement, dict_concreteness)

        dict_features["index"].append(index)
        dict_features["sent"].append(sentence)
        dict_features["n_sent"].append(neg_sentence)
        dict_features["word_original"].append(word_original)
        dict_features["word_replacement"].append(word_replacement)
        dict_features["POS"].append(neg_type)
        # dict_features["Levin"].append(levin_classes_w_original)
        dict_features["Levin-original"].append(levin_classes_w_original)
        dict_features["Levin-replacement"].append(levin_classes_w_replacement)
        # dict_features["LIWC"].append(liwc_category_w_original)
        dict_features["LIWC-original"].append(liwc_category_w_original)
        dict_features["LIWC-replacement"].append(liwc_category_w_replacement)
        # dict_features["concreteness"].append(concreteness_score_w_original)
        dict_features["concreteness-original"].append(concreteness_w_original)
        dict_features["concreteness-replacement"].append(concreteness_w_replacement)
        dict_features["cosine-sim"].append(cosine_sim)

        # TODO: predict when CLIP result is pos or neg? - I think we want to learn when CLIP fails
        dict_features["label"].append(int(clip_prediction != "pos"))
        dict_features["clip-score-diff"].append(clip_score_diff)

    levin_liwc = [item for sublist in dict_features["Levin-original"] + dict_features["Levin-replacement"] +
                  dict_features["LIWC-original"] + dict_features["LIWC-replacement"] for item in sublist]
    features_count = (levin_liwc + ["POS-" + v for v in dict_features["POS"]] +
                      ["cosine-sim"] * len(dict_features["cosine-sim"]) +
                      ["concreteness-original"] * len(dict_features["concreteness-original"]) +
                      ["concreteness-replacement"] * len(dict_features["concreteness-replacement"]))
    df = pd.DataFrame.from_dict(dict_features)
    return df, features_count


def pre_process_features(one_hot_pos: np.ndarray, one_hot_levin: np.ndarray, one_hot_liwc: np.ndarray,
                         concreteness_diff: np.ndarray, cosine_sim: np.ndarray, feat_categorical_names: Sequence[str],
                         feat_continuous_names: Sequence[str]) -> Tuple[np.ndarray, Sequence[str]]:
    feat_categorical = np.concatenate((one_hot_pos, one_hot_levin, one_hot_liwc), axis=1)
    print(f"categorical feature size: {feat_categorical.shape}")

    feat_continuous = np.concatenate((concreteness_diff, cosine_sim), axis=1)
    feat_continuous_scaled = preprocessing.StandardScaler().fit_transform(feat_continuous)
    print(f"continuous feature scaled size: {feat_continuous_scaled.shape}")

    # concatenate categorical (one hot) with scaled continuous features
    features = np.concatenate((feat_categorical, feat_continuous_scaled), axis=1)
    feature_names = list(itertools.chain(feat_categorical_names, feat_continuous_names))

    print(f"all (categorical + continuous) features size: {features.shape}")
    return features, feature_names


def plot_coef_weights(coef_weights: np.ndarray, feature_names: Sequence[str],
                      path: str = "data/coef_importance.png") -> None:
    top_features = 5
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


def merge_csvs_and_filter_data(probes_path: str = "data/svo_probes.csv", neg_path: str = "data/neg_d.csv",
                               output_path: str = "data/merged.csv") -> None:
    df_probes = pd.read_csv(probes_path, index_col="index")

    df_probes.drop(df_probes.index[df_probes["sentence"] == "woman, ball, outside"], replacement=True)
    df_probes.drop(df_probes.index[df_probes["sentence"] == "woman, music, notes"], replacement=True)

    df_neg = pd.read_csv(neg_path, header=0)
    df_neg.drop(df_neg.index[df_neg["neg_sentence"] == "woman, ball, outside"], replacement=True)
    df_neg.drop(df_neg.index[df_neg["neg_sentence"] == "woman, music, notes"], replacement=True)

    result = pd.concat([df_probes, df_neg["neg_sentence"]], axis=1)
    result = result[result["sentence"].notna()]

    result.to_csv(output_path)


def delete_multiple_element(list_object: List[int], indices: Sequence[int]) -> None:
    indices = sorted(indices, reverse=True)
    for i in indices:
        if i < len(list_object):
            list_object.pop(i)


def process_features(clip_results: Sequence[Instance],
                     max_feature_count: Optional[int] = None) -> Tuple[np.ndarray, Sequence[str], Sequence[int],
                                                                       np.ndarray]:
    df, features_count = get_features(clip_results, max_feature_count=max_feature_count)
    # labels = df["label"].to_numpy()
    labels = df["clip-score-diff"].to_numpy()

    (one_hot_pos, one_hot_levin, one_hot_liwc, concreteness_diff, cosine_sim, feat_categorical_names,
     feat_continuous_names) = transform_features(df)

    # categorical feature selection, standardize (scale) continuous features and concatenate categorical & continuous
    features, feature_names = pre_process_features(one_hot_pos, one_hot_levin, one_hot_liwc, concreteness_diff,
                                                   cosine_sim, feat_categorical_names, feat_continuous_names)

    print_metrics(df, feature_names, features)
    return features, feature_names, features_count, labels


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


def compute_ols_summary(features: np.ndarray, feature_names: Sequence[str], labels: np.ndarray) -> None:
    df_features = pd.DataFrame(features, columns=feature_names)

    df_features = df_features.drop(["POS_o", "POS_s", "POS_v"], axis="columns")
    df_features = sm.add_constant(df_features)

    print(sm.OLS(labels, df_features).fit().summary())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--iterations", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # # merge_csvs_and_filter_data()
    #
    clip_results = read_data()
    features, feature_names, features_count, labels = process_features(clip_results,
                                                                       max_feature_count=1000 if args.debug else None)
    # coef_weights, coef_significance, coef_sign = analyse_coef_weights(features, labels, args.iterations)
    compute_ols_summary(features, feature_names, labels)

    # print_sorted_coef_weights(coef_weights, coef_significance, coef_sign, feature_names, features_count)

    # plot_coef_weights(coef_weights, feature_names)


if __name__ == "__main__":
    main()
