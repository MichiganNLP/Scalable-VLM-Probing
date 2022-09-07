#!/usr/bin/env python
import argparse
import ast
import json
import math
import string
from collections import Counter
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn import preprocessing, svm
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from word_forms.word_forms import get_word_forms

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def parse_triplets(triplets: str) -> Sequence[Tuple[str, str, str]]:
    if triplets.startswith("["):
        return [triplet.split(",") for triplet in ast.literal_eval(triplets)]
    else:
        return [triplets.split(",")]  # noqa


def get_first_triplet(triplets: Sequence[Tuple[str, str, str]]):
    return next(iter(triplets), ("", "", ""))


def get_sentence_match_triplet(triplets: Sequence[Tuple[str, str, str]], sentence: str):
    if "people" in sentence.split():
        # print(sentence, triplets)
        triplets = [list(map(lambda x: x.replace('person', 'people'), t)) for t in triplets]
        # print(sentence, triplets)
        # print("----------------")

    if len(triplets) == 1:
        return triplets[0]
    else:
        words_sentence = sentence.split()
        lemmatized_words_sentence = [lemmatizer.lemmatize(word) for word in sentence.split()]
        stemmed_words_sentence = [ps.stem(word) for word in sentence.split()]
        all_words = set(words_sentence + lemmatized_words_sentence + stemmed_words_sentence)
        for triplet in triplets:
            if triplet[0] in all_words and triplet[1] in all_words and triplet[2] in all_words:
                return triplet

    # print(f"ERROR: triplets: {triplets}; sentence: {sentence}; all words: {all_words}")
    return triplets[0][0]


def pre_process_sentences(sentence: str) -> str:
    if type(sentence) == str:
        sentence = sentence.lower()
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    else:
        sentence = ""
    return sentence


def read_data(path: str = "data/merged.csv") -> Sequence[Tuple[int, str, str, Tuple[str, str, str],
                                                               Tuple[str, str, str, bool, float], str]]:
    # df = pd.read_csv("data/svo_probes.csv", index_col='index')
    df = pd.read_csv(path, index_col=0)
    df = df.sort_index()
    df['index'] = df.index
    results = []
    for index, sentence, neg_sentence, pos_triplet, neg_triplet, neg_type, clip_prediction, clip_score_diff in \
            zip(df['index'],
                df['sentence'],
                df['neg_sentence'],
                df['pos_triplet'],
                df['neg_triplet'],
                df['neg_type'],
                df['clip prediction'],
                df['clip_score_diff']):

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


def parse_liwc_file(path: str = 'data/LIWC.2015.all.txt') -> Tuple[Dict[str, Sequence[str]], Set[str]]:
    dict_liwc = {}
    liwc_categ = set()
    with open(path) as file:
        for line in file:
            word, category = [w.strip() for w in line.strip().split(",")]
            if word not in dict_liwc:
                dict_liwc[word] = []
            dict_liwc[word].append(category)
            liwc_categ.add(category)
    return dict_liwc, liwc_categ


def parse_concreteness_file(path: str = 'data/concretness.txt') -> Dict[str, float]:
    dict_concreteness = {}
    with open(path) as file:
        lines = file.readlines()
    for line in lines[1:]:
        word, _, conc_m, _, _, _, _, _, _ = line.split("	")
        dict_concreteness[word] = round(float(conc_m))
    return dict_concreteness


# def get_levin_category(word, dict_levin_semantic, dict_levin_alternations):
def get_levin_category(word: str, dict_levin_semantic: Dict[str, Sequence[str]]) -> Sequence[str]:
    list_categories = []
    for key, category in dict_levin_semantic.items():
        if word in category:
            list_categories.append(key)
    # for key, category in dict_levin_alternations.items():
    #     if word in category:
    #         list_categories.append(key)
    # if not list_categories:
    #     print(f"{word} not found in Levin file")
    return list_categories


def get_liwc_category(word: str, dict_liwc: Dict[str, Sequence[str]]) -> Sequence[str]:
    list_categories = []
    for key, category in dict_liwc.items():
        if key == word:
            list_categories.append(category)
        elif key[-1] == "*" and key[:-1] in word:
            list_categories.append(category)

    list_categories = [x for xs in list_categories for x in xs]
    # if not list_categories:
    #     print(f"{word} not found in LIWC file")
    return list_categories


# def get_wnet_category(word, pos):
#     for word, pos in zip(["stand", "fry", "cook"], ["v", "v", "v"]):
#         if pos == 's' or pos == 'o':
#             pos = 'n'  # TODO: might have other than nouns
#         synset = wordnet.synset(".".join([word, pos, '01']))
#         hypernyms = synset.hypernyms()  # gives other synsets
#         lexname = hypernyms[0].lexname()
#         name = hypernyms[0].name()
#     # return hypernyms

def get_wup_similarity(word_changed: str, word_inplace: str, pos: str) -> float:
    if pos == 's' or pos == 'o':
        pos = 'n'  # TODO: might have other than nouns
    try:
        syn1 = wordnet.synset(".".join([word_changed, pos, '01']))
    except:
        syn1 = wordnet.synsets(word_changed)[0]
    try:
        syn2 = wordnet.synset(".".join([word_inplace, pos, '01']))
    except:
        syn2 = wordnet.synsets(word_inplace)[0]

    wup_similarity_value = syn1.wup_similarity(syn2)
    return wup_similarity_value


def compute_embedding(word_type: str, sentence: str) -> np.ndarray:
    if sentence:
        inputs = tokenizer(sentence, return_tensors="pt")
        map_word_token_idx = {x: tokenizer.encode(x, add_special_tokens=False) for x in sentence.split()}
        stemmed_map_word_token_idx = {ps.stem(word): map_word_token_idx[word] for word in map_word_token_idx.keys()}
        token_ids = []
        if word_type in stemmed_map_word_token_idx:
            token_ids = stemmed_map_word_token_idx[word_type]
        elif word_type in map_word_token_idx:
            token_ids = map_word_token_idx[word_type]
        else:
            all_word_forms = get_word_forms(word_type)['v']
            for word in all_word_forms:
                if word in map_word_token_idx:
                    token_ids = map_word_token_idx[word]
                    break
        if token_ids:
            index_word = [inputs['input_ids'].tolist()[0].index(token_id) for token_id in token_ids]
        else:  # treat as separate word / not part of a sentence
            inputs = tokenizer(word_type, return_tensors="pt")
            index_word = [1]
    else:  # treat as separate word / not part of a sentence
        inputs = tokenizer(word_type, return_tensors="pt")
        index_word = [1]

    with torch.no_grad():
        outputs = model(**inputs)

    embedding_word = np.mean(outputs.last_hidden_state.detach().numpy()[0][index_word[0]:index_word[-1] + 1], axis=0)
    embedding_word = np.expand_dims(embedding_word, axis=0)
    return embedding_word


def save_bert_embeddings(list_words: Sequence[str], path: str = "data/bert_embeddings.npy") -> None:
    word_embeddings = {}
    for word in list_words:
        inputs = tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embedding_word = outputs.last_hidden_state.detach().numpy()[0][1]
        embedding_word = np.expand_dims(embedding_word, axis=0)
        word_embeddings[word] = embedding_word

    np.save(path, word_embeddings)


def save_bert_embeddings_sentences(list_word_sentence: Sequence[Tuple[str, str]],
                                   path: str = "data/bert_embeddings_sentences.npy") -> None:
    dict_data = {}
    for word, sentence in tqdm(list_word_sentence, desc="Computing BERT embeddings"):
        embedding_word = compute_embedding(word, sentence)
        dict_data[(word, sentence)] = embedding_word
    np.save(path, dict_data)


def get_cosine_similarity_sent(word_changed: str, word_inplace: str, sentence: str, neg_sentence: str,
                               bert_embeddings: Dict[Tuple[str, str], np.ndarray]) -> float:
    embedding_word_changed = bert_embeddings[(word_changed, sentence)]
    embedding_word_inplace = bert_embeddings[(word_inplace, neg_sentence)]
    return cosine_similarity(embedding_word_changed, embedding_word_inplace)[0][0]


def get_cosine_similarity(word_changed: str, word_inplace: str, bert_embeddings: Dict[str, np.ndarray]) -> float:
    embedding_word_changed = bert_embeddings[word_changed]
    embedding_word_inplace = bert_embeddings[word_inplace]
    return cosine_similarity(embedding_word_changed, embedding_word_inplace)[0][0]


def get_concreteness_score(word: str, dict_concreteness: Dict[str, float]) -> float:
    # 3 is the mean of all the scores, to not influence the results.
    return dict_concreteness.get(word, 3)


def parse_levin_file(path: str = 'data/levin_verbs.txt') -> Tuple[Dict[str, Sequence[str]], Dict[str, Sequence[str]]]:
    content = ""
    levin_dict, compressed_levin_dict = {}, {}
    with open(path) as file:
        for line in file:
            line = line.lstrip()
            if line and line[0].isnumeric():
                key = " ".join(line.split())
                key_compressed = key.split(" ")[0].split(".")[0]
                if key_compressed not in compressed_levin_dict:
                    compressed_levin_dict[key_compressed] = []
            else:
                if line:
                    content += line.replace('\r\n', "").rstrip()
                    content += " "
                else:
                    if '-*-' not in content:
                        levin_dict[key] = [x.lower() for x in content.split()]
                        for k in levin_dict[key]:
                            compressed_levin_dict[key_compressed].append(k)
                    content = ""
        if '-*-' not in content:
            levin_dict[key] = [x.lower() for x in content.split()]
            for k in levin_dict[key]:
                compressed_levin_dict[key_compressed].append(k)
    return levin_dict, compressed_levin_dict


def parse_levin_dict(levin_dict: Dict[str, Sequence[str]],
                     path: str = 'data/levin_semantic_broad.json') -> Tuple[Dict[str, Sequence[str]],
                                                                            Dict[str, Sequence[str]],
                                                                            Dict[str, Sequence[str]]]:
    with open(path) as file:
        map_int_to_name = json.load(file)

    levin_semantic_broad, levin_semantic_all, levin_alternations = {}, {}, {}
    for key, value in levin_dict.items():
        int_key = int(key.split(" ", maxsplit=1)[0].split(".", maxsplit=1)[0])
        if int_key <= 8:
            levin_alternations[key] = value
        else:
            levin_semantic_all[key] = value
            name_key = map_int_to_name[str(int_key)]
            if name_key not in levin_semantic_broad:
                levin_semantic_broad[name_key] = []
            for v in value:
                if v not in levin_semantic_broad[name_key]:
                    levin_semantic_broad[name_key].append(v)
    return levin_semantic_broad, levin_semantic_all, levin_alternations


def transform_features(df: pd.DataFrame) -> Tuple:
    ohe = OneHotEncoder(handle_unknown="ignore")
    transformed = ohe.fit_transform(df[['POS']])
    one_hot_pos = transformed.toarray()
    pos_categ = ohe.categories_[0].tolist()

    mlb = MultiLabelBinarizer()
    df_concat = pd.concat([df['Levin-change'], df['Levin-inplace']])
    one_hot_levin = mlb.fit_transform(df_concat)
    one_hot_levin_change, one_hot_levin_inplace = np.split(one_hot_levin, 2)
    levin_classes = mlb.classes_.tolist()

    mlb = MultiLabelBinarizer()
    df_concat = pd.concat([df['LIWC-change'], df['LIWC-inplace']])
    one_hot_liwc = mlb.fit_transform(df_concat)
    one_hot_liwc_change, one_hot_liwc_inplace = np.split(one_hot_liwc, 2)
    liwc_classes = mlb.classes_.tolist()

    concret_w_change = df['concret-change'].to_numpy()
    concret_w_change = np.expand_dims(concret_w_change, axis=1)
    concret_w_inplace = df['concret-inplace'].to_numpy()
    concret_w_inplace = np.expand_dims(concret_w_inplace, axis=1)

    cosine_sim = df['cosine-sim'].to_numpy()
    cosine_sim = np.expand_dims(cosine_sim, axis=1)

    # wup_sim = df['wup_sim'].to_numpy() #TODO: What to do with words with no synsets?
    # wup_sim = np.expand_dims(wup_sim, axis=1)
    feat_categorical_names = ["_POS-" + str(i) for i in pos_categ] + \
                             ["Levin-change_" + str(i) for i in levin_classes] + \
                             ["Levin-inplace_" + str(i) for i in levin_classes] + \
                             ["LIWC-change_" + str(i) for i in liwc_classes] + \
                             ["LIWC-inplace_" + str(i) for i in liwc_classes]
    feat_continuous_names = ["_concret-change", "_concret-inplace", "_cosine-sim"]

    return (one_hot_pos, one_hot_levin_change, one_hot_levin_inplace, one_hot_liwc_change, one_hot_liwc_inplace,
            concret_w_change, concret_w_inplace, cosine_sim, feat_categorical_names, feat_continuous_names)


def get_changed_word(pos_triplet: str, neg_triplet: str, neg_type: str) -> Tuple[str, str]:
    if neg_type == 's':
        return pos_triplet[0], neg_triplet[0]
    elif neg_type == 'v':
        return pos_triplet[1], neg_triplet[1]
    elif neg_type == 'o':
        return pos_triplet[2], neg_triplet[2]
    else:
        raise ValueError(f"Wrong neg_type: {neg_type}, needs to be from s,v,o")


def get_bert_data(clip_results: Sequence) -> None:
    list_word_sentence, set_words = [], set()
    for _, sentence, neg_sentence, pos_triplet, neg_triplet, neg_type, _, _ in clip_results:
        word_changed, word_inplace = get_changed_word(pos_triplet, neg_triplet, neg_type)
        if not word_inplace or not word_changed:
            continue
        if (word_changed, sentence) not in list_word_sentence:
            list_word_sentence.append((word_changed, sentence))
        if (word_inplace, neg_sentence) not in list_word_sentence:
            list_word_sentence.append((word_inplace, neg_sentence))
        set_words.add(word_inplace)
        set_words.add(word_changed)
    print(len(list_word_sentence))
    save_bert_embeddings_sentences(list_word_sentence)
    # save_bert_embeddings(set_words)


def get_features(clip_results, path: str = "data/bert_embeddings.npy",
                 max_feature_count: Optional[int] = None) -> Tuple[pd.DataFrame, Sequence[int]]:
    dict_features = {"index": [], "sent": [], "n_sent": [], "word_changed": [], "word_inplace": [], "POS": [],
                     "Levin-change": [], "Levin-inplace": [], "LIWC-change": [], "LIWC-inplace": [],
                     "concret-change": [], "concret-inplace": [], "cosine-sim": [], "label": [],
                     "clip-score-diff": []}

    levin_dict, compressed_levin_dict = parse_levin_file()
    levin_semantic_broad, levin_semantic_all, levin_alternations = parse_levin_dict(levin_dict)
    dict_liwc, _ = parse_liwc_file()
    dict_concreteness = parse_concreteness_file()
    # bert_embeddings_sentences = np.load("data/bert_embeddings_sentences.npy", allow_pickle=True).item()  # transform from ndarray to dict
    bert_embeddings = np.load(path, allow_pickle=True).item()  # transform from ndarray to dict

    if max_feature_count:
        clip_results = clip_results[:max_feature_count]

    for index, sentence, neg_sentence, pos_triplet, neg_triplet, neg_type, clip_prediction, clip_score_diff in tqdm(
            clip_results, desc="Computing the features"):
        word_changed, word_inplace = get_changed_word(pos_triplet, neg_triplet, neg_type)

        if not word_inplace or not word_changed:
            print(f"Found empty word changed or word inplace in index {index} not processing data and continue ...")
            continue

        cosine_sim = get_cosine_similarity(word_changed, word_inplace, bert_embeddings)  # noqa
        if neg_type == 'v':  # TODO: How []/ No Levin or LIWC class influence results
            levin_classes_w_changed = get_levin_category(word_changed, levin_semantic_broad)
            levin_classes_w_inplace = get_levin_category(word_inplace, levin_semantic_broad)
        else:
            levin_classes_w_changed, levin_classes_w_inplace = [], []

        liwc_category_w_changed = get_liwc_category(word_changed, dict_liwc)
        liwc_category_w_inplace = get_liwc_category(word_inplace, dict_liwc)

        concret_w_change = get_concreteness_score(word_changed, dict_concreteness)
        concret_w_inplace = get_concreteness_score(word_inplace, dict_concreteness)

        dict_features["index"].append(index)
        dict_features["sent"].append(sentence)
        dict_features["n_sent"].append(neg_sentence)
        dict_features["word_changed"].append(word_changed)
        dict_features["word_inplace"].append(word_inplace)
        dict_features["POS"].append(neg_type)
        # dict_features["Levin"].append(levin_classes_w_changed)
        dict_features["Levin-change"].append(levin_classes_w_changed)
        dict_features["Levin-inplace"].append(levin_classes_w_inplace)
        # dict_features["LIWC"].append(liwc_category_w_changed)
        dict_features["LIWC-change"].append(liwc_category_w_changed)
        dict_features["LIWC-inplace"].append(liwc_category_w_inplace)
        # dict_features["concreteness"].append(concreteness_score_w_changed)
        dict_features["concret-change"].append(concret_w_change)
        dict_features["concret-inplace"].append(concret_w_inplace)
        dict_features["cosine-sim"].append(cosine_sim)
        # dict_features["wup_sim"].append(wnet_sim)
        # dict_features["WordNet"].append(wnet_category)

        # TODO: predict when CLIP result is pos or neg? - I think we want to learn when CLIP fails
        dict_features["label"].append(0 if clip_prediction == 'pos' else 1)
        # dict_features["label"].append(1 if clip_prediction == 'pos' else 0)
        dict_features["clip-score-diff"].append(clip_score_diff)

    levin_liwc = [item for sublist in dict_features["Levin-change"] + dict_features["Levin-inplace"] + \
                  dict_features["LIWC-change"] + dict_features["LIWC-inplace"] for item in sublist]
    features_count = levin_liwc + ["POS-" + v for v in dict_features["POS"]] + \
                     ["cosine-sim"] * len(dict_features["cosine-sim"]) + \
                     ["concret-change"] * len(dict_features["concret-change"]) + \
                     ["concret-inplace"] * len(dict_features["concret-inplace"])
    df = pd.DataFrame.from_dict(dict_features)
    # print(df[['word_changed', 'word_inplace', 'cosine-sim']].to_string())
    return df, features_count


def pre_process_features(labels, one_hot_pos, one_hot_levin_w_change, one_hot_levin_w_inplace,
                         one_hot_liwc_change, one_hot_liwc_inplace, concret_w_change, concret_w_inplace, cosine_sim,
                         feat_categorical_names, feat_continuous_names) -> Tuple[np.ndarray, Sequence[str]]:
    feat_categorical = np.concatenate((one_hot_pos, one_hot_levin_w_change, one_hot_levin_w_inplace,
                                       one_hot_liwc_change, one_hot_liwc_inplace), axis=1)
    print(f"categorical feature size: {feat_categorical.shape}")

    # categorical feature selection
    # print('Before categ feat selection: feat_categorical_shape: %s feat_categorical_names_len: %d' % (feat_categorical.shape, len(feat_categorical_names)))
    # feat_categorical, feat_categorical_names = filter_categorical_features(labels, feat_categorical, feat_categorical_names)
    # print('After categ feat selection: feat_categorical_shape: %s feat_categorical_names_len: %d' % (feat_categorical.shape, len(feat_categorical_names)))

    feat_continuous = np.concatenate((concret_w_change, concret_w_inplace, cosine_sim), axis=1)
    scaler = preprocessing.StandardScaler()
    scaler.fit(feat_continuous)  # standardize continuous features
    feat_continuous_scaled = scaler.transform(feat_continuous)
    print(f"continuous feature scaled size: {feat_continuous_scaled.shape}")

    # concatenate categorical (one hot) with scaled continuous features
    features = np.concatenate((feat_categorical, feat_continuous_scaled), axis=1)
    feature_names = feat_categorical_names + feat_continuous_names

    # sel = VarianceThreshold(threshold=(.95 * (1 - .95)))
    # features_remove_low_var = sel.fit_transform(features_scaled)
    # print(f"feature after remove low var size: {features_remove_low_var.shape}")
    print(f"all (categorical + continuous) features size: {features.shape}")
    return features, feature_names


def eval_split(features: np.ndarray, labels: np.ndim) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                                test_size=0.1,
                                                                                random_state=5)
    print('feat_shape: %s\n labels_shape: %s\n train_feat_shape: %s\n test_feat_shape: %s' % (
        features.shape, labels.shape, features_train.shape, features_test.shape))
    return features_train, labels_train, features_test, labels_test


def majority_class(labels_test: np.ndarray) -> None:
    method_name = 'Majority class'
    predicted = [Counter(labels_test).most_common()[0][0]] * len(labels_test)
    evaluate(method_name, labels_test, predicted)


def run_svm(feat_train: np.ndarray, labels_train: np.ndarray, feat_test: np.ndarray,
            labels_test: np.ndarray) -> np.ndarray:
    method_name = "SVM"
    method = build_classifier()
    method.fit(feat_train, labels_train)
    coef_weights = method.coef_  # Weights assigned to the features when kernel="linear"
    predicted = method.predict(feat_test)
    evaluate(method_name, labels_test, predicted)  # TODO - MIGHT NOT NEED TO TEST?
    return coef_weights


# def run_regression(features_scaled, df):
#     method = Ridge() #normalize=False as we input features_scaled
#
#     # clip_scores = df['clip-score-diff'].to_numpy()
#     clip_labels = df['label'].to_numpy()
#     method.fit(features_scaled, clip_labels)
#     score = method.score(features_scaled, clip_labels) #Return the coefficient of determination
#     print(f"Ridge regression score: {score:.2f}")
#     coef_weights = method.coef_
#     # predicted = method.predict(feat_test)
#     # evaluate(method_name, labels_test, predicted)
#     return coef_weights

def plot_coef_weights(coef_weights: np.ndarray, feature_names: Sequence[str],
                      path: str = "data/coef_importance.png") -> None:
    top_features = 5
    coef = coef_weights.ravel()  # flatten array
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    fig = plt.figure(figsize=(18, 7))
    colors = ['red' if c < 0 else 'green' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(2 * top_features), feature_names[top_coefficients], rotation=45, ha='right')
    fig.savefig(path, bbox_inches='tight')


# https://www.kaggle.com/code/pierpaolo28/pima-indians-diabetes-database/notebook
def print_sorted_coef_weights(coef: np.ndarray, coef_significance: np.ndarray, feature_names: Sequence[str],
                              features_count: Sequence[int], output_path: str = "data/sorted_features.csv") -> None:
    sorted_coefficients_idx = np.argsort(coef)[::-1]  # in descending order
    sorted_coefficients = [np.round(weight, 2) for weight in coef[sorted_coefficients_idx]]

    feature_names = np.array(feature_names)
    sorted_feature_names = feature_names[sorted_coefficients_idx].tolist()
    sorted_feature_significance = coef_significance[sorted_coefficients_idx].tolist()
    sorted_feature_counts = [features_count.count(feature.split("_")[1]) for feature in sorted_feature_names]

    df = pd.DataFrame(
        zip(sorted_feature_names, sorted_feature_significance, sorted_feature_counts, sorted_coefficients),
        columns=['Feature', 'Significance', 'Data Count', 'Weight (abs)'])
    df.to_csv(output_path, index=False)


def evaluate(method_name: str, labels_test: np.ndarray, predicted: Sequence[int]) -> None:
    accuracy = accuracy_score(labels_test, predicted) * 100
    precision = precision_score(labels_test, predicted) * 100
    recall = recall_score(labels_test, predicted) * 100
    f1 = f1_score(labels_test, predicted) * 100
    roc_auc = roc_auc_score(labels_test, predicted) * 100
    print(
        f"Method {method_name}, A: {accuracy:.2f}, P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}, ROC-AUC: {roc_auc:.2f}")
    print(f"Counter predicted: {Counter(predicted)}")
    print(f"Counter GT: {Counter(labels_test)}")


def print_metrics(df: pd.DataFrame, feature_names: Sequence[str]) -> None:
    main_feature_names = [feature_name.split("_")[0] for feature_name in feature_names]
    print(f"Counter all labels: {Counter(df['label'].tolist())}")
    print(f"Data size: {len(df['index'].tolist())}")
    print(f"Features size: {len(feature_names)}, {Counter(main_feature_names)}")

    levin_dict, compressed_levin_dict = parse_levin_file()
    levin_semantic_broad, levin_semantic_all, levin_alternations = parse_levin_dict(levin_dict)
    print(f"--Levin semantic_broad nb classes: {len(levin_semantic_broad.keys())}")
    print(f"--Levin semantic_all nb classes: {len(levin_semantic_all.keys())}")
    print(f"--Levin alternations nb classes: {len(levin_alternations.keys())}")

    liwc_dict, liwc_categ = parse_liwc_file()
    print(f"LIWC total nb classes: {len(liwc_categ)}")


def merge_csvs_and_filter_data(probes_path: str = "data/svo_probes.csv", neg_path: str = "data/neg_d.csv",
                               output_path: str = "data/merged.csv") -> None:
    df_probes = pd.read_csv(probes_path, index_col="index")
    df_probes.drop(df_probes.index[df_probes["sentence"] == 'woman, ball, outside'], inplace=True)
    df_probes.drop(df_probes.index[df_probes["sentence"] == 'woman, music, notes'], inplace=True)
    # df_probes.drop(df_probes.index[df_probes["sentence"] == 'People on a team running.'], inplace=True)
    # df_probes.drop(df_probes.index[df_probes["sentence"] == 'People playing on a team.'], inplace=True)

    df_neg = pd.read_csv(neg_path, header=0)
    df_neg.drop(df_neg.index[df_neg["neg_sentence"] == 'woman, ball, outside'], inplace=True)
    df_neg.drop(df_neg.index[df_neg["neg_sentence"] == 'woman, music, notes'], inplace=True)
    # df_neg.drop(df_neg.index[df_neg["neg_sentence"] == 'People on a team running.'], inplace=True)
    # df_neg.drop(df_neg.index[df_neg["neg_sentence"] == 'People playing on a team.'], inplace=True)

    result = pd.concat([df_probes, df_neg["neg_sentence"]], axis=1)
    result = result[result["sentence"].notna()]

    result.to_csv(output_path)


def analyse_feat_correlations(df: pd.DataFrame) -> None:
    # df_feat = df[["POS", "Levin-change", "Levin-inplace", "LIWC-change", "LIWC-inplace", "concret-change",
    df_feat = df[["concret-change", "concret-inplace", "cosine-sim", "label"]]
    corr = df_feat.corr()
    print(corr)


# feature selection
def select_categorical_features(X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, SelectKBest]:
    # fs = SelectKBest(score_func=chi2, k='all') #chi2 = a test for independence between categorical variables.
    fs = SelectKBest(score_func=mutual_info_classif,
                     k='all')  # chi2 = a test for independence between categorical variables.
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


def delete_multiple_element(list_object: List[int], indices: Sequence[int]) -> None:
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


def filter_categorical_features(labels: np.ndarray, feat_categorical: np.ndarray,
                                feat_categorical_names: Sequence[str],
                                threshold: float = 0.001) -> Tuple[np.ndarray, Sequence[str]]:
    feat_train, labels_train, feat_test, labels_test = eval_split(feat_categorical, labels)
    X_train_fs, X_test_fs, fs = select_categorical_features(feat_train, labels_train, feat_test)
    feat_scores = [0 if math.isnan(x) else x for x in fs.scores_]
    feat_names_scores = zip(feat_categorical_names, feat_scores)

    # sorted_feat_names_scores = sorted(feat_names_scores, key=lambda x: x[1], reverse=True)
    # # # what are scores for the features
    # for (feat_name, feat_score) in sorted_feat_names_scores:
    #     print('Feature %s: %f' % (feat_name, feat_score))

    indices = []
    for i, (feat_name, feat_score) in enumerate(feat_names_scores):
        if feat_score < threshold:
            indices.append(i)
    # print("Removed features: " + str([feat_categorical_names[i] for i in indices]))
    feat_categorical = np.delete(feat_categorical, indices, axis=1)
    delete_multiple_element(feat_categorical_names, indices)

    return feat_categorical, feat_categorical_names


def process_features(clip_results,
                     max_feature_count: Optional[int] = None) -> Tuple[np.ndarray, Sequence[str], np.ndarray,
                                                                       np.ndarray]:
    df, features_count = get_features(clip_results, max_feature_count=max_feature_count)
    labels = df['label'].to_numpy()
    # analyse_feat_correlations(df)
    # get_bert_data(clip_results)

    one_hot_pos, one_hot_levin_w_change, one_hot_levin_w_inplace, one_hot_liwc_change, one_hot_liwc_inplace, \
    concret_w_change, concret_w_inplace, cosine_sim, feat_categorical_names, feat_continuous_names = \
        transform_features(df)

    # categorical feature selection, standardize (scale) continuous features and concatenate categorical & continuous
    features, feature_names = pre_process_features(labels, one_hot_pos, one_hot_levin_w_change, one_hot_levin_w_inplace,
                                                   one_hot_liwc_change, one_hot_liwc_inplace, concret_w_change,
                                                   concret_w_inplace,
                                                   cosine_sim, feat_categorical_names, feat_continuous_names)

    print_metrics(df, feature_names)
    return features, feature_names, features_count, labels


def build_classifier() -> svm.LinearSVC:
    return svm.LinearSVC(class_weight='balanced', max_iter=1_000_000)


def classify_shuffled(feat_train: np.ndarray, labels_train: np.ndarray, index: int) -> np.ndarray:
    np.random.seed(index)

    feat_train = np.random.permutation(feat_train.T).T

    clf = build_classifier()
    clf.fit(feat_train, labels_train)

    return abs(clf.coef_.ravel())


def analyse_coef_weights(feat_train: np.ndarray, labels_train: np.ndarray,
                         iterations: int = 10_000) -> Tuple[np.ndarray, np.ndarray]:
    # categ_feat_train = feat_train[:, :-3]
    # categ_feature_names = np.array(feature_names[:-3])
    clf = build_classifier()

    print("Computing the weights with the real features…")
    clf.fit(feat_train, labels_train)
    print("Weights computed.")

    coef_weights = abs(clf.coef_.ravel())  # flatten array and take absolute value

    with Pool() as pool:
        list_shuffled_coef_weights = list(tqdm(
            pool.imap_unordered(partial(classify_shuffled, feat_train, labels_train), range(iterations)),
            total=iterations, desc="Computing the weights with shuffled columns"))

    coef_significance = []
    for i, coef in enumerate(coef_weights):
        significance = 0
        for list_coef in list_shuffled_coef_weights:
            if list_coef[i] <= coef:
                significance += 1
        # coef_significance.append([categ_feature_names[i], coef, significance])
        # coef_significance.append(True if significance >= 0.95 * total_count else False)
        coef_significance.append(significance)
    # coef_significance.append(100)  # concret_change
    # coef_significance.append(100)  # concret_inplace
    # coef_significance.append(100)  # cosine_sim
    coef_significance = np.array(coef_significance)
    return coef_weights, coef_significance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()

    # get_wnet_category(word='', pos='')
    # merge_csvs_and_filter_data()

    clip_results = read_data()
    features, feature_names, features_count, labels = process_features(clip_results,
                                                                       max_feature_count=1000 if args.debug else None)
    feat_train, labels_train, feat_test, labels_test = eval_split(features, labels)
    coef_weights, coef_significance = analyse_coef_weights(feat_train, labels_train, args.iterations)
    # coef_weights_svm = run_SVM(feat_train, labels_train, feat_test, labels_test)
    # # # coef_weights = run_regression(features_scaled, df)
    print_sorted_coef_weights(coef_weights, coef_significance, feature_names, features_count)
    # # # plot_coef_weights(coef_weights, feature_names)
    #
    # # majority_class(labels_test) # A: 82.93, P: 0.00, R: 0.00, F1: 0.00, ROC-AUC: 50.00


if __name__ == "__main__":
    main()
