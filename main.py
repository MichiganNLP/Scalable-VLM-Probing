from collections import Counter

import pandas as pd
import ast
from typing import Sequence, Tuple

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import svm
from sklearn import preprocessing
import matplotlib.pyplot as plt


def parse_triplets(triplets: str) -> Sequence[Tuple[str, str, str]]:
    if triplets.startswith("["):
        return [triplet.split(",") for triplet in ast.literal_eval(triplets)]
    else:
        return [triplets.split(",")]


def get_first_triplet(triplets: Sequence[Tuple[str, str, str]]):
    return next(iter(triplets), ("", "", ""))


def read_SVO_CLIP_results_file():
    df = pd.read_csv("data/svo_probes.csv", index_col='index')
    df = df.sort_index()
    df['index'] = df.index
    results = []
    for index, sentence, pos_triplet, neg_triplet, neg_type, clip_prediction in zip(df['index'], df['sentence'],
                                                                                    df['pos_triplet'],
                                                                                    df['neg_triplet'],
                                                                                    df['neg_type'],  # s, v or o
                                                                                    df['clip prediction']):
        parsed_pos_triplet = parse_triplets(pos_triplet)
        parsed_neg_triplet = parse_triplets(neg_triplet)
        first_pos_triplet = get_first_triplet(parsed_pos_triplet)
        first_neg_triplet = get_first_triplet(parsed_neg_triplet)
        results.append([index, sentence, first_pos_triplet, first_neg_triplet, neg_type[0], clip_prediction])

    return results


def parse_liwc_file():
    dict_liwc = {}
    with open('data/LIWC.2015.all.txt') as file:
        for line in file:
            word, category = [w.strip() for w in line.strip().split(",")]
            if word not in dict_liwc:
                dict_liwc[word] = []
            dict_liwc[word].append(category)
    return dict_liwc


def parse_concreteness_file():
    dict_concreteness = {}
    with open('data/concretness.txt') as file:
        lines = file.readlines()
    for line in lines[1:]:
        word, _, conc_m, _, _, _, _, _, _ = line.split("	")
        dict_concreteness[word] = round(float(conc_m))
    return dict_concreteness


def get_liwc_category(word, dict_liwc):
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


def get_wnet_category(word, pos):
    for word, pos in zip(["stand", "fry", "cook"], ["v", "v", "v"]):
        if pos == 's' or pos == 'o':
            pos = 'n'  # TODO: might have other than nouns
        synset = wordnet.synset(".".join([word, pos, '01']))
        hypernyms = synset.hypernyms()  # gives other synsets
        lexname = hypernyms[0].lexname()
        name = hypernyms[0].name()
    # return hypernyms


def get_concreteness_score(word, dict_concreteness):
    if word in dict_concreteness:
        return dict_concreteness[word]
    else:
        return 0 #TODO: nan?


def parse_levin_file():
    content = ""
    levin_dict = {}
    with open('data/levin_verbs.txt') as file:
        for line in file:
            line = line.lstrip()
            if line and line[0].isnumeric():
                key = " ".join(line.split())
            else:
                if not line:
                    if '-*-' not in content:
                        levin_dict[key] = [x.lower() for x in content.split()]
                    content = ""
                else:
                    content += line.replace('\r\n', "").rstrip()
                    content += " "
    return levin_dict


def map_levin_keys():
    map_keys = {}
    index = 1
    levin_dict = parse_levin_file()
    for key in levin_dict.keys():
        map_keys[key] = index
        index += 1
    print(f"There are {index} verb Levin classes")  # TODO: Compress classes?
    return map_keys


def get_levin_class_per_verb(verb):
    levin_dict = parse_levin_file()
    levin_classes = []
    for key, values in levin_dict.items():
        if verb in values:
            levin_classes.append(key)
    return levin_classes


def get_verb_properties(verb, map_keys):
    levin_classes = get_levin_class_per_verb(verb)
    index_classes = [map_keys[levin_class] for levin_class in levin_classes]
    return index_classes


def transform_features(df):
    ohe = OneHotEncoder()
    transformed = ohe.fit_transform(df[['POS']])
    one_hot_pos = transformed.toarray()
    # print(ohe.categories_)

    mlb = MultiLabelBinarizer()
    one_hot_levin = mlb.fit_transform(df['Levin'])
    # print(mlb.classes_)
    # print(pd.DataFrame(one_hot_levin, columns=mlb.classes_, index=df.index).to_string())

    mlb = MultiLabelBinarizer()
    one_hot_liwc = mlb.fit_transform(df['LIWC'])

    int_concreteness = df['concreteness'].to_numpy()
    int_concreteness = np.expand_dims(int_concreteness, axis=1)

    feature_names = ["POS"] * len(one_hot_pos) + ["Levin"] * len(one_hot_levin) + ["LIWC"] * len(one_hot_liwc) + ["Concretness"]
    return one_hot_pos, one_hot_levin, one_hot_liwc, int_concreteness, feature_names


def get_changed_word(pos_triplet, neg_type):
    if neg_type == 's':
        return pos_triplet[0]
    elif neg_type == 'v':
        return pos_triplet[1]
    elif neg_type == 'o':
        return pos_triplet[2]
    else:
        raise ValueError(f"Wrong neg_type: {neg_type}, needs to be from s,v,o")


def get_all_properties(clip_results):
    # dict_properties = {"index": [], "sent": [], "POS": [], "Levin": [], "LIWC": [], "WordNet": [], "label": []}
    dict_properties = {"index": [], "word": [], "POS": [], "Levin": [], "LIWC": [], "concreteness": [], "label": []}
    map_keys = map_levin_keys()
    dict_liwc = parse_liwc_file()
    dict_concreteness = parse_concreteness_file()
    # nb_no_liwc = 0

    clip_results = clip_results[:10]
    for index, sentence, pos_triplet, neg_triplet, neg_type, clip_prediction in tqdm(clip_results):
        word = get_changed_word(pos_triplet, neg_type)
        if neg_type == 'v':
            levin_classes = get_verb_properties(word, map_keys)
        else:
            levin_classes = []
        liwc_category = get_liwc_category(word, dict_liwc)
        # wnet_category = get_wnet_category(word, neg_type)
        concreteness_score = get_concreteness_score(word, dict_concreteness)
        # if not liwc_category:
        # if not concreteness_score:
        #     nb_no_liwc += 1
        #     print(word)

        dict_properties["index"].append(index)
        # dict_properties["sent"].append(sentence)
        dict_properties["word"].append(word)
        dict_properties["POS"].append(neg_type)
        dict_properties["Levin"].append(levin_classes)
        dict_properties["LIWC"].append(liwc_category)
        dict_properties["concreteness"].append(concreteness_score)
        # dict_properties["WordNet"].append(wnet_category)

        # TODO: predict when CLIP result is pos or neg? - I think we want to learn when CLIP fails
        dict_properties["label"].append(0 if clip_prediction == 'pos' else 1)
        # dict_properties["label"].append(1 if clip_prediction == 'pos' else 0)

    df = pd.DataFrame.from_dict(dict_properties)
    # print(df.to_string())
    # nb_total = len(dict_properties["index"])
    # print(f"There are {nb_no_liwc} examples with no LIWC category from {nb_total}")
    return df

def pre_process_features(one_hot_pos, one_hot_levin, int_concreteness):
    # print(one_hot_pos.shape, one_hot_levin.shape, int_concreteness.shape)
    all_features = np.concatenate((one_hot_pos, one_hot_levin, int_concreteness), axis=1)
    # standardize features
    scaler = preprocessing.StandardScaler().fit(all_features)
    features_scaled = scaler.transform(all_features)
    return features_scaled

def eval_split(all_features_scaled, df):
    all_labels = df['label'].to_numpy()
    features_train, features_test, labels_train, labels_test = train_test_split(all_features_scaled, all_labels, test_size=0.1,
                                                                        random_state=5)
    # print(all_features_scaled.shape, all_labels.shape, features_train.shape, features_test.shape)
    return features_train, labels_train, features_test, labels_test


def majority_class(labels_test):
    method_name = 'Majority class'
    predicted = [Counter(labels_test).most_common()[0][0]] * len(labels_test)
    evaluate(method_name, labels_test, predicted)


def run_SVM(feat_train, labels_train, feat_test, labels_test):
    method_name = "SVM"
    # method = svm.SVC(kernel='linear')
    # method = svm.SVC()
    method = svm.SVC(class_weight='balanced', kernel='linear')
    method.fit(feat_train, labels_train)
    coef_weights = method.coef_ # Weights assigned to the features when kernel="linear"
    predicted = method.predict(feat_test)
    evaluate(method_name, labels_test, predicted)
    return coef_weights


def analyse_coef_weights(coef_weights, feature_names):
    top_features = 5
    coef = coef_weights.ravel() #flatten array
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    fig = plt.figure(figsize=(18, 7))
    colors = ['green' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1 + 2 * top_features), feature_names[top_coefficients], rotation=45, ha='right')
    fig.savefig("data/coef_importance.png", bbox_inches='tight')

def evaluate(method_name, labels_test, predicted):
    accuracy = accuracy_score(labels_test, predicted) * 100
    precision = precision_score(labels_test, predicted) * 100
    recall = recall_score(labels_test, predicted) * 100
    f1 = f1_score(labels_test, predicted) * 100
    roc_auc = roc_auc_score(labels_test, predicted) * 100
    print(f"Method {method_name}, A: {accuracy:.2f}, P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}, ROC-AUC: {roc_auc:.2f}")
    print(Counter(predicted))
    print(Counter(labels_test))

def print_metrics(df):
    print(f"Counter all labels: {Counter(df['label'].tolist())}")

if __name__ == "__main__":
    # get_wnet_category(word='', pos='')

    clip_results = read_SVO_CLIP_results_file()
    df = get_all_properties(clip_results)
    one_hot_pos, one_hot_levin, one_hot_liwc, int_concreteness, feature_names = transform_features(df)
    print_metrics(df)

    features_scaled = pre_process_features(one_hot_pos, one_hot_levin, int_concreteness)
    feat_train, labels_train, feat_test, labels_test = eval_split(features_scaled, df)

    coef_weights = run_SVM(feat_train, labels_train, feat_test, labels_test)
    analyse_coef_weights(coef_weights, feature_names)
    # majority_class(labels_test)
