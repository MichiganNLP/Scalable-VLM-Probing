from collections import Counter

import pandas as pd
import ast
from typing import Sequence, Tuple
from rich.progress import track
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import svm
import numpy as np

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
    return one_hot_pos, one_hot_levin


def get_all_properties(clip_results):
    dict_properties = {"index": [], "sent": [], "POS": [], "Levin": [], "label":[]}
    map_keys = map_levin_keys()
    # clip_results = clip_results[:100]
    for index, sentence, pos_triplet, neg_triplet, neg_type, clip_prediction in tqdm(clip_results):
        [subj, verb, obj] = pos_triplet
        if neg_type == 'v':
            levin_classes = get_verb_properties(verb, map_keys)
        else:
            levin_classes = []

        dict_properties["index"].append(index)
        dict_properties["sent"].append(sentence)
        dict_properties["POS"].append(neg_type)
        dict_properties["Levin"].append(levin_classes)

        dict_properties["label"].append(0 if clip_prediction == 'pos' else 1) #TODO: predict when CLIP result is pos or neg?
        # dict_properties["label"].append(1 if clip_prediction == 'pos' else 0) #TODO: predict when CLIP result is pos or neg?

    df = pd.DataFrame.from_dict(dict_properties)
    # print(df.to_string())
    return df


def eval_split(one_hot_pos, one_hot_levin, df):
    all_feat = np.concatenate((one_hot_pos, one_hot_levin), axis=1)
    all_labels = df['label'].to_numpy()
    feat_train, feat_test, labels_train, labels_test = train_test_split(all_feat, all_labels, test_size=0.1,
                                                                        random_state=5)
    print(all_feat.shape, all_labels.shape, feat_train.shape, feat_test.shape)
    return feat_train, labels_train, feat_test, labels_test

def majority_class(labels_test):
    method_name = 'Majority class'
    predicted = [Counter(labels_test).most_common()[0][0]] * len(labels_test)
    evaluate(method_name, labels_test, predicted)

def run_SVM(feat_train, labels_train, feat_test, labels_test):
    method_name = "SVM"
    # method = svm.SVC(kernel='linear')
    method = svm.SVC()
    method.fit(feat_train, labels_train)
    # print(method.coef_)
    predicted = method.predict(feat_test)
    evaluate(method_name, labels_test, predicted)


def evaluate(method_name, labels_test, predicted):
    accuracy = accuracy_score(labels_test, predicted) * 100
    precision = precision_score(labels_test, predicted) * 100
    recall = recall_score(labels_test, predicted) * 100
    f1 = f1_score(labels_test, predicted) * 100
    print(f"Method {method_name}, A: {accuracy:.1f}, P: {precision:.1f}, R: {recall:.1f}, F1: {f1:.1f} ")
    print(Counter(predicted))
    print(Counter(labels_test))


if __name__ == "__main__":
    clip_results = read_SVO_CLIP_results_file()
    df = get_all_properties(clip_results)
    one_hot_pos, one_hot_levin = transform_features(df)

    feat_train, labels_train, feat_test, labels_test = eval_split(one_hot_pos, one_hot_levin, df)
    run_SVM(feat_train, labels_train, feat_test, labels_test)
    majority_class(labels_test)
