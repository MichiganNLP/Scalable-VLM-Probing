import string
from collections import Counter
import pandas as pd
import ast
from typing import Sequence, Tuple
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
import numpy as np
from pygments.lexers import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import svm
from sklearn import preprocessing
import matplotlib.pyplot as plt
from word_forms.word_forms import get_word_forms

from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()



def parse_triplets(triplets: str) -> Sequence[Tuple[str, str, str]]:
    if triplets.startswith("["):
        return [triplet.split(",") for triplet in ast.literal_eval(triplets)]
    else:
        return [triplets.split(",")]


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


def pre_process_sentences(sentence):
    if type(sentence) == str:
        sentence = sentence.lower()
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    else:
        sentence = ""
    return sentence


def read_SVO_CLIP_results_file():
    # df = pd.read_csv("data/svo_probes.csv", index_col='index')
    df = pd.read_csv("data/merged.csv", index_col=0)
    df = df.sort_index()
    df['index'] = df.index
    results = []
    for index, sentence, neg_sentence, pos_triplet, neg_triplet, neg_type, clip_prediction in zip(df['index'], df['sentence'],
                                                                                    df['neg_sentence'],
                                                                                    df['pos_triplet'],
                                                                                    df['neg_triplet'],
                                                                                    df['neg_type'],  # s, v or o
                                                                                    df['clip prediction']):

        sentence = pre_process_sentences(sentence) #remove punctuation
        neg_sentence = pre_process_sentences(neg_sentence)

        parsed_pos_triplet = parse_triplets(pos_triplet)
        parsed_neg_triplet = parse_triplets(neg_triplet)
        if not parsed_pos_triplet or not parsed_neg_triplet or not sentence:
            continue

        match_pos_triplet = get_sentence_match_triplet(parsed_pos_triplet, sentence)
        match_neg_triplet = get_sentence_match_triplet(parsed_neg_triplet, neg_sentence)

        results.append([index, sentence, neg_sentence, match_pos_triplet, match_neg_triplet, neg_type[0], clip_prediction])

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


# def get_wnet_category(word, pos):
#     for word, pos in zip(["stand", "fry", "cook"], ["v", "v", "v"]):
#         if pos == 's' or pos == 'o':
#             pos = 'n'  # TODO: might have other than nouns
#         synset = wordnet.synset(".".join([word, pos, '01']))
#         hypernyms = synset.hypernyms()  # gives other synsets
#         lexname = hypernyms[0].lexname()
#         name = hypernyms[0].name()
#     # return hypernyms

def get_wup_similarity(word_changed, word_inplace, pos):
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

def compute_embedding(word_type, sentence):
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
        else: # treat as separate word / not part of a sentence
            inputs = tokenizer(word_type, return_tensors="pt")
            index_word = [1]
    else: # treat as separate word / not part of a sentence
        inputs = tokenizer(word_type, return_tensors="pt")
        index_word = [1]

    with torch.no_grad():
        outputs = model(**inputs)

    embedding_word = np.mean(outputs.last_hidden_state.detach().numpy()[0][index_word[0]:index_word[-1]+1], axis=0)
    embedding_word = np.expand_dims(embedding_word, axis=0)
    return embedding_word

def save_BERT_embeddings(list_word_sentence):
    dict_data = {}
    for [word, sentence] in tqdm(list_word_sentence):
        embedding_word = compute_embedding(word, sentence)
        dict_data[(word, sentence)] = embedding_word

    np.save("data/bert_embeddings.npy", dict_data)

def get_cosine_similarity(word_changed, word_inplace, sentence, neg_sentence, bert_embeddings):
    embedding_word_changed = bert_embeddings[(word_changed, sentence)]
    embedding_word_inplace = bert_embeddings[(word_inplace, neg_sentence)]
    cosine_sim = cosine_similarity(embedding_word_changed, embedding_word_inplace)[0][0]
    return cosine_sim


def get_concreteness_score(word, dict_concreteness):
    if word in dict_concreteness:
        return dict_concreteness[word]
    else:
        return 0  # TODO: nan?, mean of all scores??  - to not influence


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
    pos_categ = ohe.categories_[0].tolist()

    mlb = MultiLabelBinarizer()
    one_hot_levin = mlb.fit_transform(df['Levin'])
    levin_classes_indexes = mlb.classes_.tolist()
    map_keys = map_levin_keys()
    levin_classes = [key for index in levin_classes_indexes for (key, value) in map_keys.items() if value == index]
    # print(pd.DataFrame(one_hot_levin, columns=mlb.classes_, index=df.index).to_string())

    mlb = MultiLabelBinarizer()
    one_hot_liwc = mlb.fit_transform(df['LIWC'])
    liwc_classes = mlb.classes_.tolist()

    int_concreteness = df['concreteness'].to_numpy()
    int_concreteness = np.expand_dims(int_concreteness, axis=1)

    cosine_sim = df['cosine_sim'].to_numpy()
    cosine_sim = np.expand_dims(cosine_sim, axis=1)

    # wup_sim = df['wup_sim'].to_numpy() #TODO: What to do with words with no synsets?
    # wup_sim = np.expand_dims(wup_sim, axis=1)

    feature_names = ["POS_" + str(i) for i in pos_categ] + ["Levin_" + str(i) for i in levin_classes] + [
        "LIWC_" + str(i) for i in liwc_classes] + ["Concreteness", "cosine_sim"]
    return one_hot_pos, one_hot_levin, one_hot_liwc, int_concreteness, cosine_sim, feature_names


def get_changed_word(pos_triplet, neg_triplet, neg_type):
    if neg_type == 's':
        return pos_triplet[0], neg_triplet[0]
    elif neg_type == 'v':
        return pos_triplet[1], neg_triplet[1]
    elif neg_type == 'o':
        return pos_triplet[2], neg_triplet[2]
    else:
        raise ValueError(f"Wrong neg_type: {neg_type}, needs to be from s,v,o")


def get_bert_data(clip_results):
    list_word_sentence = []
    for index, sentence, neg_sentence, pos_triplet, neg_triplet, neg_type, clip_prediction in tqdm(clip_results):
        word_changed, word_inplace = get_changed_word(pos_triplet, neg_triplet, neg_type)
        if not word_inplace or not word_changed:
            continue
        if [word_changed, sentence] not in list_word_sentence:
            list_word_sentence.append([word_changed, sentence])
        if [word_inplace, neg_sentence] not in list_word_sentence:
            list_word_sentence.append([word_inplace, neg_sentence])
    print(len(list_word_sentence))
    save_BERT_embeddings(list_word_sentence)



def get_all_properties(clip_results):
    dict_properties = {"index": [], "sent": [], "n_sent": [], "word_changed": [], "word_inplace": [], "POS": [], "Levin": [], "LIWC": [],
                       "concreteness": [], "cosine_sim": [], "label": []}
    map_keys = map_levin_keys()
    dict_liwc = parse_liwc_file()
    dict_concreteness = parse_concreteness_file()
    # nb_no_liwc = 0
    bert_embeddings = np.load("data/bert_embeddings.npy", allow_pickle=True).item() # transform from ndarray to dict
    # clip_results = clip_results[:10]
    for index, sentence, neg_sentence, pos_triplet, neg_triplet, neg_type, clip_prediction in tqdm(clip_results):
        word_changed, word_inplace = get_changed_word(pos_triplet, neg_triplet, neg_type)

        if not word_inplace or not word_changed:
            print(f"Found empty word changed or word inplace in index {index} not processing data and continue ...")
            continue

        # wnet_sim = get_wup_similarity(word_changed, word_inplace, neg_type)
        cosine_sim = get_cosine_similarity(word_changed, word_inplace, sentence, neg_sentence, bert_embeddings)

        if neg_type == 'v':
            levin_classes = get_verb_properties(word_changed, map_keys)
        else:
            levin_classes = []
        liwc_category = get_liwc_category(word_changed, dict_liwc)
        concreteness_score = get_concreteness_score(word_changed, dict_concreteness)
        # if not liwc_category:
        # if not concreteness_score:
        #     nb_no_liwc += 1
        #     print(word)

        dict_properties["index"].append(index)
        dict_properties["sent"].append(sentence)
        dict_properties["n_sent"].append(neg_sentence)
        dict_properties["word_changed"].append(word_changed)
        dict_properties["word_inplace"].append(word_inplace)
        dict_properties["POS"].append(neg_type)
        dict_properties["Levin"].append(levin_classes)
        dict_properties["LIWC"].append(liwc_category)
        dict_properties["concreteness"].append(concreteness_score)
        dict_properties["cosine_sim"].append(cosine_sim)
        # dict_properties["wup_sim"].append(wnet_sim)
        # dict_properties["WordNet"].append(wnet_category)

        # TODO: predict when CLIP result is pos or neg? - I think we want to learn when CLIP fails
        # dict_properties["label"].append(0 if clip_prediction == 'pos' else 1)
        dict_properties["label"].append(1 if clip_prediction == 'pos' else 0)

    df = pd.DataFrame.from_dict(dict_properties)
    # print(df.to_string())
    # nb_total = len(dict_properties["index"])
    # print(f"There are {nb_no_liwc} examples with no LIWC category from {nb_total}")
    return df


def pre_process_features(one_hot_pos, one_hot_levin, int_concreteness, cosine_sim):
    # print(one_hot_pos.shape, one_hot_levin.shape, int_concreteness.shape)
    all_features = np.concatenate((one_hot_pos, one_hot_levin, int_concreteness, cosine_sim), axis=1)
    # standardize features
    scaler = preprocessing.StandardScaler().fit(all_features)
    features_scaled = scaler.transform(all_features)
    return features_scaled


def eval_split(all_features_scaled, df):
    all_labels = df['label'].to_numpy()
    features_train, features_test, labels_train, labels_test = train_test_split(all_features_scaled, all_labels,
                                                                                test_size=0.1,
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
    coef_weights = method.coef_  # Weights assigned to the features when kernel="linear"
    predicted = method.predict(feat_test)
    evaluate(method_name, labels_test, predicted)
    return coef_weights


def plot_coef_weights(coef_weights, feature_names):
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
    fig.savefig("data/coef_importance.png", bbox_inches='tight')


# https://www.kaggle.com/code/pierpaolo28/pima-indians-diabetes-database/notebook
def print_sorted_coef_weights(coef_weights, feature_names):
    coef = coef_weights.ravel()  # flatten array
    sorted_coefficients_idx = np.argsort(coef)[::-1]  # in descending order
    sorted_coefficients = [np.round(weight, 2) for weight in coef[sorted_coefficients_idx]]

    feature_names = np.array(feature_names)
    sorted_feature_names = feature_names[sorted_coefficients_idx]

    df = pd.DataFrame(zip(sorted_feature_names, sorted_coefficients), columns=['Feature', 'Weight'])
    df.to_csv("data/sorted_features.csv", index=False)


def evaluate(method_name, labels_test, predicted):
    accuracy = accuracy_score(labels_test, predicted) * 100
    precision = precision_score(labels_test, predicted) * 100
    recall = recall_score(labels_test, predicted) * 100
    f1 = f1_score(labels_test, predicted) * 100
    roc_auc = roc_auc_score(labels_test, predicted) * 100
    print(
        f"Method {method_name}, A: {accuracy:.2f}, P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}, ROC-AUC: {roc_auc:.2f}")
    print(Counter(predicted))
    print(Counter(labels_test))


def print_metrics(df, feature_names):
    main_feature_names = [feature_name.split("_")[0] for feature_name in feature_names]
    print(f"Counter all labels: {Counter(df['label'].tolist())}")
    print(f"Data size: {len(df['index'].tolist())}")
    print(f"Features size: {len(feature_names)}, {Counter(main_feature_names)}")


def merge_csvs_and_filter_data():
    df_probes = pd.read_csv("data/svo_probes.csv", index_col="index")
    df_probes.drop(df_probes.index[df_probes["sentence"] == 'woman, ball, outside'], inplace=True)
    df_probes.drop(df_probes.index[df_probes["sentence"] == 'woman, music, notes'], inplace=True)
    # df_probes.drop(df_probes.index[df_probes["sentence"] == 'People on a team running.'], inplace=True)
    # df_probes.drop(df_probes.index[df_probes["sentence"] == 'People playing on a team.'], inplace=True)

    df_neg = pd.read_csv("data/neg_d.csv", header=0)
    df_neg.drop(df_neg.index[df_neg["neg_sentence"] == 'woman, ball, outside'], inplace=True)
    df_neg.drop(df_neg.index[df_neg["neg_sentence"] == 'woman, music, notes'], inplace=True)
    # df_neg.drop(df_neg.index[df_neg["neg_sentence"] == 'People on a team running.'], inplace=True)
    # df_neg.drop(df_neg.index[df_neg["neg_sentence"] == 'People playing on a team.'], inplace=True)

    result = pd.concat([df_probes, df_neg["neg_sentence"]], axis=1)
    result = result[result["sentence"].notna()]

    result.to_csv("data/merged.csv")


if __name__ == "__main__":
    # get_wnet_category(word='', pos='')
    # merge_csvs_and_filter_data()

    clip_results = read_SVO_CLIP_results_file()
    # get_bert_data(clip_results)
    df = get_all_properties(clip_results)
    # one_hot_pos, one_hot_levin, one_hot_liwc, concreteness, cosine_sim, feature_names = transform_features(df)
    # print_metrics(df, feature_names)
    #
    # features_scaled = pre_process_features(one_hot_pos, one_hot_levin, concreteness, cosine_sim)
    # feat_train, labels_train, feat_test, labels_test = eval_split(features_scaled, df)
    #
    # coef_weights = run_SVM(feat_train, labels_train, feat_test, labels_test)
    # print_sorted_coef_weights(coef_weights, feature_names)
    # # plot_coef_weights(coef_weights, feature_names)
    #
    # # majority_class(labels_test)
