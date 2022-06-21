import pandas as pd


def read_results_file():
    df = pd.read_csv("data/svo_probes.csv")

    results = []
    for index, sentence, pos_triplet, neg_triplet, neg_type, clip_prediction in zip(df['index'], df['sentence'],
                                                                                    df['pos_triplet'],
                                                                                    df['neg_triplet'],
                                                                                    df['neg_type'],
                                                                                    df['clip prediction']):
        results.append([index, sentence, pos_triplet, neg_triplet, neg_type, clip_prediction])

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
                    levin_dict[key] = content.split()
                    content = ""
                else:
                    content += line.replace('\r\n', "").rstrip()
                    content += " "
    return levin_dict

def get_levin_class_per_verb(verb):
    levin_dict = parse_levin_file()
    levin_classes = []
    for key, values in levin_dict.items():
        if verb in values:
            levin_classes.append(key)
    return levin_classes

def get_verb_properties(verb):
    levin_classes = get_levin_class_per_verb(verb)


def get_features(results):
    dict_features = {}
    for index, sentence, pos_triplet, neg_triplet, neg_type, clip_prediction in results:
        if neg_type == 'subj':
            one_hot = '100'
        elif neg_type == 'verb':
            one_hot = '010'
        else:
            one_hot = '001'
    dict_features[index] = {"sentence": sentence, "clip_prediction": clip_prediction, "features": [one_hot]}


if __name__ == "__main__":
    get_verb_properties(verb='iron')

    # results = read_results_file()
    # get_features(results)
