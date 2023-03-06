import json
import re
from collections import Counter
from typing import Sequence

import pandas as pd


def read_laion():
    data = pd.read_parquet("data/part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet", engine="auto")
    captions = data["TEXT"]
    captions.to_csv('data/LAION_captions.csv')
    return captions

def preproces_captions(captions) -> None:
    # captions["TEXT"] = captions["TEXT"].str.replace('[^\w\s]', ' ') # remove punctuation
    captions["TEXT"] = captions["TEXT"].replace(r'[^a-zA-Z ]', ' ', regex=True).replace("'", '')# remove punctuation
    captions["TEXT"] = captions["TEXT"].str.lower() #lowercase
    captions.to_csv('data/LAION_captions_prep.csv')


def count_words() -> None:
    word_captions_LAION = [word for sentence in
                           pd.read_csv('data/LAION_captions_prep.csv')['TEXT'].dropna().values.tolist()
                           for word in sentence.split()]
    counter_word_LAION = Counter(word_captions_LAION)
    with open("data/words_counter_LAION.json", "w") as f:
        json.dump(counter_word_LAION, f)

def main() -> None:
    # captions = pd.read_csv('data/LAION_captions.csv')
    # preproces_captions(captions)
    count_words()


if __name__ == '__main__':
    main()