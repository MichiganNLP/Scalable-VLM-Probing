#!/usr/bin/env python
import itertools
import json
import re
from collections import Counter
from typing import Iterable

from datasets import load_dataset
from tqdm.auto import tqdm


def load_laion_texts() -> Iterable[str]:
    for instance in load_dataset("laion/laion400m", split="train", streaming=True):
        if text := instance["TEXT"]:
            yield re.sub(r'[^a-zA-Z ]', ' ', text.lower()).replace("'", '')


def main() -> None:
    max_count = 10_000_000  # This is a reasonable number, and it's also roughly what LAION's 1st parquet file has.
    word_counts = Counter(word
                          for text in tqdm(itertools.islice(load_laion_texts(), max_count), total=max_count)
                          for word in text.split())
    with open("data/words_counter_LAION.json", "w") as f:
        json.dump(word_counts, f)


if __name__ == "__main__":
    main()
