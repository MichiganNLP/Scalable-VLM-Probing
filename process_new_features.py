from collections import defaultdict
from typing import Mapping, Sequence

import pandas as pd
from pandas.core.dtypes.inference import is_bool, is_float


def read_general_inq(path: str = "data/inquireraugmented.xls") -> Mapping[str, Sequence[str]]:
    data = pd.read_excel(path, index_col=0)
    classes = list(data.columns)
    word_to_classes = defaultdict(list)
    for class_name in classes[1:-2]:
        for word in data[class_name][1:].index:
            if not is_float(data[class_name][word]) and not is_bool(word):
                word_to_classes[word.lower()].append(class_name)
    return word_to_classes


def main() -> None:
    print(read_general_inq())


if __name__ == "__main__":
    main()
