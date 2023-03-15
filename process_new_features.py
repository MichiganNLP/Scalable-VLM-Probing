import pandas as pd
from pandas.core.dtypes.inference import is_float, is_bool


def read_generalinq():
    data = pd.read_excel('/home/oana/Desktop/CLIP_Probes/GeneralInquirel/inquireraugmented.xls', index_col=0)
    classes = list(data.columns)
    dict_general = {}
    for class_name in classes[1:-2]:
        words = [word.lower() for word in data[class_name][1:].index if not is_float(data[class_name][word]) and not is_bool(word)]
        for word in words:
            if word not in dict_general:
                dict_general[word] = []
            dict_general[word].append(class_name)
    return dict_general

def read_rouge():
    return

def main() -> None:
    dict_general = read_generalinq()
    dict_rouge = read_rouge()
    print(dict_rouge)


if __name__ == "__main__":
    main()