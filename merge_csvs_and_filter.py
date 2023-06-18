#!/usr/bin/env python
import argparse

import pandas as pd

from argparse_with_defaults import ArgumentParserWithDefaults


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("--probes-path", default="data/svo_probes_with_scores.csv")
    parser.add_argument("--neg-path", default="data/neg_d.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df_probes = pd.read_csv(args.probes_path, index_col="index")

    df_probes = df_probes[df_probes.sentence != "woman, ball, outside"]
    df_probes = df_probes[df_probes.sentence != "woman, music, notes"]

    df_neg = pd.read_csv(args.neg_path, header=0)
    df_neg = df_neg[df_neg.sentence != "woman, ball, outside"]
    df_neg = df_neg[df_neg.sentence != "woman, music, notes"]

    result = pd.concat([df_probes, df_neg.neg_sentence], axis=1)
    result = result[result.sentence.notna()]

    print(result.to_csv())


if __name__ == "__main__":
    main()
