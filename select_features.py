#!/usr/bin/env python
import sys

import pandas as pd


def main() -> None:
    df = pd.read_csv(sys.argv[1])
    print(df[(df.Significance >= 9500) & (df["Weight (abs)"] > 0)].to_csv(index=False))


if __name__ == "__main__":
    main()
