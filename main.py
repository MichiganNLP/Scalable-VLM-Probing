#!/usr/bin/env python
import argparse
from collections import Counter
from functools import partial
from multiprocessing import Pool
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import svm
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

from features import is_feature_binary, load_features

CLASSIFICATION_MODELS = {"dominance-score"}
REGRESSION_MODELS = {"ols", "ridge", "svm"}
MODELS = CLASSIFICATION_MODELS | REGRESSION_MODELS


def _plot_coef_weights_svm(coef_weights: np.ndarray, features: pd.DataFrame,
                           path: str = "data/coef_importance.png", top_features: int = 5) -> None:
    coef = coef_weights.ravel()  # flatten array
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    fig = plt.figure(figsize=(18, 7))
    colors = ["red" if c < 0 else "green" for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = features.columns
    plt.xticks(np.arange(2 * top_features), feature_names[top_coefficients], rotation=45, ha="right")
    fig.savefig(path, bbox_inches="tight")


# https://www.kaggle.com/code/pierpaolo28/pima-indians-diabetes-database/notebook
def _print_sorted_coef_weights_svm(coef: np.ndarray, coef_significance: np.ndarray, coef_sign: np.ndarray,
                                   features: pd.DataFrame, features_count: Sequence[int],
                                   output_path: str = "data/sorted_features.csv") -> None:
    sorted_coefficients_idx = np.argsort(coef)[::-1]  # in descending order
    sorted_coefficients = [np.round(weight, 2) for weight in coef[sorted_coefficients_idx]]

    feature_names = features.columns
    sorted_feature_names = feature_names[sorted_coefficients_idx].tolist()
    sorted_feature_significance = coef_significance[sorted_coefficients_idx].tolist()
    sorted_feature_sign = coef_sign[sorted_coefficients_idx].tolist()
    sorted_feature_counts = [features_count.count(feature.split("_")[1]) for feature in sorted_feature_names]

    df = pd.DataFrame(
        zip(sorted_feature_names, sorted_feature_significance, sorted_feature_counts, sorted_coefficients,
            sorted_feature_sign),
        columns=["Feature", "Significance", "Data Count", "Weight (abs)", "Weight sign"])
    df.to_csv(output_path, index=False)


def _build_classifier_svm() -> svm.LinearSVC:
    return svm.LinearSVC(class_weight="balanced", max_iter=1_000_000)


def _classify_shuffled_svm(features: pd.DataFrame, labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    features = features.sample(frac=1, random_state=rng)

    clf = _build_classifier_svm()
    clf.fit(features, labels)

    return abs(clf.coef_.ravel())


def _analyze_coef_weights_svm(features: pd.DataFrame, labels: np.ndarray,
                              iterations: int = 10_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    clf = _build_classifier_svm()

    print("Computing the coefficients with the real features…")
    clf.fit(features, labels)
    print("Coefficients computed.")

    coef_weights = clf.coef_.ravel()
    coef_sign = np.sign(coef_weights)
    coef_weights = abs(coef_weights)

    with Pool() as pool:
        list_shuffled_coef_weights = list(tqdm(
            pool.imap_unordered(partial(_classify_shuffled_svm, features, labels), range(iterations)),
            total=iterations, desc="Computing the coefficients with shuffled columns"))

    coef_significance = np.array([sum(list_coef[i] <= coef for list_coef in list_shuffled_coef_weights)
                                  for i, coef in enumerate(coef_weights)])

    return coef_weights, coef_significance, coef_sign


def compute_svm_regression(features: pd.DataFrame, labels: np.ndarray, features_count: Sequence[int],
                           iterations: int) -> None:
    coef_weights, coef_significance, coef_sign = _analyze_coef_weights_svm(features, labels, iterations)
    _print_sorted_coef_weights_svm(coef_weights, coef_significance, coef_sign, features, features_count)
    _plot_coef_weights_svm(coef_weights, features)


def compute_ols_regression(features: pd.DataFrame, labels: np.ndarray, raw_features: pd.DataFrame) -> None:
    features = sm.add_constant(features)

    model = sm.OLS(labels, features)
    results = model.fit()
    summary = results.summary()
    print(summary)
    print()
    print()

    table_as_html = summary.tables[1].as_html()
    df = pd.read_html(table_as_html, header=0, index_col=0)[0]

    df = df[df["P>|t|"] <= .05]
    df = df.sort_values(by=["coef"], ascending=False)

    print("Significant features:")
    print(df.to_string())

    print()
    print("Example words from the significant features:")
    for feature_name in df.index:
        prefix = feature_name.split("_", maxsplit=1)[0]
        if prefix in {"LIWC", "Levin"}:
            category = feature_name.split("_", maxsplit=2)[-1]

            original_words = raw_features[raw_features[f"{prefix}-original"].apply(
                lambda categories: category in categories)].word_original
            replacement_words = raw_features[raw_features[f"{prefix}-replacement"].apply(
                lambda categories: category in categories)].word_replacement
            words = pd.concat([original_words, replacement_words])
            counter = Counter(words.tolist())

            print(feature_name, "--", ", ".join(f"{word} ({freq})" for word, freq in counter.most_common(5)))


def compute_ridge_regression(features: pd.DataFrame, labels: np.ndarray) -> None:
    clf = Ridge(alpha=0.1)
    clf.fit(features, labels)
    r_squared = clf.score(features, labels)
    print("Ridge regression R^2:", r_squared)
    coef = clf.coef_
    df = pd.DataFrame.from_dict({"features": features.columns, "coef": coef})
    df = df.sort_values(by=["coef"], ascending=False)
    print(df.to_string())


def compute_dominance_score(features: pd.DataFrame, labels: np.ndarray) -> None:
    assert len(features) == len(labels)
    assert is_feature_binary(labels)

    total_pos = labels.sum()

    neg_labels = ~labels
    total_neg = neg_labels.sum()

    dominance_scores = {}

    for column in features.columns:
        feature = features[column]
        if is_feature_binary(feature):
            pos_coverage = feature[labels].sum() / total_pos
            neg_coverage = feature[neg_labels].sum() / total_neg
            dominance_scores[column] = pos_coverage / neg_coverage

    print("Dominance scores:")
    for column, score in sorted(dominance_scores.items(), key=lambda x: x[1], reverse=True):
        print(score, column)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ols", choices=MODELS)
    parser.add_argument("--input-path", default="data/merged.csv")
    parser.add_argument("--feature-min-non-zero-values", type=int, default=50)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--iterations", type=int, default=10_000, help="Only applies to the SVM model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    max_feature_count = 1000 if args.debug else None

    do_regression = args.model in REGRESSION_MODELS
    merge_original_and_replacement_features = do_regression

    raw_features, features, features_count, labels = load_features(
        path=args.input_path, max_feature_count=max_feature_count,
        merge_original_and_replacement_features=merge_original_and_replacement_features, do_regression=do_regression,
        feature_min_non_zero_values=args.feature_min_non_zero_values)

    if args.model == "dominance-score":
        compute_dominance_score(features, labels)
    elif args.model == "ols":
        compute_ols_regression(features, labels, raw_features)
    elif args.model == "ridge":
        compute_ridge_regression(features, labels)
    elif args.model == "svm":
        compute_svm_regression(features, labels, features_count, iterations=args.iterations)
    else:
        raise ValueError(f"Unknown model: {args.model} (should be in {MODELS}).")


if __name__ == "__main__":
    main()
