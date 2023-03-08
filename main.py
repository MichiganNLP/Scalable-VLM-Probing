#!/usr/bin/env python
import argparse
from collections import Counter
from functools import partial
from multiprocessing import Pool
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import svm
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

from features import is_feature_binary, is_feature_multi_label, load_features

CLASSIFICATION_MODELS = {"dominance-score", "svm"}
REGRESSION_MODELS = {"ols", "ridge"}
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
                                   features: pd.DataFrame, output_path: str = "data/sorted_features.csv") -> None:
    sorted_coefficients_idx = np.argsort(coef)[::-1]  # In descending order.
    sorted_coefficients = [np.round(weight, 2) for weight in coef[sorted_coefficients_idx]]

    feature_names = features.columns
    sorted_feature_names = feature_names[sorted_coefficients_idx].tolist()
    sorted_feature_significance = coef_significance[sorted_coefficients_idx].tolist()
    sorted_feature_sign = coef_sign[sorted_coefficients_idx].tolist()

    sorted_features = features[sorted_feature_names]
    sorted_feature_counts = (sorted_features != 0 & sorted_features.notna()).sum(axis=0)

    df = pd.DataFrame(
        zip(sorted_feature_names, sorted_feature_significance, sorted_feature_counts, sorted_coefficients,
            sorted_feature_sign),
        columns=["Feature", "Significance", "Not zero nor NaN", "Weight (abs)", "Weight sign"])
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


def compute_svm_regression(features: pd.DataFrame, labels: np.ndarray, iterations: int) -> None:
    coef_weights, coef_significance, coef_sign = _analyze_coef_weights_svm(features, labels, iterations)
    _print_sorted_coef_weights_svm(coef_weights, coef_significance, coef_sign, features)
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

    multi_label_features = {feature_name.split("-", maxsplit=1)[0]
                            for feature_name in df.index
                            if (feature_name.split("_", maxsplit=1)[0] in raw_features
                                and is_feature_multi_label(raw_features[feature_name.split("_", maxsplit=1)[0]]))}

    df["Examples"] = ""

    for feature_name in df.index:
        prefix = feature_name.split("-", maxsplit=1)[0]
        if prefix in multi_label_features:
            suffix = feature_name.split("_", maxsplit=2)[-1]

            original_words = raw_features[raw_features[f"{prefix}-original"].apply(
                lambda labels: suffix in labels)].word_original
            replacement_words = raw_features[raw_features[f"{prefix}-replacement"].apply(
                lambda labels: suffix in labels)].word_replacement
            words = pd.concat([original_words, replacement_words])
            counter = Counter(words.tolist())

            df.loc[feature_name, "Examples"] = ", ".join(f"{word} ({freq})" for word, freq in counter.most_common(5))

    print("Significant features:")
    print(df.to_string())


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
    parser.add_argument("-r", "--remove-features", dest="feature_deny_list", nargs="+",
                        default={"wup_similarity", "lch_similarity", "path_similarity"})
    parser.add_argument("--feature-min-non-zero-values", type=int, default=50)
    parser.add_argument("--merge-original-and-replacement-features", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--iterations", type=int, default=10_000, help="Only applies to the SVM model.")
    args = parser.parse_args()

    args.feature_deny_list = set(args.feature_deny_list)

    return args


def main() -> None:
    args = parse_args()

    print("Disabled features:", args.feature_deny_list)

    raw_features, features, labels = load_features(
        path=args.input_path, max_feature_count=1000 if args.debug else None,
        feature_deny_list=args.feature_deny_list,
        merge_original_and_replacement_features=args.merge_original_and_replacement_features,
        do_regression=args.model in REGRESSION_MODELS, feature_min_non_zero_values=args.feature_min_non_zero_values)

    if args.model == "dominance-score":
        compute_dominance_score(features, labels)
    elif args.model == "ols":
        compute_ols_regression(features, labels, raw_features)
    elif args.model == "ridge":
        compute_ridge_regression(features, labels)
    elif args.model == "svm":
        compute_svm_regression(features, labels, iterations=args.iterations)
    else:
        raise ValueError(f"Unknown model: {args.model} (should be in {MODELS}).")


if __name__ == "__main__":
    main()
