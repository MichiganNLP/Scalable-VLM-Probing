#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
from collections import Counter
from functools import partial
from multiprocessing import Pool
from typing import Any, Collection, Iterable, Literal, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import svm
from sklearn.linear_model import Ridge
from statsmodels.regression.linear_model import RegressionResults
from tqdm.auto import tqdm

from features import is_feature_binary, is_feature_multi_label, load_features

CLASSIFICATION_MODELS = {"dominance-score", "svm"}
REGRESSION_MODELS = {"ols", "ridge"}
MODELS = CLASSIFICATION_MODELS | REGRESSION_MODELS

EXAMPLE_MODES = ["top", "sample", "disabled"]


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
    sorted_feature_names = feature_names[sorted_coefficients_idx].array
    sorted_feature_significance = coef_significance[sorted_coefficients_idx].array
    sorted_feature_sign = coef_sign[sorted_coefficients_idx].array

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


def compute_svm_regression(features: pd.DataFrame, labels: np.ndarray, iterations: int = 10_000) -> pd.DataFrame:
    coef_weights, coef_significance, coef_sign = _analyze_coef_weights_svm(features, labels, iterations)
    _print_sorted_coef_weights_svm(coef_weights, coef_significance, coef_sign, features)
    _plot_coef_weights_svm(coef_weights, features)
    return pd.DataFrame(coef_weights, columns=["coef"], index=features.columns)


def _value_contains_label(v: Any, label: str) -> bool:
    if isinstance(v, str):
        return v == label
    elif isinstance(v, Collection):
        return label in v
    else:
        raise ValueError(f"Unexpected value type: {type(v)}")


def obtain_top_examples_and_co_occurrences(feature_names: Iterable[str], raw_features: pd.DataFrame,
                                           max_word_count: int = 5,
                                           sample_size: int | None = None) -> Tuple[Sequence[str], Sequence[str]]:
    multi_label_features = {main_feature_name
                            for feature_name in feature_names
                            if ((main_feature_name := feature_name.split("_", maxsplit=1)[0]) in raw_features
                                and is_feature_multi_label(raw_features[main_feature_name]))}

    examples = []
    co_occurrence_examples = []

    for feature_name in tqdm(feature_names, desc="Computing examples and co-occurrences"):
        underscore_split = feature_name.split("_", maxsplit=1)
        if (main_feature_name := underscore_split[0]) in multi_label_features:
            label = underscore_split[1]

            main_feature_name_prefix, word_type = main_feature_name.split("-", maxsplit=1)
            if word_type in {"common", "original", "replacement"}:
                mask = raw_features[main_feature_name].map(lambda labels: label in labels)
                rows_with_label = raw_features[mask]

                if sample_size:
                    rows_with_label = rows_with_label.sample(min(sample_size, len(rows_with_label)))

                if word_type == "common":
                    lists_of_words_with_label = rows_with_label.apply(
                        lambda row: [w
                                     for i, w in enumerate(row["words-common"])
                                     if _value_contains_label(row[f"{main_feature_name_prefix}-common-{i}"], label)],
                        axis=1)
                    # We could also use `lists_of_words_with_label.explode()`, but this is likely faster:
                    words = (w for word_iter in lists_of_words_with_label for w in word_iter)

                    list_of_words_without_label = rows_with_label.apply(
                        lambda row: [w
                                     for i, w in enumerate(row["words-common"])
                                     if not _value_contains_label(row[f"{main_feature_name_prefix}-common-{i}"],
                                                                  label)], axis=1)
                    # We could also use `list_of_words_without_label.explode()`, but this is likely faster:
                    co_occurrence_words = (w for word_iter in list_of_words_without_label for w in word_iter)
                else:
                    words = rows_with_label[f"word-{word_type}"]
                    other_word_type = next(iter({"original", "replacement"} - {word_type}))
                    co_occurrence_words = itertools.chain((w for w in rows_with_label[f"word-{other_word_type}"]),
                                                          (w
                                                           for word_iter in rows_with_label[f"words-common"]
                                                           for w in word_iter))

                examples_str = ", ".join(f"{w} ({freq})" for w, freq in Counter(words).most_common(max_word_count))
                co_occurrence_example_str = ", ".join(
                    f"{w} ({freq})" for w, freq in Counter(co_occurrence_words).most_common(max_word_count))
            else:
                examples_str = ""
                co_occurrence_example_str = ""
        else:
            examples_str = ""
            co_occurrence_example_str = ""

        examples.append(examples_str)
        co_occurrence_examples.append(co_occurrence_example_str)

    return examples, co_occurrence_examples


def compute_ols_regression(features: pd.DataFrame, dependent_variable: np.ndarray, confidence: float = .95,
                           regularization: Literal["ridge", "lasso"] | None = None, alpha: float = 0.1) -> pd.DataFrame:
    features = sm.add_constant(features)

    model = sm.OLS(dependent_variable, features)

    if regularization:
        results = model.fit_regularized(L1_wt=0 if regularization == "ridge" else 1, alpha=alpha)
        print("R^2:", RegressionResults(model, results.params).rsquared)
        df = pd.DataFrame(results.params, columns=["coef"], index=features.columns)
    else:
        results = model.fit()
        summary = results.summary()
        print(summary)

        table_as_html = summary.tables[1].as_html()
        df = pd.read_html(table_as_html, header=0, index_col=0)[0]
        df = df[df["P>|t|"] <= (1 - confidence)]

        print()
        print()
        print(f"Features whose coefficient is significantly different from zero ({len(df)}):")

    return df


def compute_ridge_regression(features: pd.DataFrame, dependent_variable: np.ndarray,
                             alpha: float = 0.1) -> pd.DataFrame:
    model = Ridge(alpha=alpha)
    model.fit(features, dependent_variable)
    print("R^2:", model.score(features, dependent_variable))
    return pd.DataFrame(model.coef_, columns=["coef"], index=features.columns)


def compute_dominance_score(features: pd.DataFrame, dependent_variable: np.ndarray) -> pd.DataFrame:
    assert len(features) == len(dependent_variable)
    assert is_feature_binary(dependent_variable)

    total_pos = dependent_variable.sum()

    neg_labels = ~dependent_variable
    total_neg = neg_labels.sum()

    dominance_scores = {}

    for column in features.columns:
        feature = features[column]
        if is_feature_binary(feature):
            pos_coverage = feature[dependent_variable].sum() / total_pos
            neg_coverage = feature[neg_labels].sum() / total_neg
            dominance_scores[column] = pos_coverage / neg_coverage

    return pd.DataFrame(dominance_scores.values(), columns=["coef"], index=dominance_scores.keys())  # noqa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ols", choices=MODELS)
    parser.add_argument("--input-path", default="data/merged.csv")
    parser.add_argument("--dependent-variable-name")
    parser.add_argument("-r", "--remove-features", dest="feature_deny_list", nargs="+",
                        default={"wup-similarity", "lch-similarity", "path-similarity"})
    parser.add_argument("--feature-min-non-zero-values", type=int, default=50)
    parser.add_argument("--no-neg-features", dest="compute_neg_features", action="store_false")
    parser.add_argument("--merge-original-and-replacement-features", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--iterations", type=int, default=10_000, help="Only applies to the SVM model.")
    parser.add_argument("--examples", choices=EXAMPLE_MODES, default="top")
    args = parser.parse_args()

    args.dependent_variable_name = (args.dependent_variable_name
                                    or ("clip_score_diff" if args.model in REGRESSION_MODELS else "clip prediction"))
    args.feature_deny_list = set(args.feature_deny_list)

    assert args.compute_neg_features or not args.merge_original_and_replacement_features, \
        "Cannot merge original and replacement features if neg features are not computed."

    return args


def main() -> None:
    args = parse_args()

    print("Disabled features:", args.feature_deny_list)

    raw_features, features, dependent_variable = load_features(
        path=args.input_path, dependent_variable_name=args.dependent_variable_name,
        max_feature_count=1000 if args.debug else None, feature_deny_list=args.feature_deny_list,
        standardize_dependent_variable=args.model in REGRESSION_MODELS,
        standardize_binary_features=args.model in REGRESSION_MODELS,
        compute_neg_features=args.compute_neg_features,
        compute_similarity_features=args.model in REGRESSION_MODELS,
        merge_original_and_replacement_features=args.merge_original_and_replacement_features,
        feature_min_non_zero_values=args.feature_min_non_zero_values)

    if args.model == "dominance-score":
        df = compute_dominance_score(features, dependent_variable)
    elif args.model == "ols":
        df = compute_ols_regression(features, dependent_variable)
    elif args.model == "ridge":
        df = compute_ridge_regression(features, dependent_variable)
    elif args.model == "svm":
        df = compute_svm_regression(features, dependent_variable, iterations=args.iterations)
    else:
        raise ValueError(f"Unknown model: {args.model} (should be in {MODELS}).")

    df = df.sort_values(by=["coef"], ascending=False)

    if args.examples != "disabled":
        if args.examples == "top":
            sample_size = None
        elif args.examples == "sample":
            sample_size = 100
        else:
            raise ValueError(f"Unknown examples mode: {args.examples} (should be in {EXAMPLE_MODES}).")

        (df["examples"],
         df["co-occurring word examples"]) = obtain_top_examples_and_co_occurrences(df.index, raw_features,
                                                                                    sample_size=sample_size)

    print(df.to_string())


if __name__ == "__main__":
    main()
