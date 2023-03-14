#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
from collections import Counter
from typing import Any, Collection, Iterable, Literal, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.tools.tools import pinv_extended
from tqdm.auto import tqdm

from features import VALID_LEVIN_RETURN_MODES, is_feature_binary, is_feature_multi_label, load_features

CLASSIFICATION_MODELS = {"dominance-score", "sklearn-clf"}
REGRESSION_MODELS = {"mean-diff-and-corr", "lasso", "ols", "ridge", "sklearn"}
MODELS = CLASSIFICATION_MODELS | REGRESSION_MODELS

EXAMPLE_MODES = ["top", "sample", "disabled"]


pd.options.display.float_format = "{:,.3f}".format


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
    multi_label_features = {main_name
                            for name in feature_names
                            if ((main_name := name.split("_", maxsplit=1)[0]) in raw_features
                                # We can just use a single value to infer the type:
                                and is_feature_multi_label(raw_features.loc[raw_features.index[:1], main_name]))}

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

                    lists_of_words_without_label = rows_with_label.apply(
                        lambda row: [w
                                     for i, w in enumerate(row["words-common"])
                                     if not _value_contains_label(row[f"{main_feature_name_prefix}-common-{i}"],
                                                                  label)], axis=1)
                    # We could also use `lists_of_words_without_label.explode()`, but this is likely faster:
                    co_occurrence_words = (w for word_iter in lists_of_words_without_label for w in word_iter)
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


def compute_ols_regression(features: pd.DataFrame, dependent_variable: pd.Series,
                           regularization: Literal["ridge", "lasso"] | None = None, alpha: float = 1.0) -> pd.DataFrame:
    model = sm.OLS(dependent_variable, features)

    if regularization:
        alpha /= len(features)  # See https://stackoverflow.com/a/72260809/1165181
        results = model.fit_regularized(L1_wt=int(regularization == "lasso"), alpha=alpha)
    else:
        results = model.fit()

    try:
        summary = results.summary()
    except NotImplementedError:
        summary = None

    if summary:
        print(summary)
        df = pd.read_html(summary.tables[1].as_html(), header=0, index_col=0)[0]
    else:
        print("R^2:", RegressionResults(model, results.params).rsquared)
        df = pd.DataFrame(results.params, columns=["coef"], index=features.columns)

    return df


def compute_sklearn_regression(features: pd.DataFrame, dependent_variable: pd.Series) -> pd.DataFrame:
    model = RandomForestRegressor(n_jobs=-1, verbose=1)
    model.fit(features, dependent_variable)
    print("R^2:", model.score(features, dependent_variable))
    return pd.DataFrame(model.feature_importances_, columns=["coef"], index=features.columns)


def compute_dominance_score(features: pd.DataFrame, dependent_variable: pd.Series) -> pd.DataFrame:
    assert len(features) == len(dependent_variable)
    assert is_feature_binary(dependent_variable)

    total_pos = dependent_variable.sum()

    neg_labels = ~dependent_variable
    total_neg = neg_labels.sum()

    dominance_scores = {}

    for column_name in features.columns:
        feature = features[column_name]
        if is_feature_binary(feature):
            pos_coverage = feature[dependent_variable].sum() / total_pos
            neg_coverage = feature[neg_labels].sum() / total_neg
            dominance_scores[column_name] = pos_coverage / neg_coverage

    return pd.DataFrame(dominance_scores.values(), columns=["coef"], index=dominance_scores.keys())  # noqa


def compute_sklearn_clf(features: pd.DataFrame, dependent_variable: pd.Series) -> pd.DataFrame:
    clf = svm.LinearSVC(class_weight="balanced", max_iter=1_000_000)
    clf.fit(features, dependent_variable)
    return pd.DataFrame(clf.coef_, columns=["coef"], index=features.columns)


def compute_mean_diff_and_corr(features: pd.DataFrame, dependent_variable: pd.Series,
                               confidence: float = .95) -> pd.DataFrame:
    assert len(features) == len(dependent_variable)

    coef_type = {}
    score = {}
    std_err = {}
    t = {}
    p = {}
    lower_bound = {}
    upper_bound = {}

    for feature_name in tqdm(features.columns, desc="Computing mean diff and corr"):
        feature = features[feature_name]
        if is_feature_binary(feature):
            coef_type[feature_name] = "diff"

            feature = feature.astype(bool)

            pos_group = dependent_variable[feature]
            neg_group = dependent_variable[~feature]

            t[feature_name], p[feature_name] = stats.ttest_ind(pos_group, neg_group, equal_var=False)

            # The following code was adapted from `stats.ttest_ind`:

            score[feature_name] = pos_group.mean() - neg_group.mean()

            pos_group_var = pos_group.var(ddof=1)
            neg_group_var = neg_group.var(ddof=1)

            pos_group_size = len(pos_group)
            neg_group_size = len(neg_group)

            pos_group_vn = pos_group_var / pos_group_size
            neg_group_vn = neg_group_var / neg_group_size

            std_err[feature_name] = np.sqrt(pos_group_vn + neg_group_vn)

            with np.errstate(divide="ignore", invalid="ignore"):
                df = (pos_group_vn + neg_group_vn) ** 2 / (pos_group_vn ** 2 / (pos_group_size - 1)
                                                           + neg_group_vn ** 2 / (neg_group_size - 1))

            # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
            # Hence, it doesn't matter what df is as long as it's not NaN.
            df = np.where(np.isnan(df), 1, df)

            half_interval_size = stats.t.ppf(confidence + (1 - confidence) / 2, df) * std_err[feature_name]

            lower_bound[feature_name] = score[feature_name] - half_interval_size
            upper_bound[feature_name] = score[feature_name] + half_interval_size
        elif np.issubdtype(feature.dtype, np.number):
            coef_type[feature_name] = "corr"

            corr_result = stats.pearsonr(feature, dependent_variable)
            score[feature_name] = corr_result.statistic
            p[feature_name] = corr_result.pvalue
            lower_bound[feature_name], upper_bound[feature_name] = corr_result.confidence_interval(confidence)

            std_err[feature_name] = np.sqrt((1 - score[feature_name] ** 2) / (len(feature) - 2))
            t[feature_name] = score[feature_name] / std_err[feature_name]

    df = pd.DataFrame({"coef-type": coef_type.values(), "coef": score.values(), "std err": std_err.values(),
                       "t": t.values(), "P>|t|": p.values(), f"[{(1 - confidence) / 2:.3f}": lower_bound.values(),
                       f"{(confidence + (1 - confidence) / 2):.3f}]": upper_bound.values()}, index=t.keys())  # noqa

    print(df.to_string())

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ols", choices=MODELS)
    parser.add_argument("--input-path", default="data/merged.csv")

    parser.add_argument("--max-data-count", type=int)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--dependent-variable-name")
    parser.add_argument("-r", "--remove-features", dest="feature_deny_list", nargs="+",
                        default={"wup-similarity", "lch-similarity", "path-similarity"})
    parser.add_argument("--feature-min-non-zero-values", type=int, default=50)
    parser.add_argument("--no-neg-features", dest="compute_neg_features", action="store_false")
    parser.add_argument("--levin-return-mode", choices=VALID_LEVIN_RETURN_MODES, default="semantic_fine_grained")
    parser.add_argument("--merge-original-and-replacement-features", action="store_true")
    parser.add_argument("--no-remove-correlated-features", dest="remove_correlated_features", action="store_false")
    parser.add_argument("---feature-correlation-keep-threshold", type=float, default=.8)
    parser.add_argument("--do-vif", action="store_true")

    parser.add_argument("--alpha", type=float, default=1, help="Only applies to the ridge regression model.")

    parser.add_argument("--iterations", type=int, default=10_000, help="Only applies to the SVM model.")

    parser.add_argument("--confidence", type=float, default=.95)

    parser.add_argument("--examples", choices=EXAMPLE_MODES, default="top")

    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    assert args.max_data_count is None or not args.debug, "Cannot specify max data count in debug mode."
    args.max_data_count = 1000 if args.debug else args.max_data_count

    args.dependent_variable_name = (args.dependent_variable_name
                                    or ("clip_score_diff" if args.model in REGRESSION_MODELS else "clip prediction"))
    args.feature_deny_list = set(args.feature_deny_list)

    assert args.compute_neg_features or not args.merge_original_and_replacement_features, \
        "Cannot merge original and replacement features if neg features are not computed."

    args.do_standardization = args.model in {"lasso", "ols", "ridge"}

    return args


def main() -> None:
    args = parse_args()

    print(args)

    raw_features, features, dependent_variable = load_features(
        path=args.input_path, dependent_variable_name=args.dependent_variable_name,
        max_data_count=args.max_data_count, feature_deny_list=args.feature_deny_list,
        standardize_dependent_variable=args.do_standardization,
        standardize_binary_features=args.do_standardization,
        compute_neg_features=args.compute_neg_features, levin_return_mode=args.levin_return_mode,
        compute_similarity_features=args.model in REGRESSION_MODELS,
        merge_original_and_replacement_features=args.merge_original_and_replacement_features,
        remove_correlated_features=args.remove_correlated_features,
        feature_correlation_keep_threshold=args.feature_correlation_keep_threshold, do_vif=args.do_vif,
        feature_min_non_zero_values=args.feature_min_non_zero_values)

    if args.model in {"ols", "ridge", "lasso"}:
        regularization = {"ols": None}.get(args.model, args.model)
        df = compute_ols_regression(features, dependent_variable, regularization=regularization, alpha=args.alpha)
    elif args.model == "sklearn":
        df = compute_sklearn_regression(features, dependent_variable)
    elif args.model == "dominance-score":
        df = compute_dominance_score(features, dependent_variable)
    elif args.model == "sklearn-clf":
        df = compute_sklearn_clf(features, dependent_variable)
    elif args.model == "mean-diff-and-corr":
        df = compute_mean_diff_and_corr(features, dependent_variable)
    else:
        raise ValueError(f"Unknown model: {args.model} (should be in {MODELS}).")

    df = df.sort_values(by=["coef"], ascending=False)

    confidence = args.confidence

    if "P>|t|" not in df.columns:
        pinv = pinv_extended(features[df.index])[0]
        normalized_cov_params = pinv @ pinv.T
        results = LikelihoodModelResults(None, df.coef, normalized_cov_params=normalized_cov_params)
        df["std err"] = results.bse
        df["t"] = results.tvalues
        df["P>|t|"] = results.pvalues
        confidence_intervals = results.conf_int(alpha=(1 - args.confidence))
        df[f"[{(1 - confidence) / 2:.3f}"] = confidence_intervals[:, 0]
        df[f"{(confidence + (1 - confidence) / 2):.3f}]"] = confidence_intervals[:, 1]

        print(df.to_string())

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

    df = df[df["P>|t|"] <= (1 - args.confidence)]
    print()
    print()
    print(f"Features whose coefficient is significantly different from zero ({len(df)}):")

    print(df.to_string())

    df.to_csv(f"data/output_{args.dependent_variable_name}.csv")

    if args.plot:
        top_k = 10
        top_df = pd.concat([df.iloc[:top_k], df[-top_k:]])

        df_to_plot = top_df.reset_index(names="feature")
        # Hack to get error bars (just one datapoint per feature would not call the function):
        df_to_plot = pd.concat([df_to_plot, df_to_plot], ignore_index=True)

        def _error_bar(x: pd.Series) -> Tuple[float, float]:
            return tuple(df_to_plot.loc[x.index[0]][[f"[{(1 - confidence) / 2:.3f}",
                                                     f"{(confidence + (1 - confidence) / 2):.3f}]"]])

        good_color, bad_color = sns.color_palette("deep", 4)[-2:]
        sns.catplot(data=df_to_plot, x="coef", y="feature", errorbar=_error_bar, kind="point", join=False, aspect=1.5,
                    palette=[good_color] * top_k + [bad_color] * top_k)
        plt.show()

        # Hack to undo the standardization:
        non_standardized_features = features.copy()
        non_standardized_features[features == features.min()] = 0
        non_standardized_features[features == features.max()] = 1
        non_standardized_features = non_standardized_features.astype(int)

        binary_feature_names = [feature_name
                                for feature_name in top_df.index[:2]
                                if is_feature_binary(non_standardized_features[feature_name])]
        binary_features = non_standardized_features[binary_feature_names]
        non_standardized_dependent_variable = raw_features[args.dependent_variable_name]
        repeated_dependent_variable = pd.concat([non_standardized_dependent_variable] * len(binary_features.columns),
                                                ignore_index=True)
        df_to_plot2 = pd.concat([binary_features.melt(var_name="feature"), repeated_dependent_variable], axis="columns")

        sns.catplot(data=df_to_plot2, x=args.dependent_variable_name, y="feature", hue="value", kind="box", aspect=1.5)
        plt.show()

        df_to_plot3_1 = raw_features[["concreteness-common", args.dependent_variable_name]].copy()
        df_to_plot3_1 = df_to_plot3_1.sort_values(by="concreteness-common")
        df_to_plot3_1["type"] = "original"

        sns.regplot(data=df_to_plot3_1, x="concreteness-common", y=args.dependent_variable_name,
                    line_kws={"color": "salmon"})
        plt.show()

        df_to_plot3_2 = df_to_plot3_1.copy()
        df_to_plot3_2[args.dependent_variable_name] = scipy.signal.savgol_filter(
            df_to_plot3_2[args.dependent_variable_name], window_length=1000, polyorder=3)
        df_to_plot3_2["type"] = "smoothed"

        df_to_plot3 = pd.concat([df_to_plot3_1, df_to_plot3_2], ignore_index=True)

        sns.relplot(data=df_to_plot3, x="concreteness-common", y=args.dependent_variable_name, hue="type", kind="line")
        plt.show()

        sns.displot(data=raw_features, x="frequency-common", y=args.dependent_variable_name, kind="kde",
                    log_scale=[True, False])
        plt.show()

        sns.displot(data=raw_features, x=args.dependent_variable_name, kind="kde")
        plt.show()


if __name__ == "__main__":
    main()
