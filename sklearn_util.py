from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.impute._base import _check_inputs_dtype
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import _is_pandas_na, is_scalar_nan
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted


class SelectMinBinaryUniqueValues(SelectorMixin, BaseEstimator):
    def __init__(self, binary_feature_min_unique_values: int = 50, leave_at_least_one: bool = True) -> None:
        self.binary_feature_min_unique_values = binary_feature_min_unique_values
        self.leave_at_least_one = leave_at_least_one

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> SelectMinBinaryUniqueValues:  # noqa
        assert np.unique(X).size <= 2  # Only binary.

        X = self._validate_data(X, ensure_2d=True)

        non_zero = (X != 0).sum(axis=0)  # For sparse arrays, it's efficient to check the non-zero elements.
        self.min_counts_ = np.minimum(non_zero, X.shape[0] - non_zero)
        if isinstance(self.min_counts_, np.matrix):  # Can happen with CSR matrices.
            self.min_counts_ = self.min_counts_.A1  # noqa
        return self

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self)
        mask = self.min_counts_ >= self.binary_feature_min_unique_values

        if self.leave_at_least_one and not mask.any():
            # We do this because, with sklearn-pandas, when we use a `MultiLabelBinarizer` (because they are
            # transformed one by one), there may be no features left afterward and the next transformers in the
            # pipeline may fail for that multi-label feature.
            mask[self.min_counts_.argmax()] = True

        return mask


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Wraps `MultiLabelBinarizer` in a form that can work with `ColumnTransformer`. It makes it accept multiple inputs.

    Note that the input `X` has to be a `pandas.DataFrame`.
    """

    def __init__(self, binarizer_creator: Callable[[], Any] | None = None, dtype: npt.DTypeLike | None = None) -> None:
        self.binarizer_creator = binarizer_creator or MultiLabelBinarizer
        self.dtype = dtype

        self.binarizers = []
        self.categories_ = self.classes_ = []
        self.columns = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> MultiHotEncoder:  # noqa
        self.columns = X.columns.to_list()

        for column_name in X:
            binarizer = self.binarizer_creator().fit(X[column_name])
            self.binarizers.append(binarizer)
            self.classes_.append(binarizer.classes_)  # noqa

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)

        if len(self.classes_) != X.shape[1]:
            raise ValueError(f"The fit transformer deals with {len(self.classes_)} columns "
                             f"while the input has {X.shape[1]}.")

        return np.concatenate([binarizer.transform(X[c]).astype(self.dtype)
                               for c, binarizer in zip(X, self.binarizers)], axis=1)

    def get_feature_names_out(self, input_features: Sequence[str] = None) -> np.ndarray:
        check_is_fitted(self)

        cats = self.categories_

        if input_features is None:
            input_features = self.columns
        elif len(input_features) != len(self.categories_):
            raise ValueError(f"input_features should have length equal to number of features ({len(self.categories_)}),"
                             f" got {len(input_features)}")

        return np.asarray([input_features[i] + "_" + str(t) for i in range(len(cats)) for t in cats[i]])


class BoolImputer(SimpleImputer):
    """Impute missing values from a boolean array.

    It doesn't crash when there aren't any missing values. See https://github.com/scikit-learn/scikit-learn/issues/26292
    """

    def _validate_input(self, X, in_fit):
        if self.strategy in ("most_frequent", "constant"):
            # If input is a list of strings, dtype = object.
            # Otherwise ValueError is raised in SimpleImputer
            # with strategy='most_frequent' or 'constant'
            # because the list is converted to Unicode numpy array
            if isinstance(X, list) and any(
                isinstance(elem, str) for row in X for elem in row
            ):
                dtype = object
            else:
                dtype = None
        else:
            dtype = FLOAT_DTYPES

        if not in_fit and self._fit_dtype.kind == "O":
            # Use object dtype if fitted on object dtypes
            dtype = self._fit_dtype

        if _is_pandas_na(self.missing_values) or is_scalar_nan(self.missing_values):
            force_all_finite = "allow-nan"
        else:
            force_all_finite = True

        try:
            X = self._validate_data(
                X,
                reset=in_fit,
                accept_sparse="csc",
                dtype=dtype,
                force_all_finite=force_all_finite,
                copy=self.copy,
            )
        except ValueError as ve:
            if "could not convert" in str(ve):
                new_ve = ValueError(
                    "Cannot use {} strategy with non-numeric data:\n{}".format(
                        self.strategy, ve
                    )
                )
                raise new_ve from None
            else:
                raise ve

        if in_fit:
            # Use the dtype seen in `fit` for non-`fit` conversion
            self._fit_dtype = X.dtype

        _check_inputs_dtype(X, self.missing_values)
        if X.dtype.kind not in ("b", "i", "u", "f", "O"):  # The change to support boolean arrays is here.
            raise ValueError(
                "SimpleImputer does not support data with dtype "
                "{0}. Please provide either a numeric array (with"
                " a floating point or integer dtype) or "
                "categorical data represented either as an array "
                "with integer dtype or an array of string values "
                "with an object dtype.".format(X.dtype)
            )

        return X
