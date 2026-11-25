"""Preprocessing and pipeline builders for the readmission risk model.

All preprocessing lives inside a single :class:`sklearn.pipeline.Pipeline` so
that imputation, encoding and scaling are fit on the training split only.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from carereturn import config


class AsStringTransformer(BaseEstimator, TransformerMixin):
    """Coerce every column to string while keeping missing values as NaN.

    The source data mixes categorical columns stored as integers
    (``admission_type_id`` and friends) with real string columns. Casting to
    string makes both share one encoding path. Missing values are restored
    afterwards so that the downstream :class:`~sklearn.impute.SimpleImputer`
    imputes them (the literal ``"None"`` used for lab results is a genuine
    category and is preserved as-is).
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> AsStringTransformer:  # noqa: N803, ARG002
        """No-op fit, kept for scikit-learn API compatibility.

        Args:
            X: Input frame (unused, the transform is stateless).
            y: Optional target (ignored).

        Returns:
            The fitted transformer.

        """
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Cast columns to string and keep originally missing cells as NaN.

        Numeric columns (integer administrative codes possibly holding NaN,
        hence stored as floats) are normalized through a nullable Int64 so
        that ``1`` and ``1.0`` always encode to the same category ``"1"``.

        Args:
            X: DataFrame or array-like holding the categorical columns.

        Returns:
            Object array of string values with NaN preserved for missing cells.

        """
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        missing = frame.isna()
        out = frame.copy()
        for column in out.columns:
            series = out[column]
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                with contextlib.suppress(TypeError, ValueError):
                    out[column] = series.astype("Int64")
        return out.astype(str).mask(missing, np.nan).to_numpy(dtype=object)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return the output feature names (identity mapping).

        Args:
            input_features: Names of the input columns.

        Returns:
            Array of the input column names.

        """
        if input_features is None:
            raise ValueError("input_features is required for AsStringTransformer")
        return np.asarray(input_features, dtype=object)


def build_preprocessor(
    *,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    cat_min_frequency: float = 0.01,
) -> ColumnTransformer:
    """Build the preprocessing column transformer.

    Numeric features are median-imputed then standardized. Categorical
    features are cast to string, constant-imputed with the token ``"missing"``
    and one-hot encoded. Categories seen less than ``cat_min_frequency``
    (proportion of rows) are grouped into a single infrequent bucket, and
    unseen categories at predict time are encoded as all-zeros.

    Args:
        numeric_features: Numeric column names (defaults to
            :data:`carereturn.config.NUMERIC_FEATURES`).
        categorical_features: Categorical column names (defaults to
            :data:`carereturn.config.CATEGORICAL_FEATURES`).
        cat_min_frequency: Minimum category frequency; rarer levels are
            grouped together.

    Returns:
        A fitted-on-demand (not yet fitted) :class:`ColumnTransformer`.

    """
    numeric = list(numeric_features or config.NUMERIC_FEATURES)
    categorical = list(categorical_features or config.CATEGORICAL_FEATURES)

    numeric_pipeline: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline: Pipeline = Pipeline(
        steps=[
            ("to_str", AsStringTransformer()),
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=cat_min_frequency,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def build_pipeline(
    estimator: Any,
    *,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    cat_min_frequency: float = 0.01,
) -> Pipeline:
    """Wrap preprocessing and an estimator into one scikit-learn pipeline.

    Args:
        estimator: Unfitted classifier implementing the scikit-learn API.
        numeric_features: Numeric column names (defaults to config).
        categorical_features: Categorical column names (defaults to config).
        cat_min_frequency: Minimum category frequency for the one-hot encoder.

    Returns:
        A :class:`sklearn.pipeline.Pipeline` with steps
        ``("preprocess", "clf")``, to be fit on the training split only.

    """
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    cat_min_frequency=cat_min_frequency,
                ),
            ),
            ("clf", estimator),
        ]
    )
