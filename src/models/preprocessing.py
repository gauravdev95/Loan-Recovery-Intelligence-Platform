"""Preprocessing transformers.

Built to be learned ONLY on the training split (inside a scikit-learn
pipeline), so that no test information leaks into imputation/encoding/scaling.

* Numeric features : median imputation + optional standard scaling.
* Categorical features (age_bucket) : unknown-aware ordinal encoding (the
  model sees a single numeric code; trees handle it, LR sees it scaled).

``MODEL_FEATURES`` is the exact ordered feature list the backend must send,
so it lives here as part of the preprocessing contract.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.config import RAW_FEATURES
from src.features.engineering import MODEL_FEATURES, NUMERIC_ENGINEERED

CATEGORICAL_FEATURES = ["age_bucket"]

# Ordered feature list consumed by the model and required from the API.
FEATURE_ORDER = MODEL_FEATURES


def _numeric_cols() -> list[str]:
    return RAW_FEATURES + NUMERIC_ENGINEERED


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric + categorical features.

    Parameters
    ----------
    scale_numeric : whether to standard-scale numeric features. True for
        linear models (LR), False for tree ensembles (invariant to scaling).
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric = Pipeline(steps=numeric_steps)
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, _numeric_cols()),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names_after_transform() -> list[str]:
    """Return the transformed feature names (num + encoded cat) in order."""
    return _numeric_cols() + CATEGORICAL_FEATURES


def preprocessor_feature_count() -> int:
    """Number of columns the preprocessor emits (for validation)."""
    return len(feature_names_after_transform())
