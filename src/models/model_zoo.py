"""Model factories for the experiment zoo.

All models are CPU-friendly by construction — moderate tree counts, thread
counts bounded, deterministic seeds. Class imbalance is handled at the model
level (class weighting / scale_pos_weight) rather than by accuracy-driven
selection.
"""
from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_SEED


def _n_jobs() -> int:
    import os

    return int(os.environ.get("LRP_N_JOBS", "-1")) or -1


def logistic_regression(**overrides: Any) -> LogisticRegression:
    params = dict(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_SEED,
        n_jobs=_n_jobs(),
    )
    params.update(overrides)
    return LogisticRegression(**params)


def random_forest(**overrides: Any) -> RandomForestClassifier:
    params = dict(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=_n_jobs(),
    )
    params.update(overrides)
    return RandomForestClassifier(**params)


def xgboost(**overrides: Any) -> XGBClassifier:
    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=10.0,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=_n_jobs(),
        random_state=RANDOM_SEED,
    )
    params.update(overrides)
    return XGBClassifier(**params)


def lightgbm(**overrides: Any) -> LGBMClassifier:
    params = dict(
        n_estimators=300,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=20,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=10.0,
        n_jobs=_n_jobs(),
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    params.update(overrides)
    return LGBMClassifier(**params)


FACTORIES: dict[str, Any] = {
    "Logistic Regression": logistic_regression,
    "Random Forest": random_forest,
    "XGBoost": xgboost,
    "LightGBM": lightgbm,
}
