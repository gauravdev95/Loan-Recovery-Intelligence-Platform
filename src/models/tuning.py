"""CPU-friendly hyperparameter tuning for tree-ensemble candidates.

Uses sklearn ``RandomizedSearchCV`` with a small internal CV on the TRAINING
split only (the test split is never touched). The search is deliberately small
(<= ~24 iterations) so it runs reasonably on a laptop without a GPU.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.config import RANDOM_SEED

_RNG_PARAM = {"random_state": RANDOM_SEED}


def _search(X, y, pipeline: Pipeline, param_grid: dict[str, Any], n_iter: int, cv: int = 2):
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        n_jobs=_tune_jobs(),
        random_state=RANDOM_SEED,
        refit=True,
        verbose=0,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def _tune_jobs() -> int:
    """Bound the search to avoid nested-parallel CPU oversubscription."""
    import os

    return int(os.environ.get("LRP_TUNE_JOBS", "4"))


def _spw(y: np.ndarray) -> float:
    """Class-ratio scale_pos_weight for the positive (default) class."""
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return float(neg / pos) if pos else 1.0


def build_xgb_grid(spw: float) -> dict[str, Any]:
    return {
        "model__n_estimators": [200, 400],
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.9],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_alpha": [0.0, 0.1, 1.0],
        "model__reg_lambda": [0.1, 1.0, 3.0],
        "model__scale_pos_weight": [spw],
    }


def build_lgbm_grid(spw: float) -> dict[str, Any]:
    return {
        "model__n_estimators": [200, 400],
        "model__num_leaves": [31, 63, 127],
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.9],
        "model__min_child_samples": [10, 20, 40],
        "model__reg_alpha": [0.0, 0.1, 1.0],
        "model__reg_lambda": [0.1, 1.0, 3.0],
        "model__scale_pos_weight": [spw],
    }


def tune_candidates(
    X_train,
    y_train,
    preprocessor,
    model_builders: dict[str, Any],
    n_iter: int = 16,
    cv: int = 2,
) -> dict[str, RandomizedSearchCV]:
    """Tune each named candidate on the training split.

    Returns a dict {candidate_name: fitted RandomizedSearchCV}.
    """
    spw = _spw(np.asarray(y_train))
    grids = {
        "XGBoost": build_xgb_grid(spw),
        "LightGBM": build_lgbm_grid(spw),
    }
    results: dict[str, RandomizedSearchCV] = {}
    for name in grids:
        if name not in model_builders:
            continue
        pipe = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model_builders[name]())]
        )
        print(f"  [tuning] {name} ({n_iter} iters, cv={cv})...", flush=True)
        results[name] = _search(
            X_train, y_train, pipe, grids[name], n_iter=n_iter, cv=cv
        )
        best = results[name].best_params_
        score = results[name].best_score_
        print(f"    -> best CV ROC-AUC {score:.4f} | {_compact(best)}")
    return results


def _compact(d: dict[str, Any]) -> str:
    return ", ".join(f"{k.replace('model__','')}={v}" for k, v in d.items())
