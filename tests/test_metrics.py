"""Metric utility tests."""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import best_threshold_by_f1, compute_metrics


def test_compute_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, p, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_metric_keys_present():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.3, 0.7, 0.6, 0.2, 0.9])
    m = compute_metrics(y, p)
    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "threshold",
        "confusion_matrix",
        "n_samples",
    ):
        assert key in m
    assert m["n_samples"] == len(y)


def test_best_threshold_by_f1_returns_valid():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.8 + 0.1 * rng.normal(size=200), 0, 1)
    t = best_threshold_by_f1(y, p)
    assert 0.01 <= t <= 0.99
