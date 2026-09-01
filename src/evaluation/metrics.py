"""Classification metrics helpers.

Returns a compact, JSON-serialisable dict of the business-relevant metrics
used throughout the project (accuracy, precision, recall, F1, ROC-AUC,
PR-AUC and a confusion-matrix breakdown).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the standard metric bundle for binary classification.

    Parameters
    ----------
    y_true : 1-D array of true 0/1 labels.
    y_proba : 1-D array of predicted probabilities for class 1.
    threshold : decision threshold applied to y_proba for hard labels.

    Returns
    -------
    dict with accuracy, precision, recall, f1, roc_auc, pr_auc, threshold,
    confusion matrix and derived rates.
    """
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "true_positive_rate": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "n_samples": int(len(y_true)),
    }


def best_threshold_by_f1(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_grid: int = 200,
) -> float:
    """Find the probability threshold that maximises F1 on a target set.

    Used on the VALIDATION split to pick a decision threshold for the risk
    engine — never on the test set.
    """
    thresholds = np.linspace(0.01, 0.99, n_grid)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_t, best_f1 = float(t), float(f)
    return best_t
