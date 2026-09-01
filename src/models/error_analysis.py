"""Error analysis for the final model.

Examines False Positives / False Negatives, high-confidence mistakes and
low-confidence predictions on a held-out split, and summarises patterns in the
minority class (defaulters) that the model struggles with.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def run_error_analysis(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    top_k: int = 8,
) -> dict[str, Any]:
    """Produce an error-analysis summary from predictions on a split.

    Returns a dict describing the error mix and per-group patterns, suitable
    for JSON serialisation and report rendering.
    """
    y_pred = (y_proba >= threshold).astype(int)
    y_true = np.asarray(y_true)

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    tp_mask = (y_pred == 1) & (y_true == 1)
    tn_mask = (y_pred == 0) & (y_true == 0)

    # Confidence categories on confident (wrong) predictions.
    fp_confident = fp_mask & (y_proba >= 0.8)
    fn_confident = fn_mask & (y_proba <= 0.2)
    low_confidence = np.abs(y_proba - 0.5) < 0.15  # near the decision boundary

    summary: dict[str, Any] = {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "counts": {
            "true_positive": int(tp_mask.sum()),
            "true_negative": int(tn_mask.sum()),
            "false_positive": int(fp_mask.sum()),
            "false_negative": int(fn_mask.sum()),
        },
        "confidence_analysis": {
            "high_confidence_false_positive": int(fp_confident.sum()),
            "high_confidence_false_negative": int(fn_confident.sum()),
            "low_confidence_predictions": int(low_confidence.sum()),
        },
        "error_groups": {},
    }

    # Defaulters missed (FN) tend to be the most actionable — describe them.
    for name, mask in (
        ("missed_defaulters_false_negatives", fn_mask),
        ("false_alarms_false_positives", fp_mask),
        ("correct_defaulters_true_positives", tp_mask),
    ):
        if X is not None and mask.sum() > 0:
            summary["error_groups"][name] = _describe(X.loc[mask], top_k=top_k)

    return summary


def _describe(sub: pd.DataFrame, top_k: int = 8) -> dict[str, Any]:
    """Summarise the mean feature profile of a subgroup."""
    desc: dict[str, Any] = {"size": int(len(sub))}
    numeric = sub.select_dtypes(include=[np.number]).columns
    if len(numeric):
        means = sub[numeric].mean().sort_values(ascending=False)
        desc["top_mean_features"] = {
            str(k): round(float(v), 4) for k, v in means.head(top_k).items()
        }
    return desc


def render_error_report(report: dict[str, Any]) -> str:
    lines = ["ERROR ANALYSIS", "-----------------"]
    counts = report["counts"]
    lines.append(
        f"Threshold: {report['threshold']:.3f} | N={report['n_samples']}"
    )
    lines.append(
        f"TP={counts['true_positive']} TN={counts['true_negative']} "
        f"FP={counts['false_positive']} FN={counts['false_negative']}"
    )
    conf = report["confidence_analysis"]
    lines.append(
        f"High-confidence FP: {conf['high_confidence_false_positive']} | "
        f"High-confidence FN: {conf['high_confidence_false_negative']} | "
        f"Low-confidence: {conf['low_confidence_predictions']}"
    )
    for group, desc in report["error_groups"].items():
        lines.append(f"\n[{group}] (n={desc['size']})")
        for feat, val in desc.get("top_mean_features", {}).items():
            lines.append(f"  {feat}: {val}")
    return "\n".join(lines)


def save_error_report(report: dict[str, Any], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
