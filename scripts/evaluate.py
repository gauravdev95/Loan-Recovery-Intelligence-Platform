"""Evaluate the serialized final model on the held-out test split.

Loads ``models/best_model.joblib`` + metadata, runs the honest test-set
evaluation, produces a SHAP local explanation for a sample borrower and prints
a concise results report. Requires artifacts from ``scripts/train.py``.

Usage
-----
    python scripts/evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    EXPLANATIONS_DIR,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET,
    ensure_dirs,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.features.engineering import MODEL_FEATURES  # noqa: E402
from src.models import artifacts  # noqa: E402
from src.models.pipeline import load_split_data  # noqa: E402


def main() -> int:
    ensure_dirs()
    model_path = Path(__file__).resolve().parent.parent / "models/best_model.joblib"
    if not model_path.exists():
        print("No trained model found. Run: python scripts/train.py")
        return 1

    model = artifacts.load_model()
    meta = artifacts.load_metadata()

    print("Splitting data (stratified 70/15/15)...")
    X_tr, X_val, X_test, y_tr, y_val, y_test = load_split_data()

    print("Running final test-set evaluation...")
    test_proba = model.predict_proba(X_test)[:, 1]
    threshold = float(meta.get("validation_threshold", 0.5))
    metrics = compute_metrics(y_test, test_proba, threshold=threshold)
    print(
        f"  F1={metrics['f1']:.4f} ROC-AUC={metrics['roc_auc']:.4f} "
        f"PR-AUC={metrics['pr_auc']:.4f} Recall={metrics['recall']:.4f}"
    )

    report = {
        "model_name": meta.get("model_name"),
        "model_version": meta.get("model_version"),
        "test_metrics": metrics,
        "threshold": threshold,
        "risk_thresholds": meta.get("threshold_configuration", {}).get(
            "risk_level_thresholds"
        ),
    }
    (REPORTS_DIR / "final_test_evaluation.json").write_text(
        json.dumps(report, indent=2)
    )
    print("Saved -> artifacts/reports/final_test_evaluation.json")

    # SHAP local explanation for a sample borrower.
    try:
        from src.explainability.shap_explainer import ShapExplainer
        from src.models.pipeline import _transform_df

        # SHAP operates on the *preprocessed* numeric matrix (the tree was fit
        # on transformed features, incl. the one-hot encoded age_bucket), so we
        # push the background + the sample row through the pipeline first.
        Xb = X_tr.sample(min(300, len(X_tr)), random_state=RANDOM_SEED)
        Xb_t = _transform_df(model, Xb)
        model_est = model.named_steps["model"]
        explainer = ShapExplainer.for_tree_model(model_est, Xb_t)
        row_t = _transform_df(model, X_test.head(1))
        local = explainer.local_explanation(row_t)
        local["borrower_index"] = int(X_test.index[0])
        (EXPLANATIONS_DIR / "evaluate_sample_explanation.json").write_text(
            json.dumps(local, indent=2, default=str)
        )
        print("\nSample borrower SHAP explanation:")
        print(f"  expected (log-odds): {local['expected_value']:.3f}")
        for f in local["top_factors"][:5]:
            print(f"  - {f['feature']} = {f['value']} -> {f['direction']}")
    except Exception as exc:  # noqa: BLE001 - SHAP optional here
        print(f"\nSHAP sample skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
