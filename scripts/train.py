"""Train the full ML pipeline and serialize all artifacts.

Usage
-----
    python scripts/train.py
    python scripts/train.py --no-tune   # skip hyperparameter tuning (faster)

Writes:
    models/best_model.joblib, models/preprocessing.joblib,
    models/feature_metadata.json, models/model_metadata.json,
    artifacts/metrics/*.json, artifacts/reports/experiment_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import METRICS_DIR, REPORTS_DIR, ensure_dirs  # noqa: E402
from src.models.pipeline import train_full_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the loan-recovery ML pipeline.")
    parser.add_argument("--no-tune", action="store_true", help="Skip tuning.")
    args = parser.parse_args()

    ensure_dirs()

    if not (Path(Path(__file__).resolve().parent.parent) / "data/processed/engineered.csv").exists():
        print("Processed data missing. Run: python scripts/prepare_data.py")
        return 1

    summary = train_full_pipeline(verbose=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "experiment_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nExperiment summary saved -> {REPORTS_DIR / 'experiment_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
