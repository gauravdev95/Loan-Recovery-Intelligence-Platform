"""Prepare the raw dataset: validate -> clean -> save interim & processed.

Steps
-----
1. Validate the raw CSV.
2. Clean it (drop unnamed col, impossible ages, duplicates, clamp/bounds).
3. Write the cleaned copy to ``data/interim/cleaned.csv``.
4. Apply feature engineering (see src/features/engineering.py) and write the
   engineered dataset to ``data/processed/engineered.csv`` (leakage-safe:
   imputation is NOT applied here — it happens inside the supervised
   preprocessing pipeline on the training split only).

Usage
-----
    python scripts/prepare_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_PATH,
    ensure_dirs,
)
from src.data.cleaning import clean_dataset  # noqa: E402
from src.data.loading import save_processed  # noqa: E402
from src.data.validation import validate_dataset  # noqa: E402
from src.features.engineering import create_features  # noqa: E402

INTERIM_PATH = INTERIM_DATA_DIR / "cleaned.csv"
PROCESSED_PATH = PROCESSED_DATA_DIR / "engineered.csv"


def main() -> int:
    ensure_dirs()
    if not RAW_DATA_PATH.exists():
        print("Raw dataset missing. Run: python scripts/download_data.py")
        return 1

    print("1/4 Loading raw data...")
    import pandas as pd

    df = pd.read_csv(RAW_DATA_PATH)

    print("2/4 Validating...")
    result = validate_dataset(df)
    print(result.render())
    print()

    print("3/4 Cleaning...")
    cleaned = clean_dataset(df)
    save_processed(cleaned, INTERIM_PATH)
    print(f"  Cleaned rows: {len(df)} -> {len(cleaned)}")
    print(f"  Saved interim     -> {INTERIM_PATH}")

    print("4/4 Engineering features...")
    engineered, feature_log = create_features(cleaned)
    save_processed(engineered, PROCESSED_PATH)
    print(f"  Engineered rows : {len(engineered)}")
    print(f"  Engineered cols : {len(engineered.columns)}")
    print("  Features added  :")
    for name, desc in feature_log.items():
        print(f"    - {name}: {desc}")
    print(f"  Saved processed -> {PROCESSED_PATH}")

    print("\nData preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
