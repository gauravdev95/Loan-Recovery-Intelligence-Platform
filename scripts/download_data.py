"""Automatic acquisition of the real "Give Me Some Credit" dataset.

Behaviour
---------
1. If ``data/raw/cs-training.csv`` already exists, it is validated and the
   script exits without re-downloading (never clobbers existing data).
2. If it does not exist, the script tries the official Kaggle API.
3. It detects the common Kaggle authentication failure and prints a clear
   setup message instead of fabricating data.
4. The script NEVER writes fake data. If the real dataset cannot be obtained,
   it fails loudly.

Expected columns (Give Me Some Credit, cs-training.csv): 11 (10 features +
binary target ``SeriousDlqin2yrs``).

Usage
-----
    python scripts/download_data.py
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

# Allow running this script from the repo root or from scripts/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR, RAW_DATA_PATH, ensure_dirs  # noqa: E402
from src.data.validation import validate_dataset  # noqa: E402

# Mirrors of the original FICO / Kaggle competition data on Kaggle. The first
# is a well-known complete mirror of the 2011 "Give Me Some Credit" data.
KAGGLE_DATASETS = [
    "brycecf/give-me-some-credit-dataset",
    "lihxlhx/give-me-some-credit",
]
CSV_NAME_IN_ZIP = "cs-training.csv"

EXPECTED_COLS = {
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
}


def _kaggle_available() -> bool:
    try:
        import kaggle  # noqa: F401

        return True
    except ImportError:
        return False


def _has_credentials() -> bool:
    from pathlib import Path as _P

    cred_file = _P.home() / ".kaggle" / "kaggle.json"
    return cred_file.exists() or bool(
        __import__("os").environ.get("KAGGLE_USERNAME")
    )


def download_and_place() -> Path:
    """Download the dataset via Kaggle and place it at data/raw/cs-training.csv."""
    if not _kaggle_available():
        raise RuntimeError(
            "The 'kaggle' package is not installed. Run: pip install kaggle"
        )
    if not _has_credentials():
        raise RuntimeError(
            "Kaggle credentials not found. Create ~/.kaggle/kaggle.json with:\n"
            '  {"username":"YOUR_USERNAME","key":"YOUR_KEY"}\n'
            "or set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    tmp = RAW_DATA_DIR / ".download_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    last_err = None
    for dataset in KAGGLE_DATASETS:
        try:
            print(f"Downloading Kaggle dataset: {dataset}")
            api.dataset_download_files(dataset, path=str(tmp), unzip=False)
            zips = list(tmp.glob("*.zip"))
            if not zips:
                raise RuntimeError("No .zip produced by downloader.")
            with zipfile.ZipFile(zips[0]) as zf:
                names = zf.namelist()
                if CSV_NAME_IN_ZIP not in names:
                    raise RuntimeError(f"{CSV_NAME_IN_ZIP} not found in zip.")
                zf.extract(CSV_NAME_IN_ZIP, path=str(tmp))
            shutil.move(str(tmp / CSV_NAME_IN_ZIP), str(RAW_DATA_PATH))
            print(f"Placed dataset at {RAW_DATA_PATH}")
            return RAW_DATA_PATH
        except Exception as exc:  # noqa: BLE001 - try next mirror
            last_err = exc
            print(f"Mirror {dataset} failed: {exc}")
            for f in tmp.glob("*"):
                f.unlink()
            continue
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    raise RuntimeError(f"Could not download from any mirror. Last error: {last_err}")


def main() -> int:
    ensure_dirs()

    if RAW_DATA_PATH.exists() and RAW_DATA_PATH.stat().st_size > 0:
        print(f"Dataset already present at {RAW_DATA_PATH}. Validating...")
        df = pd.read_csv(RAW_DATA_PATH)
        result = validate_dataset(df)
        print(result.render())
        if result.passed:
            print("Existing dataset is valid. Nothing to do.")
            return 0
        print("Existing dataset FAILED validation; will attempt re-download.")
        # Only overwrite after a re-download succeeds; do not delete here.

    try:
        path = download_and_place()
    except Exception as exc:  # noqa: BLE001
        print(f"\n[ERROR] Could not acquire the real dataset: {exc}")
        print(
            "\nI will NOT generate or substitute fake data. "
            "Provide Kaggle credentials and re-run this script."
        )
        return 1

    df = pd.read_csv(path)
    result = validate_dataset(df)
    print(result.render())
    if not result.passed:
        print("\n[WARNING] Downloaded dataset failed validation; inspect before use.")
        return 2

    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        print(f"\n[WARNING] Missing expected columns: {sorted(missing)}")
        return 2

    print("\nDataset downloaded and validated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
