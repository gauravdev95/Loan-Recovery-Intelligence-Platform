"""Central project configuration: paths, dataset schema and constants.

Single source of truth used by scripts, the training pipeline, the backend
and the tests so that everything stays consistent.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
EXPLANATIONS_DIR = ARTIFACTS_DIR / "explanations"

RAW_DATA_PATH = RAW_DATA_DIR / "cs-training.csv"

# Serialized artifacts
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessing.joblib"
FEATURE_METADATA_PATH = MODELS_DIR / "feature_metadata.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Dataset schema (Give Me Some Credit)
# ---------------------------------------------------------------------------
TARGET = "SeriousDlqin2yrs"

# Numeric columns first, then categorical/flag columns we engineer later.
RAW_FEATURES = [
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
]

# Drop the unnamed row-id column that appears as the first CSV column.
CSV_USE_COLS = [TARGET] + RAW_FEATURES

# Feature engineering version — bump when transform logic changes so that
# cached artifacts / model metadata stay consistent.
FEATURES_VERSION = "v1"

# ---------------------------------------------------------------------------
# Experiment / split
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VALID_FRAC = 0.15
TEST_FRAC = 0.15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create all data / artifact directories if missing."""
    for d in (
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
        EXPLANATIONS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def dataset_present() -> bool:
    return RAW_DATA_PATH.exists() and RAW_DATA_PATH.stat().st_size > 0


def env_flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
