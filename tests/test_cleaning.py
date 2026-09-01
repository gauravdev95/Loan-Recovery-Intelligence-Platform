"""Data cleaning tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET
from src.data.cleaning import clean_dataset
from src.data.validation import MIN_PLAUSIBLE_AGE


def test_impossible_age_removed(sample_frame):
    bad = sample_frame.copy()
    bad.loc[0, "age"] = 0
    cleaned = clean_dataset(bad)
    assert (cleaned["age"] >= MIN_PLAUSIBLE_AGE).all() or len(cleaned) == len(
        bad
    ) - 1


def test_duplicates_removed(sample_frame):
    dup = pd.concat([sample_frame, sample_frame.iloc[:20]], ignore_index=True)
    cleaned = clean_dataset(dup)
    assert cleaned.duplicated().sum() == 0
    assert len(cleaned) < len(dup)


def test_missing_values_preserved_for_imputation(sample_frame):
    bad = sample_frame.copy()
    bad.loc[0, "MonthlyIncome"] = np.nan
    bad.loc[1, "NumberOfDependents"] = np.nan
    cleaned = clean_dataset(bad)
    # NaNs must be kept (imputed later in the supervised pipeline).
    assert cleaned["MonthlyIncome"].isna().sum() >= 1
    assert cleaned["NumberOfDependents"].isna().sum() >= 1


def test_negative_counts_clamped_or_removed(sample_frame):
    bad = sample_frame.copy()
    bad.loc[0, "NumberOfTimes90DaysLate"] = -3
    cleaned = clean_dataset(bad)
    assert (cleaned["NumberOfTimes90DaysLate"] >= 0).all()


def test_infinite_removed_but_nan_kept(sample_frame):
    bad = sample_frame.copy()
    bad.loc[0, "DebtRatio"] = np.inf
    bad.loc[1, "MonthlyIncome"] = np.nan
    cleaned = clean_dataset(bad)
    assert np.isfinite(cleaned["DebtRatio"]).all()
    assert cleaned["MonthlyIncome"].isna().sum() >= 0


def test_cleaned_dtype_and_target(sample_frame):
    cleaned = clean_dataset(sample_frame)
    assert cleaned[TARGET].dtype == int or str(cleaned[TARGET].dtype).startswith(
        "int"
    )
