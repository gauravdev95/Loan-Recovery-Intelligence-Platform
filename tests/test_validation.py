"""Data validation tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import RAW_FEATURES, TARGET
from src.data.validation import validate_dataset


def test_schema_columns_present(sample_frame):
    result = validate_dataset(sample_frame)
    for col in RAW_FEATURES:
        assert col in sample_frame.columns
    assert TARGET in sample_frame.columns
    assert result.input_cols == len(RAW_FEATURES) + 1


def test_duplicate_detection(sample_frame):
    dup = pd.concat([sample_frame, sample_frame.iloc[:10]], ignore_index=True)
    result = validate_dataset(dup)
    assert result.duplicates >= 10


def test_impossible_age_flagged(sample_frame):
    bad = sample_frame.copy()
    bad.loc[0, "age"] = 0
    result = validate_dataset(bad)
    assert result.impossible_ages >= 1


def test_negative_income_flagged(sample_frame):
    bad = sample_frame.copy()
    bad.loc[0, "MonthlyIncome"] = -500
    result = validate_dataset(bad)
    assert result.invalid_income >= 1


def test_missing_value_detection(sample_frame):
    bad = sample_frame.copy()
    bad.loc[0, "MonthlyIncome"] = np.nan
    result = validate_dataset(bad)
    assert result.missing_values.get("MonthlyIncome", 0) >= 1


def test_infinite_detection(sample_frame):
    bad = sample_frame.copy()
    bad.loc[0, "DebtRatio"] = np.inf
    result = validate_dataset(bad)
    assert result.infinite_values >= 1


def test_target_range(sample_frame):
    result = validate_dataset(sample_frame)
    assert result.invalid_target == 0
    assert result.target_distribution.get("1", 0) > 0
    assert 0.0 < result.target_rate < 1.0


def test_render_report(sample_frame):
    result = validate_dataset(sample_frame)
    text = result.render()
    assert "DATA VALIDATION" in text
    assert "Input rows" in text


def test_missing_column_detected():
    frame = pd.DataFrame({TARGET: [0, 1, 0]})
    result = validate_dataset(frame)
    assert result.passed is False
    assert any("Missing expected columns" in m for m in result.messages)
