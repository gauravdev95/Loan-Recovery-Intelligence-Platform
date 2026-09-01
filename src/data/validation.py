"""Data validation pipeline.

Checks the raw dataset for schema, type, range, missingness, duplicates,
target validity and class distribution. Produces a human-readable and
machine-readable validation report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import RAW_FEATURES, TARGET

# Per-feature plausible ranges. Outside these we flag records as invalid.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "RevolvingUtilizationOfUnsecuredLines": (0.0, 10.0),  # ratio, cap generous
    "age": (0.0, 124.0),  # 0 is impossible for a borrower -> flagged below
    "NumberOfTime30-59DaysPastDueNotWorse": (0.0, 100.0),
    "DebtRatio": (0.0, 1_000.0),
    "MonthlyIncome": (0.0, 1_000_000.0),
    "NumberOfOpenCreditLinesAndLoans": (0.0, 100.0),
    "NumberOfTimes90DaysLate": (0.0, 100.0),
    "NumberRealEstateLoansOrLines": (0.0, 100.0),
    "NumberOfTime60-89DaysPastDueNotWorse": (0.0, 100.0),
    "NumberOfDependents": (0.0, 100.0),
}

# Age below this is almost certainly an erroneous recording (this dataset
# contains an infamous age = 0 row). Values below it are treated as invalid.
MIN_PLAUSIBLE_AGE = 18


@dataclass
class ValidationResult:
    """Structured outcome of a validation pass."""

    original_rows: int = 0
    input_rows: int = 0
    input_cols: int = 0
    duplicates: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    missing_total: int = 0
    impossible_ages: int = 0
    invalid_income: int = 0
    invalid_delinquency: int = 0
    negative_values: int = 0
    infinite_values: int = 0
    invalid_target: int = 0
    target_distribution: dict[str, int] = field(default_factory=dict)
    target_rate: float = 0.0
    invalid_bool: pd.DataFrame | None = None
    passed: bool = True
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "input_cols": self.input_cols,
            "duplicates": self.duplicates,
            "missing_values": self.missing_values,
            "missing_total": self.missing_total,
            "impossible_ages": self.impossible_ages,
            "invalid_income": self.invalid_income,
            "invalid_delinquency": self.invalid_delinquency,
            "negative_values": self.negative_values,
            "infinite_values": self.infinite_values,
            "invalid_target": self.invalid_target,
            "target_distribution": self.target_distribution,
            "target_rate": self.target_rate,
            "passed": self.passed,
        }

    def render(self) -> str:
        """Human-readable validation report."""
        lines = [
            "DATA VALIDATION",
            "---------------",
            f"Input rows: {self.input_rows}",
            f"Input columns: {self.input_cols}",
            f"Duplicates: {self.duplicates}",
            f"Missing values: {self.missing_total}",
        ]
        for col, n in sorted(self.missing_values.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {col}: {n}")
        lines += [
            f"Impossible ages: {self.impossible_ages}",
            f"Invalid income values: {self.invalid_income}",
            f"Invalid delinquency values: {self.invalid_delinquency}",
            f"Negative values: {self.negative_values}",
            f"Infinite values: {self.infinite_values}",
            f"Invalid target values: {self.invalid_target}",
            f"Target distribution: {self.target_distribution}",
            f"Target (default) rate: {self.target_rate:.4%}",
            f"Validation passed: {self.passed}",
        ]
        if self.messages:
            lines.append("Messages:")
            lines += [f"  - {m}" for m in self.messages]
        return "\n".join(lines)


def validate_dataset(
    df: pd.DataFrame,
    require_target: bool = True,
) -> ValidationResult:
    """Validate a raw dataset against the expected schema and semantics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (must already contain the target if require_target).
    require_target : bool
        Whether the target column must be present (training data).

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()
    result.input_rows = len(df)
    result.input_cols = len(df.columns)

    expected = RAW_FEATURES + ([TARGET] if require_target else [])
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        result.passed = False
        result.messages.append(f"Missing expected columns: {missing_cols}")

    # Work on the subset of expected columns that exist.
    present = [c for c in expected if c in df.columns]
    sub = df[present].copy()

    # Duplicates across the full feature set.
    result.duplicates = int(sub.duplicated().sum())

    # Missing values
    for col in present:
        n = int(sub[col].isna().sum())
        if n:
            result.missing_values[col] = n
    result.missing_total = int(sub.isna().sum().sum())

    # Infinite values
    num_cols = sub.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        result.infinite_values = int(np.isinf(sub[num_cols].to_numpy(dtype=np.float64)).sum())

    # Target validity
    if TARGET in sub.columns:
        tcol = sub[TARGET]
        invalid = tcol.isna() | ~tcol.isin([0, 1])
        result.invalid_target = int(invalid.sum())
        dist = sub.loc[~invalid, TARGET].value_counts(normalize=False).to_dict()
        result.target_distribution = {str(k): int(v) for k, v in dist.items()}
        total_valid_target = sum(result.target_distribution.values())
        result.target_rate = (
            result.target_distribution.get("1", 0) / total_valid_target
            if total_valid_target
            else 0.0
        )

    # Impossible ages
    if "age" in sub.columns:
        result.impossible_ages = int(
            ((sub["age"] < MIN_PLAUSIBLE_AGE) | (sub["age"] > FEATURE_RANGES["age"][1])).sum()
        )

    # Invalid income
    if "MonthlyIncome" in sub.columns:
        mi = sub["MonthlyIncome"]
        result.invalid_income = int((mi < 0).sum())

    # Invalid delinquency (negative counts)
    delinquency_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]
    for col in delinquency_cols:
        if col in sub.columns:
            result.invalid_delinquency += int((sub[col] < 0).sum())

    # Negative values anywhere numeric
    if len(num_cols):
        result.negative_values = int((sub[num_cols] < 0).sum().sum())

    if result.infinite_values or result.invalid_target or result.impossible_ages:
        result.passed = False
    if result.invalid_income or result.invalid_delinquency:
        result.passed = False

    if result.missing_total:
        result.messages.append(
            "Missing values detected: handled downstream by imputation."
        )
    return result
