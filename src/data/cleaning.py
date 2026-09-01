"""Data cleaning.

Applies documented, deterministic fixes to the raw dataset:

* Drops the unnamed row-id column if present.
* Removes rows with an impossible age (< MIN_PLAUSIBLE_AGE) — this dataset
  famously contains an age = 0 record.
* Clamps non-negative integer count columns at sensible bounds.
* Removes rows with infinite values.
* Drops exact duplicates.

Missing values (MonthlyIncome, NumberOfDependents) are deliberately NOT
imputed here — imputation is part of the supervised preprocessing pipeline so
it is learned only on the training split (avoiding leakage).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import RAW_FEATURES, TARGET
from src.data.validation import MIN_PLAUSIBLE_AGE

# Maximum sane value for count columns (number of delinquencies, lines, etc.)
MAX_COUNT = 100
# Reasonable ceiling for utilization (ratios far above 1 indicate extreme risk
# but are still physically possible; cap at a defensible bound).
MAX_UTILIZATION = 10.0


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning rules to the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe (index reset). Missing values are preserved.
    """
    out = df.copy()

    # 1. Drop unnamed row-id column (first column with empty header).
    unnamed = [c for c in out.columns if str(c).strip() in ("", "Unnamed: 0")]
    if unnamed:
        out = out.drop(columns=unnamed)

    # 2. Drop impossible ages.
    if "age" in out.columns:
        out = out[~(out["age"] < MIN_PLAUSIBLE_AGE)]

    # 3. Clamp non-negative count columns.
    count_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]
    for col in count_cols:
        if col in out.columns:
            out[col] = out[col].clip(lower=0, upper=MAX_COUNT)

    # 4. Clamp utilization to a defensible ceiling.
    if "RevolvingUtilizationOfUnsecuredLines" in out.columns:
        out["RevolvingUtilizationOfUnsecuredLines"] = out[
            "RevolvingUtilizationOfUnsecuredLines"
        ].clip(lower=0.0, upper=MAX_UTILIZATION)

    # 5. Remove rows with true infinite values only (NaN is KEPT for
    #    downstream imputation — np.isinf not np.isfinite, since isfinite
    #    treats NaN as non-finite and would silently drop missing records).
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        arr = out[num_cols].to_numpy(dtype=np.float64)
        out = out[~np.isinf(arr).any(axis=1)]

    # 6. Drop negative values in non-negative fields (0-age handled above).
    #    NaN must be KEPT (missing values, imputed downstream in the
    #    supervised pipeline) — NaN < 0 is False but NaN >= 0 is also False,
    #    so compare while explicitly preserving NaN.
    for col in count_cols:
        if col in out.columns:
            out = out[(out[col] >= 0) | out[col].isna()]
    for col in ("DebtRatio", "MonthlyIncome", "RevolvingUtilizationOfUnsecuredLines"):
        if col in out.columns:
            out = out[(out[col] >= 0) | out[col].isna()]

    # 7. Drop exact duplicates.
    out = out.drop_duplicates().reset_index(drop=True)

    # Coerce target / count columns to the tightest sensible dtype. Columns
    # that still hold NaN (e.g. NumberOfDependents) use pandas nullable dtypes
    # so no rows are lost before supervised imputation.
    if TARGET in out.columns:
        out[TARGET] = out[TARGET].astype(int)
    for col in count_cols:
        if col in out.columns:
            out[col] = (
                out[col].astype("Float64")
                if out[col].isna().any()
                else out[col].astype(int)
            )

    return out
