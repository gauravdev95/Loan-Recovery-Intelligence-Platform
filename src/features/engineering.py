"""Feature engineering.

Every derived feature is a **point-wise transformation** of the raw credit
columns — no dataset statistics (medians/quantiles/min-max computed across
rows) are used, because using them would leak information from the test set
into the training transforms. Fixed, documented thresholds only.

The input features are all known at credit-origination/review time; nothing
depends on the target ``SeriousDlqin2yrs`` or on post-decision outcomes, so
the engineering step is leakage-safe by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RAW_FEATURES, TARGET

# ---------------------------------------------------------------------------
# Engineered feature definitions (name -> one-line description)
# ---------------------------------------------------------------------------
ENGINEERED_FEATURES: dict[str, str] = {
    "total_delinquencies": "Sum of all 30-59, 60-89 and 90+ day past-due counts",
    "worst_delinquency_tier": "Ordinal worst delinquency tier: 0=none,1=30-59,2=60-89,3=90+",
    "any_delinquency_flag": "1 if any past-due event was reported",
    "serious_delinquency_flag": "1 if any 90+ days past due (serious delinquency)",
    "multiple_delinquency_flag": "1 if total past-due counts >= 2",
    "delinquency_weighted_index": "Weighted severity index (weight 1/2/3 by tier)",
    "income_per_dependent": "MonthlyIncome / (NumberOfDependents + 1)",
    "has_income_flag": "1 if MonthlyIncome > 0 (else income likely missing)",
    "age_bucket": "Fixed age band (Young/Prime/Middle/Senior)",
    "utilization_stress_flag": "1 if revolving utilization >= 0.9 (near/over limit)",
    "financial_stress_indicator": "1 if any delinquency and utilization >= 0.8",
    "credit_risk_score": "Composite [,1] interpretable score from delinquency + utilization + debt",
}

# Numeric vs categorical engineered columns.
CATEGORICAL_ENGINEERED = ["age_bucket"]
NUMERIC_ENGINEERED = [
    k for k in ENGINEERED_FEATURES if k not in CATEGORICAL_ENGINEERED
]

# Final model feature set (raw numeric + engineered).
MODEL_FEATURES = RAW_FEATURES + NUMERIC_ENGINEERED + CATEGORICAL_ENGINEERED

# Fixed thresholds (documented, not data-derived).
HIGH_UTILIZATION = 0.9
STRESS_UTILIZATION = 0.8


def _worst_tier(d30: float, d60: float, d90: float) -> int:
    if d90 > 0:
        return 3
    if d60 > 0:
        return 2
    if d30 > 0:
        return 1
    return 0


def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Add engineered features to a cleaned dataframe.

    Returns a copy with ENGINEERED_FEATURES appended, plus the feature log.
    The dataframe is expected to contain the RAW_FEATURES columns.
    """
    out = df.copy()

    d30 = out["NumberOfTime30-59DaysPastDueNotWorse"].fillna(0)
    d60 = out["NumberOfTime60-89DaysPastDueNotWorse"].fillna(0)
    d90 = out["NumberOfTimes90DaysLate"].fillna(0)
    util = out["RevolvingUtilizationOfUnsecuredLines"].fillna(0)
    debt = out["DebtRatio"].fillna(0)
    income = out["MonthlyIncome"].fillna(0)
    depend = out["NumberOfDependents"].fillna(0)

    out["total_delinquencies"] = (d30 + d60 + d90).astype(int)
    out["worst_delinquency_tier"] = [
        _worst_tier(a, b, c) for a, b, c in zip(d30, d60, d90)
    ]
    out["any_delinquency_flag"] = (out["total_delinquencies"] > 0).astype(int)
    out["serious_delinquency_flag"] = (d90 > 0).astype(int)
    out["multiple_delinquency_flag"] = (out["total_delinquencies"] >= 2).astype(int)
    out["delinquency_weighted_index"] = (d30 * 1 + d60 * 2 + d90 * 3).astype(int)

    out["income_per_dependent"] = income / (depend + 1)

    out["has_income_flag"] = (income > 0).astype(int)

    # Fixed age bands (borrower age, not a dataset statistic).
    age = out["age"]
    buckets = pd.cut(
        age,
        bins=[0, 25, 35, 50, 124],
        labels=["Young", "Prime", "Middle", "Senior"],
        right=True,
    ).astype(str)
    out["age_bucket"] = buckets.replace("nan", "Unknown").astype("category")

    out["utilization_stress_flag"] = (util >= HIGH_UTILIZATION).astype(int)
    out["financial_stress_indicator"] = (
        (out["any_delinquency_flag"] == 1) & (util >= STRESS_UTILIZATION)
    ).astype(int)

    # Composite interpretable risk score in [0,1]. Uses fixed monotonic
    # transforms (1 - exp(-x)) — no cross-row statistics, hence leakage-safe.
    delinq_norm = 1.0 - np.exp(-(out["delinquency_weighted_index"].to_numpy()) / 5.0)
    util_norm = np.clip(util.to_numpy(), 0.0, 2.0) / 2.0
    debt_norm = np.clip(debt.to_numpy(), 0.0, 4.0) / 4.0
    score = 0.4 * delinq_norm + 0.3 * util_norm + 0.3 * debt_norm
    out["credit_risk_score"] = np.clip(score, 0.0, 1.0)

    return out, dict(ENGINEERED_FEATURES)
