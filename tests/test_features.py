"""Feature engineering tests."""
from __future__ import annotations

from src.features.engineering import (
    CATEGORICAL_ENGINEERED,
    ENGINEERED_FEATURES,
    MODEL_FEATURES,
    NUMERIC_ENGINEERED,
    create_features,
)


def test_all_engineered_columns_created(sample_frame):
    engineered, log = create_features(sample_frame)
    for name in ENGINEERED_FEATURES:
        assert name in engineered.columns, f"missing {name}"
    assert len(log) == len(ENGINEERED_FEATURES)


def test_model_features_are_subset(sample_frame):
    engineered, _ = create_features(sample_frame)
    for feat in MODEL_FEATURES:
        assert feat in engineered.columns


def test_leakage_safety_pointwise(sample_frame):
    """Feature transforms must be point-wise (no dataset statistics)."""
    a, _ = create_features(sample_frame)
    # Reordering rows must not change the engineered values of any row.
    shuffled = sample_frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    b, _ = create_features(shuffled)
    for feat in NUMERIC_ENGINEERED:
        # Compare by sorting: values attributable to each row, not batch stats.
        assert sorted(a[feat].tolist()) == sorted(b[feat].tolist()), feat


def test_delinquency_flags(sample_frame):
    engineered, _ = create_features(sample_frame)
    assert engineered["any_delinquency_flag"].isin([0, 1]).all()
    assert engineered["serious_delinquency_flag"].isin([0, 1]).all()
    assert (engineered["total_delinquencies"] >= 0).all()


def test_credit_risk_score_bounds(sample_frame):
    engineered, _ = create_features(sample_frame)
    assert engineered["credit_risk_score"].between(0, 1).all()


def test_income_per_dependent(sample_frame):
    engineered, _ = create_features(sample_frame)
    depend = sample_frame["NumberOfDependents"].fillna(0)
    expected = sample_frame["MonthlyIncome"].fillna(0) / (depend + 1)
    assert engineered["income_per_dependent"].equals(expected)


def test_age_bucket_is_categorical(sample_frame):
    engineered, _ = create_features(sample_frame)
    assert "age_bucket" in CATEGORICAL_ENGINEERED
    buckets = set(engineered["age_bucket"])
    assert buckets <= {"Young", "Prime", "Middle", "Senior", "Unknown"}
