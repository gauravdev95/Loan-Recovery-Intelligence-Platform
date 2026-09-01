"""Root-level pytest fixtures.

Ensures the repo root (containing the ``src`` package) is importable and
constructs a small synthetic-but-realistic fixture frame for fast unit tests
that do not require the full 150k-row dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RAW_FEATURES, TARGET  # noqa: E402


@pytest.fixture(scope="session")
def sample_frame() -> pd.DataFrame:
    """A small, realistic Give-Me-Some-Credit-like DataFrame (no target leak)."""
    rng = np.random.default_rng(7)
    n = 600
    df = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1.5, n),
            "age": rng.integers(21, 85, n),
            "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 8, n),
            "DebtRatio": rng.uniform(0, 4, n),
            "MonthlyIncome": rng.uniform(100, 25000, n),
            "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n),
            "NumberOfTimes90DaysLate": rng.integers(0, 8, n),
            "NumberRealEstateLoansOrLines": rng.integers(0, 10, n),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 6, n),
            "NumberOfDependents": rng.integers(0, 6, n).astype("float"),
        }
    )
    # Target driven by interpretable rule so tests are deterministic.
    risk = (
        0.5 * df["RevolvingUtilizationOfUnsecuredLines"]
        + 0.3 * df["NumberOfTimes90DaysLate"]
        + 0.2 * df["NumberOfTime30-59DaysPastDueNotWorse"]
    )
    prob = 1 / (1 + np.exp(-(risk - 1.6)))
    df[TARGET] = (rng.random(n) < prob).astype(int)
    return df


@pytest.fixture(scope="session")
def small_xgb_model_path(tmp_path_factory):
    """Train a tiny XGBoost pipeline and return its serialized path."""
    import joblib
    from sklearn.pipeline import Pipeline

    from src.features.engineering import MODEL_FEATURES, create_features
    from src.models import model_zoo
    from src.models.preprocessing import build_preprocessor

    frame = make_frame()
    engineered, _ = create_features(frame)
    X = engineered[MODEL_FEATURES]
    y = engineered[TARGET].to_numpy()
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("model", model_zoo.xgboost(n_estimators=40, max_depth=3)),
        ]
    )
    pipe.fit(X, y)
    path = tmp_path_factory.mktemp("models") / "best_model.joblib"
    joblib.dump(pipe, path)
    return path


def make_frame(n: int = 400) -> pd.DataFrame:
    """Dataframe generator (module-level for reuse)."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1.5, n),
            "age": rng.integers(21, 85, n),
            "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 8, n),
            "DebtRatio": rng.uniform(0, 4, n),
            "MonthlyIncome": rng.uniform(100, 25000, n),
            "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n),
            "NumberOfTimes90DaysLate": rng.integers(0, 8, n),
            "NumberRealEstateLoansOrLines": rng.integers(0, 10, n),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 6, n),
            "NumberOfDependents": rng.integers(0, 6, n).astype("float"),
        }
    )
    risk = (
        0.5 * df["RevolvingUtilizationOfUnsecuredLines"]
        + 0.3 * df["NumberOfTimes90DaysLate"]
        + 0.2 * df["NumberOfTime30-59DaysPastDueNotWorse"]
    )
    prob = 1 / (1 + np.exp(-(risk - 1.6)))
    df[TARGET] = (rng.random(n) < prob).astype(int)
    return df
