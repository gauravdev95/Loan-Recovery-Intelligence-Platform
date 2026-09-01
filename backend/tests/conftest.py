"""Backend test fixtures.

Builds a tiny trained model, points the FastAPI app at a temp SQLite database
and a temp model directory, then yields a TestClient. Environment must be set
BEFORE importing backend modules (settings is built at import time).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --- set env first ----------------------------------------------------------
_TMP_DIR = Path(os.environ.get("TMP", "/tmp")) if os.name == "posix" else None
os.environ.setdefault("DATABASE_URL", "sqlite:///./backend_test.db")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:5173")


def _build_fixture_model(model_dir: Path) -> Path:
    """Train a tiny XGBoost pipeline and write metadata artefacts."""
    from sklearn.pipeline import Pipeline

    from src.features.engineering import MODEL_FEATURES, create_features
    from src.models import model_zoo
    from src.models.preprocessing import build_preprocessor

    rng = np.random.default_rng(11)
    n = 300
    frame = pd.DataFrame(
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
        1.2 * frame["RevolvingUtilizationOfUnsecuredLines"]
        + 1.0 * frame["NumberOfTimes90DaysLate"]
        + 0.6 * frame["NumberOfTime30-59DaysPastDueNotWorse"]
    )
    prob = 1 / (1 + np.exp(-(risk - 2.2)))
    frame["SeriousDlqin2yrs"] = (rng.random(n) < prob).astype(int)

    engineered, _ = create_features(frame)
    X = engineered[MODEL_FEATURES]
    y = engineered["SeriousDlqin2yrs"].to_numpy()
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("model", model_zoo.xgboost(n_estimators=40, max_depth=3)),
        ]
    )
    pipe.fit(X, y)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / "best_model.joblib")
    joblib.dump(build_preprocessor(scale_numeric=False), model_dir / "preprocessing.joblib")
    (model_dir / "feature_metadata.json").write_text(
        json.dumps(
            {
                "features_version": "v1",
                "raw_features": MODEL_FEATURES,
                "target": "SeriousDlqin2yrs",
                "n_features": len(MODEL_FEATURES),
            }
        )
    )
    (model_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_name": "XGBoost [fixture]",
                "model_version": "test-1.0",
                "features_version": "v1",
                "target": "SeriousDlqin2yrs",
                "metrics": {"f1": 0.42, "roc_auc": 0.78, "pr_auc": 0.5},
                "threshold_configuration": {"risk_level_thresholds": [0.3, 0.6, 0.8]},
                "validation_threshold": 0.5,
            }
        )
    )
    return model_dir / "best_model.joblib"


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("models")
    model_path = _build_fixture_model(model_dir)

    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["PREPROCESSOR_PATH"] = str(model_dir / "preprocessing.joblib")
    os.environ["FEATURE_METADATA_PATH"] = str(model_dir / "feature_metadata.json")
    os.environ["MODEL_METADATA_PATH"] = str(model_dir / "model_metadata.json")
    os.environ["MODEL_BASE_DIR"] = str(model_dir)

    from fastapi.testclient import TestClient

    from backend.app.main import app

    # Rebuild db engine against sqlite (settings read env already).
    import backend.app.database.db as db

    db.engine.dispose()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def valid_payload() -> dict:
    return {
        "borrower_id": "BR-FIXTURE-0001",
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 42,
        "NumberOfTime30_59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.4,
        "MonthlyIncome": 5500.0,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 1,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2,
    }
