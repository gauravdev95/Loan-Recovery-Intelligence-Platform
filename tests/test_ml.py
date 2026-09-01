"""ML pipeline tests (preprocessing, prediction, model loading)."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from src.features.engineering import MODEL_FEATURES, create_features
from src.models.preprocessing import build_preprocessor, preprocessor_feature_count


def test_preprocessor_emits_expected_columns(sample_frame):
    engineered, _ = create_features(sample_frame)
    X = engineered[MODEL_FEATURES]
    pp = build_preprocessor(scale_numeric=True)
    pp.fit(X, sample_frame["SeriousDlqin2yrs"])
    transformed = pp.transform(X)
    assert transformed.shape[1] == preprocessor_feature_count() == len(MODEL_FEATURES)


def test_model_loading_and_probability_range(small_xgb_model_path):
    model = joblib.load(small_xgb_model_path)
    assert isinstance(model, Pipeline)
    engineered, _ = create_features(make_frame_for_test())
    X = engineered[MODEL_FEATURES].head(10)
    proba = model.predict_proba(X)[:, 1]
    assert len(proba) == 10
    assert ((proba >= 0) & (proba <= 1)).all()


def test_split_stratification(sample_frame):
    from sklearn.model_selection import train_test_split

    engineered, _ = create_features(sample_frame)
    y = engineered["SeriousDlqin2yrs"]
    tr, te, ytr, yte = train_test_split(
        engineered, y, test_size=0.3, random_state=42, stratify=y
    )
    assert abs(ytr.mean() - y.mean()) < 0.05
    assert abs(yte.mean() - y.mean()) < 0.05


def make_frame_for_test():
    from tests.conftest import make_frame

    return make_frame(120)
