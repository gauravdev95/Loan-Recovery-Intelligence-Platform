"""Serialization of model artifacts and their metadata.

Saves/loads:
    models/best_model.joblib
    models/preprocessing.joblib
    models/feature_metadata.json
    models/model_metadata.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.config import (
    BEST_MODEL_PATH,
    FEATURE_METADATA_PATH,
    FEATURES_VERSION,
    MODEL_METADATA_PATH,
    PREPROCESSOR_PATH,
)
from src.features.engineering import ENGINEERED_FEATURES, MODEL_FEATURES
from src.models.preprocessing import CATEGORICAL_FEATURES
from src.risk.risk_engine import DEFAULT_THRESHOLDS


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def save_model(model: Any, path: Path | None = None) -> Path:
    path = Path(path) if path else BEST_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def save_preprocessor(pp: Any, path: Path | None = None) -> Path:
    path = Path(path) if path else PREPROCESSOR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pp, path)
    return path


def build_feature_metadata() -> dict[str, Any]:
    return {
        "features_version": FEATURES_VERSION,
        "raw_features": list(MODEL_FEATURES),
        "categorical_features": CATEGORICAL_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "n_features": len(MODEL_FEATURES),
        "target": "SeriousDlqin2yrs",
    }


def build_model_metadata(
    *,
    model_name: str,
    version: str,
    metrics: dict[str, Any],
    tuned_thresholds: list[float],
    tuned_parameters: dict[str, Any],
    class_weights: dict[str, Any],
    imbalance_strategy: str,
    train_rows: int,
    test_rows: int,
    baseline_metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the full model_metadata.json payload."""
    meta: dict[str, Any] = {
        "model_name": model_name,
        "model_version": version,
        "model_family": model_name.split("[")[0].strip(),
        "features_version": FEATURES_VERSION,
        "feature_list": MODEL_FEATURES,
        "target": "SeriousDlqin2yrs",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "threshold_configuration": {
            "risk_level_thresholds": [float(t) for t in tuned_thresholds],
            "default_thresholds": [float(t) for t in DEFAULT_THRESHOLDS],
            "decision_threshold_notes": (
                "Risk-band thresholds tuned on the validation split."
            ),
        },
        "tuned_parameters": tuned_parameters,
        "class_weights": class_weights,
        "imbalance_strategy": imbalance_strategy,
        "preprocessing_version": FEATURES_VERSION,
        "class_distribution": None,
        "data_splits": {"train_rows": int(train_rows), "test_rows": int(test_rows)},
    }
    if extra:
        meta.update(extra)
    return meta


def save_feature_metadata(path: Path | None = None) -> Path:
    path = Path(path) if path else FEATURE_METADATA_PATH
    _write_json(path, build_feature_metadata())
    return path


def save_model_metadata(meta: dict[str, Any], path: Path | None = None) -> Path:
    path = Path(path) if path else MODEL_METADATA_PATH
    _write_json(path, meta)
    return path


def load_model(path: Path | None = None) -> Any:
    return joblib.load(Path(path) if path else BEST_MODEL_PATH)


def load_preprocessor(path: Path | None = None) -> Any:
    return joblib.load(Path(path) if path else PREPROCESSOR_PATH)


def load_metadata(path: Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else MODEL_METADATA_PATH
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
