"""ModelService: loads serialized artifacts and serves predictions.

The serialized pipeline expects the full 22-feature set (10 raw + 12
engineered). This service re-applies the same engineered features from raw
inputs so API clients only need to send the 10 understandable raw columns.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import pandas as pd

from backend.app.config import settings
from src.config import RAW_FEATURES
from src.features.engineering import create_features
from src.risk import priority_engine
from src.risk.risk_engine import risk_level

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    pass


class ModelService:
    """Loads artifacts once and exposes prediction + explanation APIs."""

    def __init__(self, settings_obj=None):
        s = settings_obj or settings
        self.settings = s
        self.pipeline = None
        self.metadata: dict[str, Any] = {}
        self.feature_metadata: dict[str, Any] = {}
        self.explainer = None

    # -- lifecycle -----------------------------------------------------------
    def load(self) -> "ModelService":
        if not self.settings.model_path.exists():
            raise ModelNotLoadedError(
                f"Model artifact not found at {self.settings.model_path}. "
                "Run the training pipeline first (scripts/train.py)."
            )
        self.pipeline = joblib.load(self.settings.model_path)
        if self.settings.model_metadata_path.exists():
            with open(self.settings.model_metadata_path, encoding="utf-8") as fh:
                self.metadata = json.load(fh)
        if self.settings.feature_metadata_path.exists():
            with open(self.settings.feature_metadata_path, encoding="utf-8") as fh:
                self.feature_metadata = json.load(fh)
        # Build SHAP explainer lazily / best-effort.
        try:
            from src.explainability.shap_explainer import ShapExplainer

            model, _, names = self._tree_parts()
            # Background sample not needed for TreeExplainer; provide a dummy
            # frame with the right columns for Tree/Linear explainer health.
            self.explainer = ShapExplainer.for_tree_model(
                model, pd.DataFrame(columns=names)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAP explainer not initialised: %s", exc)
            self.explainer = None
        logger.info("Model loaded: %s", self.metadata.get("model_name"))
        return self

    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def _tree_parts(self):
        """Unwrap to (fitted XGBClassifier, preprocessor, transformed names).

        The serialized artifact is a RandomizedSearchCV whose best_estimator_
        is Pipeline(preprocessor -> XGBClassifier). SHAP's TreeExplainer needs
        the raw tree model applied to preprocessed features.
        """
        from src.models.preprocessing import feature_names_after_transform

        pipe = self.pipeline
        if hasattr(pipe, "best_estimator_"):
            pipe = pipe.best_estimator_
        preprocessor = pipe.named_steps["preprocessor"]
        model = pipe.named_steps["model"]
        return model, preprocessor, feature_names_after_transform()

    def _transform_for_shap(self, features: pd.DataFrame) -> pd.DataFrame:
        _, preprocessor, names = self._tree_parts()
        arr = preprocessor.transform(features)
        return pd.DataFrame(arr, columns=names, index=features.index)

    def raw_feature_names(self) -> list[str]:
        return self.feature_metadata.get("raw_features", RAW_FEATURES)

    # -- prediction ----------------------------------------------------------
    def raw_to_features(self, raw: dict[str, Any]) -> pd.DataFrame:
        """Build the 22-feature engineered frame from the 10 raw fields."""
        missing = [c for c in RAW_FEATURES if c not in raw]
        if missing:
            raise ValueError(f"Missing raw features: {missing}")
        frame = pd.DataFrame([raw])[RAW_FEATURES]
        engineered, _ = create_features(frame)
        return engineered

    def predict(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded():
            raise ModelNotLoadedError("Model is not loaded.")
        features = self.raw_to_features(raw)
        proba = float(self.pipeline.predict_proba(features)[0, 1])
        return self.build_output(raw, features, proba)

    def predict_proba(self, raw: dict[str, Any]) -> float:
        if not self.is_loaded():
            raise ModelNotLoadedError("Model is not loaded.")
        features = self.raw_to_features(raw)
        return float(self.pipeline.predict_proba(features)[0, 1])

    def build_output(
        self, raw: dict[str, Any], features: pd.DataFrame, proba: float
    ) -> dict[str, Any]:
        thresholds = self.metadata.get("threshold_configuration", {}).get(
            "risk_level_thresholds"
        )
        level = risk_level(proba, thresholds)
        priority = priority_engine.priority_from_borrower(proba, features.iloc[0].to_dict())
        action = priority_engine.recommended_action(priority)
        return {
            "default_probability": round(proba, 4),
            "risk_level": level,
            "recovery_priority": priority,
            "recommended_action": action,
            "model_version": self.metadata.get("model_version", "unknown"),
        }

    # -- info / explain ------------------------------------------------------
    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.metadata.get("model_name", "unknown"),
            "model_version": self.metadata.get("model_version", "unknown"),
            "features_version": self.metadata.get("features_version", "unknown"),
            "n_features": len(self.raw_feature_names()),
            "target": self.metadata.get("target", "SeriousDlqin2yrs"),
            "loaded": self.is_loaded(),
        }

    def model_metrics(self) -> dict[str, Any]:
        return self.metadata.get("metrics", {})

    def explain(self, raw: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
        if not self.is_loaded():
            raise ModelNotLoadedError("Model is not loaded.")
        features = self.raw_to_features(raw)
        proba = float(self.pipeline.predict_proba(features)[0, 1])
        output = self.build_output(raw, features, proba)

        if self.explainer is None:
            raise ModelNotLoadedError("SHAP explainer unavailable.")

        model, _, _ = self._tree_parts()
        # Force a fresh explainer on preprocessed features for correctness.
        try:
            from src.explainability.shap_explainer import ShapExplainer

            Xt = self._transform_for_shap(features)
            expl = ShapExplainer.for_tree_model(model, Xt)
            local = expl.local_explanation(Xt, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            raise ModelNotLoadedError(f"SHAP local explanation failed: {exc}") from exc

        return {
            "borrower_id": raw.get("borrower_id"),
            "default_probability": proba,
            "risk_level": output["risk_level"],
            "recovery_priority": output["recovery_priority"],
            "top_factors": local["top_factors"],
            "expected_value": local["expected_value"],
        }


model_service = ModelService()
