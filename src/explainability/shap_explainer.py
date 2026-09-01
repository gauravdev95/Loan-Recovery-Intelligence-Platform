"""SHAP explainability for the final model.

Provides:

* Global explanation — mean |SHAP| per feature (the most influential drivers
  of default risk across the population), plus an optional bar plot.
* Local explanation — top contributing SHAP factors for a single borrower.

The SHAP explainer is fitted on a sample of the **training** split so that
explanation is consistent with model training; local explanations then run on
any held-out or live row.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PLOTS_DIR

N_BACKGROUND_SAMPLE = 500


class ShapExplainer:
    """Thin wrapper around a fitted SHAP explainer object.

    ``raw`` holds the underlying explainer (e.g. TreeExplainer or
    LinearExplainer); using a generic wrapper keeps serialisation simple.
    """

    def __init__(self, raw: Any, feature_names: list[str]):
        self.raw = raw
        self.feature_names = list(feature_names)

    # -- construction helpers ------------------------------------------------
    @classmethod
    def for_tree_model(cls, model, X_background: pd.DataFrame) -> "ShapExplainer":
        import shap

        explainer = shap.TreeExplainer(model)
        # Warm the expected value (fast path for trees).
        _ = explainer.expected_value
        return cls(explainer, list(X_background.columns))

    @classmethod
    def for_linear_model(cls, model, X_background: pd.DataFrame) -> "ShapExplainer":
        import shap

        explainer = shap.LinearExplainer(model, X_background)
        return cls(explainer, list(X_background.columns))

    # -- global --------------------------------------------------------------
    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per feature over the provided (background) rows."""
        values = self.shap_values(X)
        import shap

        if isinstance(values, list):  # some SHAP versions return [neg, pos]
            values = values[1] if len(values) > 1 else values[0]
        arr = np.asarray(values)
        import shap

        mean_abs = np.abs(arr).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    # -- local ---------------------------------------------------------------
    def local_explanation(
        self, row: pd.DataFrame, top_k: int = 5
    ) -> dict[str, Any]:
        """Top contributing SHAP factors for a single row.

        Parameters
        ----------
        row : single-row DataFrame with model feature columns.
        top_k : number of top contributing features to report.

        Returns a dict with ``expected_value`` and ``top_factors`` where each
        factor has feature, value, shap_value and direction.
        """
        values = self.shap_values(row)
        import shap

        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        sv = np.asarray(values).reshape(-1)

        # Log-odds direction: positive SHAP pushes probability up (default).
        factors = []
        for feat, val, shap_val in zip(self.feature_names, row.iloc[0], sv):
            direction = "increases_risk" if shap_val >= 0 else "reduces_risk"
            factors.append(
                {
                    "feature": feat,
                    "value": _safe_scalar(val),
                    "shap_value": _safe_scalar(float(shap_val)),
                    "direction": direction,
                }
            )
        factors.sort(key=lambda f: abs(f["shap_value"]), reverse=True)
        return {
            "expected_value": _safe_scalar(float(np.asarray(self.expected_value()))),
            "top_factors": factors[:top_k],
        }

    def expected_value(self) -> Any:
        ev = self.raw.expected_value
        if isinstance(ev, np.ndarray):
            ev = float(ev[1]) if ev.size > 1 else float(ev[0])
        return ev

    def shap_values(self, X: pd.DataFrame):
        import shap

        return self.raw.shap_values(X)

    # -- persistence ---------------------------------------------------------
    def save_global_importance(
        self, importance: pd.DataFrame, path: Path | None = None
    ) -> None:
        path = Path(path) if path else PLOTS_DIR / "shap_global_importance.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        importance.to_csv(path, index=False)

    def save_plot(self, importance: pd.DataFrame, path: Path | None = None) -> None:
        path = Path(path) if path else PLOTS_DIR / "shap_global_importance.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            top = importance.head(12)[::-1]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(top["feature"], top["mean_abs_shap"], color="#4C78A8")
            ax.set_xlabel("mean |SHAP|")
            ax.set_title("Global feature importance (SHAP)")
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
        except Exception:  # pragma: no cover - plotting is non-critical
            return


def _safe_scalar(x: Any) -> float | None:
    try:
        if isinstance(x, float | int):
            return float(x)
        return float(x)
    except (TypeError, ValueError):
        return None
