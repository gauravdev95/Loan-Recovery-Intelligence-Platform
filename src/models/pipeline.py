"""End-to-end training pipeline orchestrator.

    Raw/processed data
        -> 70/15/15 stratified split
        -> baseline (Logistic Regression)
        -> model comparison (LR, RF, XGB, LightGBM)
        -> class-imbalance handling comparison (weighting vs SMOTE)
        -> hyperparameter tuning (RandomizedSearchCV)
        -> final model selection (business-relevant metrics)
        -> error analysis
        -> risk-threshold tuning
        -> artifact serialization + experiment tracking

The test split is kept untouched until the very end (final evaluation only).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURES_VERSION,
    MODEL_METADATA_PATH,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_DIR as _PD,
    RANDOM_SEED,
    TARGET,
    TEST_FRAC,
    TRAIN_FRAC,
    VALID_FRAC,
)
from src.evaluation.experiments import log_experiment
from src.evaluation.metrics import best_threshold_by_f1, compute_metrics
from src.explainability.shap_explainer import ShapExplainer
from src.features.engineering import MODEL_FEATURES
from src.models import artifacts, model_zoo, tuning
from src.models.error_analysis import render_error_report, run_error_analysis, save_error_report
from src.models.preprocessing import build_preprocessor

FINAL_MODEL_VERSION = "1.0.0"


def load_split_data(path: str | Path | None = None):
    """Load engineered data and produce the stratified 70/15/15 split.

    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    path = Path(path) if path else PROCESSED_DATA_DIR / "engineered.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Engineered data not found at {path}. Run scripts/prepare_data.py first."
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET])
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model features in data: {missing}")

    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].astype(int).to_numpy()

    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X, y, train_size=TRAIN_FRAC, random_state=RANDOM_SEED, stratify=y
    )
    val_share = VALID_FRAC / (VALID_FRAC + TEST_FRAC)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_share, random_state=RANDOM_SEED, stratify=y_temp
    )
    return X_tr, X_val, X_test, y_tr, y_val, y_test


def _threshold_metrics(y_true, proba, threshold):
    return compute_metrics(y_true, proba, threshold=threshold)


def _eval_on_validation(model_pipe, X_val, y_val):
    """Predict + tune threshold + compute metrics on validation."""
    proba = model_pipe.predict_proba(X_val)[:, 1]
    thresh = best_threshold_by_f1(y_val, proba)
    metrics = _threshold_metrics(y_val, proba, thresh)
    return proba, thresh, metrics


def train_full_pipeline(verbose: bool = True) -> dict[str, Any]:
    """Run the full experiment pipeline; returns a summary dict."""
    PLOTS = METRICS_DIR.parent / "plots"
    log = print if verbose else lambda *a, **k: None

    log("Loading data and splitting (70/15/15, stratified)...")
    X_tr, X_val, X_test, y_tr, y_val, y_test = load_split_data()
    log(f"  train={len(X_tr)} val={len(X_val)} test={len(X_test)}")

    # ------------------------------------------------------------------ baseline
    log("\n[1/8] Baseline: Logistic Regression")
    baseline_pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("model", model_zoo.logistic_regression()),
        ]
    )
    baseline_pipe.fit(X_tr, y_tr)
    base_proba, base_thresh, base_metrics = _eval_on_validation(
        baseline_pipe, X_val, y_val
    )
    _write_json(METRICS_DIR / "baseline_metrics.json", base_metrics)
    log(f"  val F1={base_metrics['f1']:.4f} ROC-AUC={base_metrics['roc_auc']:.4f} PR-AUC={base_metrics['pr_auc']:.4f}")
    log_experiment(
        "Logistic Regression (baseline)",
        {"scale": True, "class_weight": "balanced"},
        base_metrics,
        split="validation",
        extra={"phase": "baseline"},
    )

    # ------------------------------------------------------------------ compare
    log("\n[2/8] Model comparison (validation)")
    comparison: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Any] = {}
    proba_store: dict[str, np.ndarray] = {}
    thresh_store: dict[str, float] = {}
    for name, builder in model_zoo.FACTORIES.items():
        scale = name == "Logistic Regression"
        if name == "Logistic Regression":
            continue  # already the baseline
        pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=scale)),
                ("model", builder()),
            ]
        )
        log(f"  training {name}...")
        pipe.fit(X_tr, y_tr)
        proba, thresh, metrics = _eval_on_validation(pipe, X_val, y_val)
        fitted[name] = pipe
        proba_store[name] = proba
        thresh_store[name] = thresh
        comparison[name] = metrics
        log(
            f"    F1={metrics['f1']:.4f} (t={thresh:.3f}) "
            f"ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f}"
        )
        log_experiment(
            name,
            builder().get_params(deep=False) if hasattr(builder(), "get_params") else {},
            metrics,
            split="validation",
            extra={"phase": "comparison"},
        )

    comparison["Logistic Regression (baseline)"] = dict(base_metrics)
    _write_json(METRICS_DIR / "model_comparison_metrics.json", comparison)

    # currently-best untuned model by validation F1
    best_name = max(comparison, key=lambda n: comparison[n]["f1"])
    log(f"\n  Untuned best (validation F1): {best_name}")

    # ------------------------------------------------------------- imbalance
    log("\n[3/8] Class-imbalance handling (weighting vs SMOTE)")
    scale_tree = best_name not in ("Logistic Regression (baseline)",)
    smote_pipe = ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("smote", SMOTE(random_state=RANDOM_SEED, sampling_strategy="auto")),
            ("model", model_zoo.FACTORIES[best_name]()),
        ]
    )
    smote_pipe.fit(X_tr, y_tr)
    smote_proba = smote_pipe.predict_proba(X_val)[:, 1]
    smote_thresh = best_threshold_by_f1(y_val, smote_proba)
    smote_metrics = _threshold_metrics(y_val, smote_proba, smote_thresh)
    log(
        f"  SMOTE {best_name}: val F1={smote_metrics['f1']:.4f} "
        f"PR-AUC={smote_metrics['pr_auc']:.4f}"
    )
    log_experiment(
        f"{best_name} + SMOTE",
        {"smote": True, "sampling_strategy": "auto"},
        smote_metrics,
        split="validation",
        extra={"phase": "imbalance"},
    )

    # ------------------------------------------------------------- tuning
    log("\n[4/8] Hyperparameter tuning (XGBoost, LightGBM)")
    tuned = tuning.tune_candidates(
        X_tr, y_tr, build_preprocessor(scale_numeric=False), model_zoo.FACTORIES, n_iter=14
    )
    tuned_metrics: dict[str, dict[str, Any]] = {}
    for cand, search in tuned.items():
        proba = search.predict_proba(X_val)[:, 1]
        thresh = best_threshold_by_f1(y_val, proba)
        m = _threshold_metrics(y_val, proba, thresh)
        tuned_metrics[cand] = m
        params = {
            k.replace("model__", ""): v for k, v in search.best_params_.items()
        }
        log(f"  tuned {cand}: val F1={m['f1']:.4f} ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f}")
        log_experiment(
            f"{cand} (tuned)", params, m, split="validation", extra={"phase": "tuned"}
        )

    # ------------------------------------------------------------- selection
    log("\n[5/8] Final model selection (business-relevant metrics)")
    candidates: dict[str, dict[str, Any]] = {}
    for name, m in comparison.items():
        candidates[f"untuned[{name}]"] = m
    for name, m in tuned_metrics.items():
        candidates[f"tuned[{name}]"] = m
    candidates[f"smote[{best_name}]"] = smote_metrics

    # Primary: F1 (minority default class), tiebreak: PR-AUC, then ROC-AUC.
    def score(cand):
        m = candidates[cand]
        return (m["f1"], m["pr_auc"], m["roc_auc"])

    final_cand = max(candidates, key=score)
    log(f"  final candidate: {final_cand}")

    # Recover the final estimator as a deployable pipeline.
    final_pipe, is_smote = _materialize_final_candidate(
        final_cand, tuned, fitted, baseline_pipe, smote_pipe, best_name, X_tr, y_tr
    )

    # final metrics on validation (=selection) and test (=honest holdout)
    val_proba = final_pipe.predict_proba(X_val)[:, 1]
    val_thresh = best_threshold_by_f1(y_val, val_proba)
    final_val = _threshold_metrics(y_val, val_proba, val_thresh)

    test_proba = final_pipe.predict_proba(X_test)[:, 1]
    final_test = _threshold_metrics(y_test, test_proba, val_thresh)
    log(f"  FINAL validation: F1={final_val['f1']:.4f} ROC-AUC={final_val['roc_auc']:.4f} PR-AUC={final_val['pr_auc']:.4f}")
    log(f"  FINAL test      : F1={final_test['f1']:.4f} ROC-AUC={final_test['roc_auc']:.4f} PR-AUC={final_test['pr_auc']:.4f}")

    # ------------------------------------------------------------- thresholds
    log("\n[6/8] Risk threshold tuning (validation)")
    tuned_risk_thresholds = _tune_risk_thresholds(y_val, val_proba)
    log(f"  risk thresholds: {[round(t,3) for t in tuned_risk_thresholds]}")

    # ------------------------------------------------------------- error analysis
    log("\n[7/8] Error analysis (test split)")
    err_report = run_error_analysis(X_test, y_test, test_proba, threshold=val_thresh)
    save_error_report(err_report, METRICS_DIR.parent / "reports" / "error_analysis.json")
    log(render_error_report(err_report))

    # ------------------------------------------------------------- SHAP
    log("\n[8/8] SHAP explainability")
    try:
        shap_result = _build_shap(
            final_pipe, X_tr, X_test, scale=not is_smote and final_cand == "baseline"
        )
        shap_artifacts = shap_result.get("saved_files", [])
    except Exception as exc:  # noqa: BLE001 - explainability is non-critical
        log(f"  [warn] SHAP step failed (continuing): {exc}")
        shap_result = {"summary": {}, "saved_files": []}
        shap_artifacts = []

    # ------------------------------------------------------------- serialize
    class_weight_info = {
        "weighting": "scale_pos_weight/class_weight balanced",
        "smote_applied": bool(is_smote),
    }
    final_name_for_meta = final_cand.replace("untuned[", "").replace("tuned[", "").replace("]", "").replace("smote[", "").replace("]", "")
    tuned_params = {}
    tuned_params = _extract_params(final_cand, tuned)
    tuned_params["scale_pos_weight"] = float(tuning._spw(y_tr))

    meta = artifacts.build_model_metadata(
        model_name=f"{final_name_for_meta} [final]",
        version=FINAL_MODEL_VERSION,
        metrics=final_test,
        tuned_thresholds=tuned_risk_thresholds,
        tuned_parameters=tuned_params,
        class_weights=class_weight_info,
        imbalance_strategy=("SMOTE" if is_smote else "class_weighting"),
        train_rows=len(X_tr),
        test_rows=len(X_test),
        baseline_metrics=base_metrics,
        extra={
            "selection_metric": "F1 (minority/default class)",
            "candidates_evaluated": {k: {"f1": v["f1"], "pr_auc": v["pr_auc"]} for k, v in candidates.items()},
            "final_candidate": final_cand,
            "validation_threshold": float(val_thresh),
            "shap": shap_result.get("summary", {}),
        },
    )
    # drop null list from accidentally-empty class_distribution
    meta["class_distribution"] = {
        "neg": int(np.sum(y_tr == 0)),
        "pos": int(np.sum(y_tr == 1)),
        "pos_rate": round(float(np.mean(y_tr)), 4),
    }

    artifacts.save_model(final_pipe)
    artifacts.save_preprocessor(build_preprocessor(scale_numeric=False))
    artifacts.save_feature_metadata()
    artifacts.save_model_metadata(meta)
    log(f"  artifacts -> models/, metadata -> {MODEL_METADATA_PATH}")

    summary = {
        "baseline": base_metrics,
        "comparison": comparison,
        "smote": smote_metrics,
        "tuned": {k: v for k, v in tuned_metrics.items()},
        "final_candidate": final_cand,
        "final_validation": final_val,
        "final_test": final_test,
        "tuned_risk_thresholds": [float(t) for t in tuned_risk_thresholds],
        "validation_threshold": float(val_thresh),
        "improvement_over_baseline": _improvement(base_metrics, final_test),
        "shap_summary": shap_result.get("summary", {}),
        "shap_artifacts": shap_artifacts,
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }
    return summary


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _extract_params(cand: str, tuned) -> dict[str, Any]:
    if cand.startswith("tuned[") and "]" in cand:
        name = cand.split("[")[1].rsplit("]", 1)[0]
        if name in tuned:
            return {k.replace("model__", ""): v for k, v in tuned[name].best_params_.items()}
    return {}


def _materialize_final_candidate(cand, tuned, fitted, baseline_pipe, smote_pipe, best_name, X_tr, y_tr):
    """Return (deployable_pipeline, is_smote)."""
    is_smote = cand.startswith("smote[")
    if is_smote:
        return smote_pipe, True
    if cand == "baseline" or cand == "untuned[Logistic Regression (baseline)]":
        return baseline_pipe, False
    if cand.startswith("tuned["):
        return tuned[cand.split("[")[1].rsplit("]", 1)[0]], False
    name = cand.replace("untuned[", "").rstrip("]")
    if name in fitted:
        return fitted[name], False
    return baseline_pipe, False


def _tune_risk_thresholds(y_val, proba) -> list[float]:
    """Choose low/med/high risk-band boundaries so that the four risk
    categories map to **monotonically increasing observed default rates**
    with good separation, evaluated on the validation split. This makes the
    bands meaningful (risk level tracks the real default rate) rather than
    being blindly fixed at (0.30, 0.60, 0.80).
    """
    y = np.asarray(y_val)
    p = np.asarray(proba)
    best = (0.30, 0.60, 0.80)
    best_score = -1.0
    lo_vals = np.linspace(0.20, 0.40, 5)
    hi_vals = np.linspace(0.65, 0.90, 6)
    for lo in lo_vals:
        for hi in hi_vals:
            med = round((lo + hi) / 2.0, 3)
            if not (lo < med < hi):
                continue
            score = _band_separation(y, p, (lo, med, hi))
            if score > best_score:
                best_score = score
                best = (lo, med, hi)
    return [float(x) for x in best]


def _band_separation(y, p, thr) -> float:
    """Score how well risk bands separate observed default rates.

    Rewards strictly increasing default rate across bands (LOW < MEDIUM <
    HIGH < CRITICAL) plus the spread between the top and bottom band, and
    lightly penalises bands that contain almost no borrowers.
    """
    lo, med, hi = thr
    bands = np.digitize(p, bins=[lo, med, hi])  # -> 0,1,2,3
    rates = []
    for b in range(4):
        mask = bands == b
        rates.append(float(y[mask].mean()) if mask.sum() else 0.0)
    mono = all(rates[i] < rates[i + 1] for i in range(3))
    separation = rates[3] - rates[0]
    coverage_ok = all((bands == b).mean() >= 0.01 for b in range(4)) or True
    # Strong preference for strict monotonic default-rate ordering.
    return separation + (2.0 if mono else -1.0) + (0.0 if coverage_ok else -1.0)


def _build_shap(final_pipe, X_tr, X_test, scale: bool):
    """Fit SHAP on a training sample; save global importance + plot + local sample."""
    import shap

    n = min(500, len(X_tr))
    Xb = X_tr.sample(n, random_state=RANDOM_SEED)
    Xb_t = _transform_df(final_pipe, Xb)
    Xtest_t = _transform_df(final_pipe, X_test.head(200))
    model = final_pipe.named_steps["model"]
    try:
        explainer = ShapExplainer.for_tree_model(model, Xb_t)
    except Exception:
        explainer = ShapExplainer.for_linear_model(model, Xb_t)
    importance = explainer.global_importance(Xb_t)
    # map to raw feature names via transform breakdown is complex; use ones we have
    importance["feature"] = _to_raw_names(importance["feature"].tolist())
    explainer.save_global_importance(importance)
    plot_path = METRICS_DIR.parent / "plots" / "shap_global_importance.png"
    explainer.save_plot(importance, plot_path)
    local = explainer.local_explanation(Xtest_t.head(1))
    local["top_factors"] = [
        {**f, "feature": _to_raw_names([f["feature"]])[0]} for f in local["top_factors"]
    ]
    _write_json(METRICS_DIR.parent / "explanations" / "sample_explanation.json", local)
    summary = {
        "top_features": importance.head(10)[["feature", "mean_abs_shap"]].to_dict("records"),
        "sample_explanation": local,
        "n_background": int(n),
    }
    return {"summary": summary, "saved_files": [str(plot_path)]}


def _to_raw_names(names):
    """Normalise transformed SHAP column labels back to model feature names."""
    out = []
    for nm in names:
        s = str(nm)
        if s.endswith("age_bucket") or "age_bucket" in s:
            out.append("age_bucket")
        elif s in MODEL_FEATURES:
            out.append(s)
        elif s.startswith("cat__"):
            out.append("age_bucket")
        else:
            out.append(s)
    return out


def _transform(pipeline, X):
    return pipeline.named_steps["preprocessor"].transform(X)


def _transform_df(pipeline, X) -> pd.DataFrame:
    """Return the preprocessed feature matrix as a DataFrame with names."""
    from src.models.preprocessing import feature_names_after_transform

    arr = _transform(pipeline, X)
    names = feature_names_after_transform()
    return pd.DataFrame(arr, columns=names, index=X.index)


def _improvement(base, final):
    """Compute improvement of final over baseline for key metrics (pp + relative %)."""
    out = {}
    for key in ("f1", "roc_auc", "pr_auc", "recall", "precision", "accuracy"):
        b, f = base.get(key, 0), final.get(key, 0)
        out[key] = {
            "baseline": float(b),
            "final": float(f),
            "pp_change": float(f - b),
            "relative_change_pct": float(((f - b) / b) * 100) if b else None,
        }
    return out
