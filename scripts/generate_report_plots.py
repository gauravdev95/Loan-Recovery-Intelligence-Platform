"""Generate the README evaluation figures from the real measured artifacts.

Reads ``artifacts/reports/experiment_summary.json`` and
``models/model_metadata.json`` (both produced by training) and writes three
figures under ``artifacts/plots/``:

* confusion_matrix.png      — final model confusion matrix (held-out test)
* model_comparison.png      — 4-model comparison on business metrics (validation)
* improvement_baseline_vs_final.png — baseline vs final XGBoost with relative
                                      % change, like-for-like on the validation split

Run with::
    python scripts/generate_report_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PLOT = ROOT / "artifacts" / "plots"
PLOT.mkdir(parents=True, exist_ok=True)

GRAY = "#64748B"


def _load(name: str):
    return json.load(open(ROOT / "artifacts" / "reports" / name))


def cm_plot(ax, cm: dict) -> None:
    """Render a 2x2 confusion-matrix heatmap on ``ax``."""
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    labels = [
        ["True negative\n%d" % cm["tn"], "False positive\n%d" % cm["fp"]],
        ["False negative\n%d" % cm["fn"], "True positive\n%d" % cm["tp"]],
    ]
    im = ax.imshow(grid, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted: 0\n(no default)", "Predicted: 1\n(default)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual: 0\n(no default)", "Actual: 1\n(default)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha="center", va="center", size=12,
                    color="white" if grid[i, j] >= grid.max() / 2 else "black")
    ax.set_title(f"Confusion matrix — test set (n={cm['n']})" if "n" in cm
                 else "Confusion matrix — held-out test set", size=11)


def main() -> None:
    summary = _load("experiment_summary.json")
    meta = json.load(open(ROOT / "models" / "model_metadata.json"))
    cm = meta["metrics"]["confusion_matrix"]

    # ---- 1. Confusion matrix (final model, test) --------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_plot(ax, cm)
    fig.tight_layout()
    fig.savefig(PLOT / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    print("Saved -> artifacts/plots/confusion_matrix.png")

    # ---- 2. Model comparison (validation split) ---------------------------
    comp = summary["comparison"]
    models = [
        "Logistic Regression (baseline)",
        "Random Forest",
        "XGBoost",
        "LightGBM",
    ]
    short = ["LogReg\n(baseline)", "Random\nForest", "XGBoost", "LightGBM"]
    metrics = ["f1", "roc_auc", "pr_auc"]
    colors = ["#94A3B8", "#64748B", "#2563EB", "#38BDF8"]
    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, m in enumerate(metrics):
        vals = [comp[n][m] for n in models]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=m.upper(),
                      color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                    ha="center", size=8, color=GRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(short, size=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — validation split (positive = default)")
    ax.legend(loc="lower right", frameon=False)
    ax.axhline(0, color="black", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT / "model_comparison.png", dpi=140)
    plt.close(fig)
    print("Saved -> artifacts/plots/model_comparison.png")

    # ---- 3. Baseline vs final improvement (validation, like-for-like) -----
    base = summary["baseline"]
    final = summary["comparison"]["XGBoost"]
    keys = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    b_vals = [base[k] for k in keys]
    f_vals = [final[k] for k in keys]
    rel = [(f - b) / b * 100 for b, f in zip(b_vals, f_vals)]

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    bar_b = ax.bar(x - width / 2, b_vals, width, label="Logistic Regression (baseline)",
                   color="#94A3B8")
    bar_f = ax.bar(x + width / 2, f_vals, width, label="XGBoost (final)",
                   color="#2563EB")
    for i, (bb, bf, r) in enumerate(zip(bar_b, bar_f, rel)):
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.01,
                f"{b_vals[i]:.3f}", ha="center", size=8, color=GRAY)
        ax.text(bf.get_x() + bf.get_width() / 2, bf.get_height() + 0.01,
                f"{f_vals[i]:.3f}", ha="center", size=8, color="#1D4ED8")
        sign = "+" if r >= 0 else ""
        ax.annotate(f"{sign}{r:.1f}%", (bf.get_x() + bf.get_width() / 2, 0.0),
                    xytext=(0, -34), textcoords="offset points",
                    ha="center", size=9, fontweight="bold", color="#0F766E")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs final model — validation split "
                 "(relative % change shown above final bars)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT / "improvement_baseline_vs_final.png", dpi=140)
    plt.close(fig)
    print("Saved -> artifacts/plots/improvement_baseline_vs_final.png")

    # ---- small sanctioned summary to stdout -------------------------------
    print("\nLike-for-like validation improvement (baseline -> XGBoost):")
    for l, b, f, r in zip(labels, b_vals, f_vals, rel):
        sign = "+" if r >= 0 else ""
        print(f"  {l:10s} {b:.4f} -> {f:.4f}  ({sign}{r:.1f}% rel.)")


if __name__ == "__main__":
    main()
