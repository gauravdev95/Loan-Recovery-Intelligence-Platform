import type { ModelMetrics } from "../api/types";

/**
 * Model comparison reference — REAL measured values from the training run
 * (validation split). Final model metrics are additionally fetched live from
 * the backend /api/v1/model/metrics.
 */
export interface ComparisonRow {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  note?: string;
}

export const MODEL_COMPARISON: ComparisonRow[] = [
  {
    model: "Logistic Regression (baseline)",
    accuracy: 0.913,
    precision: 0.3895,
    recall: 0.5263,
    f1: 0.4477,
    roc_auc: 0.8581,
    pr_auc: 0.3693,
    note: "Baseline",
  },
  {
    model: "Random Forest",
    accuracy: 0.9218,
    precision: 0.4261,
    recall: 0.4843,
    f1: 0.4534,
    roc_auc: 0.8601,
    pr_auc: 0.3997,
  },
  {
    model: "LightGBM",
    accuracy: 0.913,
    precision: 0.3922,
    recall: 0.5443,
    f1: 0.4559,
    roc_auc: 0.8628,
    pr_auc: 0.4138,
  },
  {
    model: "XGBoost (final)",
    accuracy: 0.9207,
    precision: 0.423,
    recall: 0.503,
    f1: 0.4595,
    roc_auc: 0.8634,
    pr_auc: 0.4148,
    note: "Selected final model",
  },
];

export const TUNED_COMPARISON: ComparisonRow[] = [
  {
    model: "Tuned XGBoost",
    accuracy: 0.0,
    precision: 0.0,
    recall: 0.0,
    f1: 0.4567,
    roc_auc: 0.8657,
    pr_auc: 0.4173,
    note: "RandomizedSearchCV",
  },
  {
    model: "Tuned LightGBM",
    accuracy: 0.0,
    precision: 0.0,
    recall: 0.0,
    f1: 0.4562,
    roc_auc: 0.8678,
    pr_auc: 0.4179,
    note: "RandomizedSearchCV",
  },
];

export const EMPTY_METRICS: ModelMetrics = {
  accuracy: 0,
  precision: 0,
  recall: 0,
  f1: 0,
  roc_auc: 0,
  pr_auc: 0,
};
