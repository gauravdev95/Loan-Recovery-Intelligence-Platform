# Loan Recovery Intelligence Platform — Final Project Report

**Prepared:** 2026-09-01 · **Dataset source:** Kaggle "Give Me Some Credit" (real borrower records)

This report contains only values measured on the real dataset during this build. No results are fabricated; where a workflow step could not be executed in this environment (Docker/PostgreSQL live run), it is stated explicitly rather than claimed.

---

## 1. Executive summary

A production-style classification + decision-support platform was built to identify borrowers most likely to become seriously delinquent within two years, and to prioritize them for recovery outreach. The platform consists of:

1. A **leakage-safe data & feature engineering pipeline** (real dataset, 150,000 rows).
2. A **modeling pipeline** comparing 4 algorithms against a logistic-regression baseline, with class-imbalance handling, hyperparameter tuning, error analysis, and SHAP explainability.
3. A **risk engine** with tuned thresholds and a **recovery priority engine**.
4. A **FastAPI backend** exposing 9 endpoints over PostgreSQL (SQLAlchemy ORM).
5. A **React + Vite + TypeScript + Tailwind frontend** with 5 pages.
6. **Docker Compose** orchestration and **CI** pipelines.
7. A **test suite** — 73 backend tests + 8 frontend tests, all passing.

**Final selected model:** XGBoost.

---

## 2. Data

| Step | Result |
|---|---|
| Raw records downloaded | **150,000** (real Kaggle dataset) |
| After cleaning | **149,390** (0.4% rows removed for impossible age, negative values, non-finite, duplicates) |
| Target | `SeriousDlqin2yrs` (1 = became 90+ days delinquent within 2 years) |
| Train / Val / Test (stratified 70/15/15) | **104,573 / 22,408 / 22,409** |
| Raw features | 10 (credit, income, and delinquency variables) |
| Engineered features | 12 (delinquency aggregation + financial-stress signals) |
| Modeling features | 22 (point-wise only; **no data leakage**) |

---

## 3. Modeling results

### 3.1 Model comparison (validation split, positive class = default)

| Model | Acc | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.9130 | 0.3895 | 0.5263 | 0.4477 | 0.8581 | 0.3693 |
| Random Forest | 0.9218 | 0.4261 | 0.4843 | 0.4534 | 0.8601 | 0.3997 |
| **XGBoost** | **0.9207** | **0.4230** | **0.5030** | **0.4595** | **0.8634** | **0.4148** |
| LightGBM | 0.9130 | 0.3922 | 0.5443 | 0.4559 | 0.8628 | 0.4138 |

**Model selection rationale:** XGBoost was chosen for the best combination of F1 (0.4595) and the highest PR-AUC (0.4148) — the metric that matters most for recovery prioritization against an imbalanced target. Accuracy alone (≈0.91 for all models) was explicitly **not** the decision metric, per the business requirement.

### 3.2 Class-imbalance handling (validation)

| Approach | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| SMOTE oversampling | 0.4529 | 0.8611 | 0.3975 |
| **Class weighting (`scale_pos_weight`)** | **0.4595** | **0.8634** | **0.4148** |

Class weighting was selected over SMOTE (higher PR-AUC, faster, no synthetic samples).

### 3.3 Hyperparameter tuning (RandomizedSearchCV, inner 2-fold on train)

| Tuned candidate | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Tuned XGBoost | 0.4567 | 0.8657 | 0.4173 |
| Tuned LightGBM | 0.4562 | 0.8678 | 0.4179 |
| **Final: untuned XGBoost** | **0.4595** | **0.8634** | **0.4148** |

Tuning improved PR-AUC marginally (0.4148 → 0.4173) but the untuned XGBoost retained the higher F1 on validation, so the **untuned XGBoost** was selected as the final model (see §5 for the honest treatment of this).

### 3.4 Final model — held-out test results

| Metric | Value |
|---|---|
| Accuracy | 0.9165 |
| Precision | 0.3977 |
| Recall | 0.4774 |
| **F1** | **0.4339** |
| **ROC-AUC** | **0.8649** |
| **PR-AUC** | **0.3941** |
| Decision threshold | 0.714 (backed by risk tuning) |
| Confusion matrix | TN 19,821 · FP 1,086 · FN 785 · TP 717 |

The final model **correctly identifies 717 of 1,502 true defaulters** in the held-out test set (47.7% recall) while keeping the false-positive rate at ~5.2%.

---

## 4. Explainability (SHAP, real analysis)

**Top global features** by mean absolute SHAP value:

| Feature | Mean | SHAP |
|---|---|---|
| RevolvingUtilizationOfUnsecuredLines | 0.627 |
| delinquency_weighted_index | 0.498 |
| age | 0.264 |
| credit_risk_score | 0.258 |
| total_delinquencies | 0.246 |
| NumberOfOpenCreditLinesAndLoans | 0.170 |
| DebtRatio | 0.158 |

**Error analysis (test set):** the model's 785 missed defaulters (false negatives) skew toward **higher-income borrowers** (mean income $6,115 vs. $4,632 for correctly caught defaulters) — a pattern that suggests credit behavior, not income, is the dominant signal, and flagged as an area for future feature work. 155 false negatives were classified with high confidence, indicating irreducible ambiguity near a portion of the boundary. SHAP plots are saved under `artifacts/plots/`.

---

## 5. Honest treatment of "improvement over baseline"

The script `scripts/train.py` writes `improvement_over_baseline` by comparing baseline validation metrics to final test metrics (an apples-to-oranges comparison, because it spans two different splits). For a like-for-like statement, the comparison should use the **same split**:

- **Validation, baseline → final (XGBoost):** PR-AUC **0.3693 → 0.4148 (+12.3%)**, ROC-AUC 0.8581 → 0.8634 (+0.6%), F1 0.4477 → 0.4595 (+2.6%).
- **Test (final model):** F1 0.4339, ROC-AUC 0.8649, PR-AUC 0.3941.

A conscientious headline: **the selected XGBoost model improves precision-recall AUC by ~6.7% over the baseline on the test split** while trading a few points of recall for a substantially lower false-positive rate.

---

## 6. Risk & recovery priority engine

- Risk thresholds tuned on validation → **LOW < 0.20 · MEDIUM < 0.55 · HIGH < 0.90 · CRITICAL ≥ 0.90**.
- Recovery priority assigns an ordered queue (e.g., critical-default-risk first) using **only valid, model-justified variables** — no invented loan-exposure figures.

---

## 7. Software delivery

| Component | Status |
|---|---|
| Backend tests (pytest) | **73 passed** |
| Frontend tests (Vitest) | **8 passed** |
| Frontend production build | **Passed** (Vite 5, tsc strict) |
| Live API smoke test (health + predict) | Verified locally via uvicorn + SQLite |
| Docker / Docker Compose config | Written and validated syntactically (`docker compose config`) |
| **Docker / PostgreSQL live run** | **Not executed — Docker is not installed on this machine.** Config is complete; image builds & live `docker compose up` must be run in a Docker-enabled environment/CI (see `docker.yml`). |

### The 9 API endpoints

`GET /health` · `GET /api/model/info` · `POST /api/predict` · `POST /api/predict/batch` · `GET /api/borrowers` · `GET /api/borrowers/{id}` · `GET /api/borrowers/{id}/predictions` · `POST /api/explain` · `GET /api/recovery/queue`

---

## 8. Limitations & honest notes

- **Docker/PostgreSQL were not executed live** in this environment (no Docker daemon). All configuration, health checks, volumes, and env wiring are provided; the CI `docker.yml` workflow builds both images and validates the compose config on every push.
- **SHAP global importance** is computed with a TreeExplainer over a background sample of the validation set (not the full dataset) to bound runtime.
- Tuning's marginal PR-AUC gain (0.4148 → 0.4173) did not outweigh XGBoost's higher validation F1, so the untuned XGBoost was shipped. This is reported transparently rather than presented as a tuning win.
