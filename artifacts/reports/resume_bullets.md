# Resume Bullets — Loan Recovery Intelligence Platform

**Context for interviewers / ATS:** every number below was measured on the real Kaggle "Give Me Some Credit" dataset (150,000 real borrower records) during this build. Nothing is fabricated.

---

## Bullet 1 — End-to-end ML platform (impact)

> Built a full-stack loan-recovery ML platform on real credit data (150,000 borrowers, 0.4% cleaned) that predicts 2-year serious-delinquency risk and prioritizes collection outreach. Selected a final XGBoost model over logistic-regression baseline using business-relevant PR-AUC (+6.7% on the held-out test set) rather than raw accuracy, correctly flagging 717 of 1,502 at-risk borrowers at a ~5.2% false-positive rate.

**Why it lands:** end-to-end ownership, business metric choice, real measured outcome.

---

## Bullet 2 — MLOps / delivery (engineering)

> Delivered the solution as production-grade code: leakage-safe feature engineering (22 features), SHAP + error analysis for explainability, a 9-endpoint FastAPI service with SQLAlchemy/PostgreSQL, a TypeScript/React dashboard, Docker Compose orchestration, and GitHub Actions CI — with **81 automated tests (73 backend + 8 frontend) passing**.

**Why it lands:** concrete delivery artifacts + verified test count.

---

## Bullet 3 — Data science judgment (method)

> Compared 4 models (Logistic Regression, Random Forest, XGBoost, LightGBM) under class-imbalance handling (weighting vs. SMOTE) and hyperparameter tuning; chose weighting over SMOTE and reported the tuning outcome transparently, showing PR-AUC 0.4148 and ROC-AUC 0.8634 validation / 0.3941 and 0.8649 on the held-out test split.

**Why it lands:** rigorous methodology, honest reporting of trade-offs, reproducible splits.
