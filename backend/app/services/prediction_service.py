"""Prediction orchestration: model -> risk -> priority -> persistence."""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from backend.app.database import models as db_models
from backend.app.ml.loader import ModelService
from backend.app.schemas import PredictionResponse

logger = logging.getLogger(__name__)

_RAW_COLUMN_MAP = {
    "revolving_utilization": "RevolvingUtilizationOfUnsecuredLines",
    "num_30_59_past_due": "NumberOfTime30-59DaysPastDueNotWorse",
    "debt_ratio": "DebtRatio",
    "monthly_income": "MonthlyIncome",
    "open_credit_lines": "NumberOfOpenCreditLinesAndLoans",
    "num_90_late": "NumberOfTimes90DaysLate",
    "real_estate_loans": "NumberRealEstateLoansOrLines",
    "num_60_89_past_due": "NumberOfTime60-89DaysPastDueNotWorse",
}


def predict_and_store(
    *,
    service: ModelService,
    db: Session,
    borrower_id: str | None,
    raw: dict[str, Any],
) -> PredictionResponse:
    """Run prediction and persist borrower/prediction/priority records.

    If ``borrower_id`` already exists its prediction/priority rows are updated
    (upsert semantics) so the dashboard stays consistent.
    """
    if borrower_id is None:
        borrower_id = f"BR{hash(tuple(sorted(raw.items()))) % 10_000_000:07d}"

    output = service.predict(raw)
    proba = output["default_probability"]
    level = output["risk_level"]
    priority = output["recovery_priority"]
    action = output["recommended_action"]
    version = output["model_version"]

    borrower = (
        db.query(db_models.Borrower)
        .filter(db_models.Borrower.borrower_id == borrower_id)
        .one_or_none()
    )
    if borrower is None:
        borrower = db_models.Borrower(borrower_id=borrower_id)
        db.add(borrower)
        db.flush()

    _apply_raw_to_borrower(borrower, raw)

    existing_pred = (
        db.query(db_models.Prediction)
        .filter(db_models.Prediction.borrower_id == borrower_id)
        .one_or_none()
    )
    if existing_pred is None:
        db.add(
            db_models.Prediction(
                borrower_id=borrower_id,
                default_probability=proba,
                risk_level=level,
                model_version=version,
            )
        )
    else:
        existing_pred.default_probability = proba
        existing_pred.risk_level = level
        existing_pred.model_version = version

    existing_pri = (
        db.query(db_models.RecoveryPriority)
        .filter(db_models.RecoveryPriority.borrower_id == borrower_id)
        .one_or_none()
    )
    if existing_pri is None:
        db.add(
            db_models.RecoveryPriority(
                borrower_id=borrower_id,
                priority=priority,
                recommended_action=action,
                default_probability=proba,
            )
        )
    else:
        existing_pri.priority = priority
        existing_pri.recommended_action = action
        existing_pri.default_probability = proba

    db.commit()

    return PredictionResponse(
        borrower_id=borrower_id,
        default_probability=proba,
        risk_level=level,
        recovery_priority=priority,
        recommended_action=action,
        model_version=version,
    )


def _apply_raw_to_borrower(borrower: db_models.Borrower, raw: dict[str, Any]) -> None:
    """Copy raw feature fields into the borrower snapshot."""
    mapping = {
        "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
        "age": "age",
        "NumberOfTime30-59DaysPastDueNotWorse": "num_30_59_past_due",
        "DebtRatio": "debt_ratio",
        "MonthlyIncome": "monthly_income",
        "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
        "NumberOfTimes90DaysLate": "num_90_late",
        "NumberRealEstateLoansOrLines": "real_estate_loans",
        "NumberOfTime60-89DaysPastDueNotWorse": "num_60_89_past_due",
        "NumberOfDependents": "dependents",
    }
    for raw_key, attr in mapping.items():
        value = raw.get(raw_key)
        if value is not None:
            setattr(borrower, attr, value)
