"""Read-side queries for borrowers, predictions and recovery priorities."""
from __future__ import annotations

from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.database import models as db_models


def list_borrowers(
    db: Session,
    *,
    search: str | None = None,
    risk_level: str | None = None,
    priority: str | None = None,
    min_probability: float | None = None,
    sort_by: str = "created_at",
    order: str = "desc",
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Return (rows, total) for the borrower dashboard list with filters."""
    q = (
        db.query(
            db_models.Borrower,
            db_models.Prediction,
            db_models.RecoveryPriority,
        )
        .outerjoin(db_models.Prediction, db_models.Prediction.borrower_id == db_models.Borrower.borrower_id)
        .outerjoin(
            db_models.RecoveryPriority,
            db_models.RecoveryPriority.borrower_id == db_models.Borrower.borrower_id,
        )
    )

    if search:
        q = q.filter(db_models.Borrower.borrower_id.ilike(f"%{search}%"))
    if risk_level:
        q = q.filter(db_models.Prediction.risk_level == risk_level.upper())
    if priority:
        q = q.filter(db_models.RecoveryPriority.priority == priority.upper())
    if min_probability is not None:
        q = q.filter(db_models.Prediction.default_probability >= min_probability)

    total = q.count()

    sortable = {
        "created_at": db_models.Borrower.created_at,
        "probability": db_models.Prediction.default_probability,
        "risk_level": db_models.Prediction.risk_level,
        "priority": db_models.RecoveryPriority.priority,
        "borrower_id": db_models.Borrower.borrower_id,
    }
    col = sortable.get(sort_by, db_models.Borrower.created_at)
    col = col.desc() if order == "desc" else col.asc()
    rows = q.order_by(col).offset(offset).limit(limit).all()

    out = []
    for borrower, pred, prio in rows:
        out.append(
            {
                "borrower_id": borrower.borrower_id,
                "default_probability": pred.default_probability if pred else None,
                "risk_level": pred.risk_level if pred else None,
                "recovery_priority": prio.priority if prio else None,
                "recommended_action": prio.recommended_action if prio else None,
                "model_version": pred.model_version if pred else None,
                "created_at": borrower.created_at.isoformat() if borrower.created_at else None,
            }
        )
    return out, total


def get_borrower(db: Session, borrower_id: str) -> dict[str, Any] | None:
    borrower = (
        db.query(db_models.Borrower)
        .filter(db_models.Borrower.borrower_id == borrower_id)
        .one_or_none()
    )
    if borrower is None:
        return None
    pred = (
        db.query(db_models.Prediction)
        .filter(db_models.Prediction.borrower_id == borrower_id)
        .one_or_none()
    )
    prio = (
        db.query(db_models.RecoveryPriority)
        .filter(db_models.RecoveryPriority.borrower_id == borrower_id)
        .one_or_none()
    )
    return {
        "borrower_id": borrower.borrower_id,
        "default_probability": pred.default_probability if pred else None,
        "risk_level": pred.risk_level if pred else None,
        "recovery_priority": prio.priority if prio else None,
        "recommended_action": prio.recommended_action if prio else None,
        "model_version": pred.model_version if pred else None,
        "features": {
            "revolving_utilization": borrower.revolving_utilization,
            "age": borrower.age,
            "num_30_59_past_due": borrower.num_30_59_past_due,
            "debt_ratio": borrower.debt_ratio,
            "monthly_income": borrower.monthly_income,
            "open_credit_lines": borrower.open_credit_lines,
            "num_90_late": borrower.num_90_late,
            "real_estate_loans": borrower.real_estate_loans,
            "num_60_89_past_due": borrower.num_60_89_past_due,
            "dependents": borrower.dependents,
        },
        "created_at": borrower.created_at.isoformat() if borrower.created_at else None,
    }


def dashboard_stats(db: Session) -> dict[str, Any]:
    """Aggregate counts for the dashboard hero cards."""
    total = db.query(db_models.Borrower).count()

    def _count_by(level: str) -> int:
        return (
            db.query(db_models.Prediction)
            .filter(db_models.Prediction.risk_level == level)
            .count()
        )

    avg_prob = (
        db.query(func.avg(db_models.Prediction.default_probability))
        .scalar()
    )
    return {
        "total_borrowers": total,
        "critical_risk": _count_by("CRITICAL"),
        "high_risk": _count_by("HIGH"),
        "medium_risk": _count_by("MEDIUM"),
        "low_risk": _count_by("LOW"),
        "avg_default_probability": float(avg_prob) if avg_prob is not None else 0.0,
    }


def recovery_queue(db: Session, limit: int = 100) -> list[dict[str, Any]]:
    """Highest-priority borrowers first (CRITICAL > HIGH > FOLLOW_UP > NORMAL)."""
    order = {
        "CRITICAL_PRIORITY": 0,
        "HIGH_PRIORITY": 1,
        "FOLLOW_UP": 2,
        "NORMAL": 3,
    }
    rows, _ = list_borrowers(db, sort_by="priority", order="asc", limit=limit)
    rows_sorted = sorted(
        rows, key=lambda r: (order.get(r["recovery_priority"], 9), -(r["default_probability"] or 0))
    )
    return rows_sorted[:limit]
