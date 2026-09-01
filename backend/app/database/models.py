"""SQLAlchemy ORM models (PostgreSQL-compatible; tests may run on SQLite).

Tables:
    borrowers           - borrower snapshot + scoring attributes
    predictions         - one prediction per borrower (latest wins)
    model_runs          - registry of model versions used
    recovery_priorities - recovery follow-up decisions
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Borrower(Base):
    __tablename__ = "borrowers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    borrower_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    revolving_utilization: Mapped[float] = mapped_column(Float, nullable=True)
    age: Mapped[int] = mapped_column(Integer, nullable=True)
    num_30_59_past_due: Mapped[int] = mapped_column(Integer, nullable=True)
    debt_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    monthly_income: Mapped[float] = mapped_column(Float, nullable=True)
    open_credit_lines: Mapped[int] = mapped_column(Integer, nullable=True)
    num_90_late: Mapped[int] = mapped_column(Integer, nullable=True)
    real_estate_loans: Mapped[int] = mapped_column(Integer, nullable=True)
    num_60_89_past_due: Mapped[int] = mapped_column(Integer, nullable=True)
    dependents: Mapped[int] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    prediction: Mapped["Prediction | None"] = relationship(
        back_populates="borrower", uselist=False
    )
    priority: Mapped["RecoveryPriority | None"] = relationship(
        back_populates="borrower", uselist=False
    )


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("borrower_id", name="uq_prediction_borrower"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    borrower_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("borrowers.borrower_id"), index=True
    )
    default_probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    model_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    borrower: Mapped["Borrower"] = relationship(back_populates="prediction")


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(32))
    metrics: Mapped[str] = mapped_column(Text, nullable=True)  # JSON string
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class RecoveryPriority(Base):
    __tablename__ = "recovery_priorities"
    __table_args__ = (
        UniqueConstraint("borrower_id", name="uq_priority_borrower"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    borrower_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("borrowers.borrower_id"), index=True
    )
    priority: Mapped[str] = mapped_column(String(24))
    recommended_action: Mapped[str] = mapped_column(Text)
    default_probability: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    borrower: Mapped["Borrower"] = relationship(back_populates="priority")
