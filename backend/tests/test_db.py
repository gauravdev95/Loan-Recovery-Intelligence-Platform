"""Database CRUD tests (run against SQLite via SQLAlchemy)."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from backend.app.database.models import (
    Base,
    Borrower,
    ModelRun,
    Prediction,
    RecoveryPriority,
)


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def test_connection(db_session):
    result = db_session.execute(text("SELECT 1")).scalar()
    assert result == 1


def test_borrower_crud(db_session):
    b = Borrower(
        borrower_id="DB-0001",
        age=35,
        monthly_income=5000.0,
        debt_ratio=0.4,
    )
    db_session.add(b)
    db_session.commit()

    fetched = (
        db_session.query(Borrower)
        .filter(Borrower.borrower_id == "DB-0001")
        .one_or_none()
    )
    assert fetched is not None
    assert fetched.age == 35

    fetched.age = 36
    db_session.commit()
    assert (
        db_session.query(Borrower)
        .filter(Borrower.borrower_id == "DB-0001")
        .one()
        .age
        == 36
    )

    db_session.delete(fetched)
    db_session.commit()
    assert (
        db_session.query(Borrower)
        .filter(Borrower.borrower_id == "DB-0001")
        .one_or_none()
        is None
    )


def test_prediction_relation(db_session):
    b = Borrower(borrower_id="DB-0002", age=40)
    db_session.add(b)
    db_session.flush()
    db_session.add(
        Prediction(
            borrower_id="DB-0002",
            default_probability=0.61,
            risk_level="HIGH",
            model_version="v-test",
        )
    )
    db_session.commit()
    pred = (
        db_session.query(Prediction)
        .filter(Prediction.borrower_id == "DB-0002")
        .one()
    )
    assert pred.risk_level == "HIGH"


def test_recovery_priority_crud(db_session):
    db_session.add(Borrower(borrower_id="DB-0003"))
    db_session.add(
        RecoveryPriority(
            borrower_id="DB-0003",
            priority="CRITICAL_PRIORITY",
            recommended_action="Escalate",
            default_probability=0.9,
        )
    )
    db_session.commit()
    row = (
        db_session.query(RecoveryPriority)
        .filter(RecoveryPriority.borrower_id == "DB-0003")
        .one()
    )
    assert row.priority == "CRITICAL_PRIORITY"


def test_model_run_crud(db_session):
    db_session.add(
        ModelRun(model_name="XGBoost", model_version="1.0.0", metrics="{}")
    )
    db_session.commit()
    assert db_session.query(ModelRun).count() == 1


def test_tables_created(client):
    from backend.app.database.db import engine

    from sqlalchemy import inspect

    inspector = inspect(engine)
    for table in ("borrowers", "predictions", "model_runs", "recovery_priorities"):
        assert table in inspector.get_table_names()
