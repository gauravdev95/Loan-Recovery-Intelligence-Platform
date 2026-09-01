"""Recovery-priority engine tests."""
from __future__ import annotations

from src.risk.priority_engine import (
    PRIORITY_LEVELS,
    recommended_action,
    recovery_priority,
)


def test_low_risk_normal():
    assert recovery_priority(0.1) == "NORMAL"


def test_critical_probability_escalates():
    assert recovery_priority(0.75) == "CRITICAL_PRIORITY"
    assert recovery_priority(0.99) == "CRITICAL_PRIORITY"


def test_serious_delinquency_with_high_probability_escalates():
    assert recovery_priority(0.5, serious_delinquency=1) == "CRITICAL_PRIORITY"


def test_high_risk_level_high_priority():
    assert recovery_priority(0.65) == "HIGH_PRIORITY"


def test_follow_up_on_or_above_cutoff():
    assert recovery_priority(0.45) == "FOLLOW_UP"


def test_financial_stress_at_medium_probability():
    assert recovery_priority(0.32, financial_stress=1) == "FOLLOW_UP"


def test_any_delinquency_gentle_escalation():
    assert recovery_priority(0.26, any_delinquency=1) == "FOLLOW_UP"


def test_priority_levels_enum():
    assert PRIORITY_LEVELS == (
        "NORMAL",
        "FOLLOW_UP",
        "HIGH_PRIORITY",
        "CRITICAL_PRIORITY",
    )


def test_recommended_action_never_empty():
    for p in PRIORITY_LEVELS:
        assert recommended_action(p).strip() != ""
