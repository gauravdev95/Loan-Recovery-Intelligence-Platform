"""Recovery-priority engine (decision-support layer).

Combines the predicted default probability with **valid, available** risk
indicators to recommend a recovery follow-up priority. The "Give Me Some
Credit" dataset contains NO loan balance / exposure columns, so we deliberately
do not fabricate financial exposure. Priority is driven by:

  1. Predicted default probability (risk level).
  2. Financial-stress / serious-delinquency indicators that are part of the
     feature set and are reliable, interpretable risk signals.

Priority bands:
  NORMAL, FOLLOW_UP, HIGH_PRIORITY, CRITICAL_PRIORITY

This is a decision-support prototype, not an automated collections decision.
"""
from __future__ import annotations

from typing import Any

from src.risk.risk_engine import risk_level

PRIORITY_LEVELS = ("NORMAL", "FOLLOW_UP", "HIGH_PRIORITY", "CRITICAL_PRIORITY")

# Minimum probability to elevate out of the baseline bands (tuned defaults).
BASE_PRIORITY_CUTOFF = 0.45
# Probability above which priority is always CRITICAL regardless of flags.
CRITICAL_PROBABILITY = 0.70

ACTIONS: dict[str, str] = {
    "NORMAL": "No action required; continue standard monitoring.",
    "FOLLOW_UP": "Schedule a standard account-review follow-up.",
    "HIGH_PRIORITY": "Priority follow-up; escalate to a recovery specialist.",
    "CRITICAL_PRIORITY": "Immediate escalation — urgent recovery intervention.",
}


def recovery_priority(
    probability: float,
    financial_stress: int | bool = 0,
    serious_delinquency: int | bool = 0,
    any_delinquency: int | bool = 0,
    *,
    thresholds: tuple | None = None,
) -> str:
    """Compute a recovery priority for a borrower.

    Parameters
    ----------
    probability : predicted default probability in [0, 1].
    financial_stress : 1 if the financial-stress indicator is set.
    serious_delinquency : 1 if a 90+ day delinquency was recorded.
    any_delinquency : 1 if any past-due event was recorded.
    thresholds : optional 3 risk thresholds (passed through to risk_level).

    Returns
    -------
    One of PRIORITY_LEVELS.
    """
    level = risk_level(probability, thresholds)

    # Hard escalation on severe risk signals.
    if probability >= CRITICAL_PROBABILITY or (
        serious_delinquency and probability >= BASE_PRIORITY_CUTOFF
    ):
        return "CRITICAL_PRIORITY"

    if level in ("HIGH", "CRITICAL"):
        return "HIGH_PRIORITY"

    if probability >= BASE_PRIORITY_CUTOFF or (
        (financial_stress or serious_delinquency) and probability >= 0.30
    ):
        return "FOLLOW_UP"

    if any_delinquency and probability >= 0.25:
        return "FOLLOW_UP"

    return "NORMAL"


def recommended_action(priority: str) -> str:
    """Return the human-readable recommended action for a priority."""
    return ACTIONS.get(priority, ACTIONS["NORMAL"])


def priority_from_borrower(probability: float, features: dict[str, Any]) -> str:
    """Convenience: compute priority from a feature dict (API use)."""
    return recovery_priority(
        probability,
        financial_stress=int(features.get("financial_stress_indicator", 0) or 0),
        serious_delinquency=int(
            features.get("serious_delinquency_flag", 0) or 0
        ),
        any_delinquency=int(features.get("any_delinquency_flag", 0) or 0),
    )
