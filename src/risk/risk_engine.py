"""Risk engine: maps a predicted default probability to a risk category.

The default thresholds below are a starting point (as per the brief). They are
**finetuned on the validation split** by the training pipeline, which writes
the chosen thresholds into the model metadata. The engine always uses the
thresholds passed to it (defaulting to the documented candidate bands).
"""
from __future__ import annotations

from typing import Sequence

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# Candidate starting thresholds (probability band edges), overridden by tuning.
DEFAULT_THRESHOLDS: Sequence[float] = (0.30, 0.60, 0.80)

# Thresholds tuned on validation data; filled in by the training pipeline.
TUNED_THRESHOLDS: Sequence[float] | None = None


def risk_level(
    probability: float,
    thresholds: Sequence[float] | None = None,
    validate: bool = False,
) -> str:
    """Map a probability to a risk level.

    Parameters
    ----------
    probability : default probability in [0, 1].
    thresholds : 3 ascending boundaries (low/med/high/critical). Defaults to
        TUNED_THRESHOLDS if set, else DEFAULT_THRESHOLDS.
    validate : if True, raise on out-of-range probability (for the API).
    """
    if validate and not (0.0 <= probability <= 1.0):
        raise ValueError(f"probability must be in [0,1], got {probability}")

    if thresholds is None:
        thresholds = TUNED_THRESHOLDS or DEFAULT_THRESHOLDS
    t_low, t_med, t_high = thresholds

    if probability < t_low:
        return "LOW"
    if probability < t_med:
        return "MEDIUM"
    if probability < t_high:
        return "HIGH"
    return "CRITICAL"
