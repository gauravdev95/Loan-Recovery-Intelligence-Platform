"""Risk engine tests (threshold boundaries and categories)."""
from __future__ import annotations

import pytest

from src.risk.risk_engine import DEFAULT_THRESHOLDS, RISK_LEVELS, risk_level


@pytest.mark.parametrize(
    "prob,expected",
    [
        (0.0, "LOW"),
        (0.29, "LOW"),
        (0.30, "MEDIUM"),
        (0.59, "MEDIUM"),
        (0.60, "HIGH"),
        (0.79, "HIGH"),
        (0.80, "CRITICAL"),
        (1.0, "CRITICAL"),
    ],
)
def test_default_boundaries(prob, expected):
    assert risk_level(prob) == expected


def test_custom_thresholds():
    thr = (0.2, 0.5, 0.7)
    assert risk_level(0.1, thr) == "LOW"
    assert risk_level(0.4, thr) == "MEDIUM"
    assert risk_level(0.6, thr) == "HIGH"
    assert risk_level(0.9, thr) == "CRITICAL"


def test_validate_raises():
    with pytest.raises(ValueError):
        risk_level(1.5, validate=True)
    with pytest.raises(ValueError):
        risk_level(-0.1, validate=True)


def test_default_thresholds_shape():
    assert len(DEFAULT_THRESHOLDS) == 3
    assert list(DEFAULT_THRESHOLDS) == sorted(DEFAULT_THRESHOLDS)


def test_all_levels_exposed():
    assert RISK_LEVELS == ("LOW", "MEDIUM", "HIGH", "CRITICAL")
