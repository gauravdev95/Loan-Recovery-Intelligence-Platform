"""Lightweight, dependency-free experiment tracking.

Every model run is appended as a JSON record to
``artifacts/metrics/experiments.jsonl`` so the full history is reproducible
and auditable without standing up MLflow. Fields follow the spec:

    experiment_id, model_name, parameters, features_version, metrics, timestamp
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from src.config import FEATURES_VERSION, METRICS_DIR


def _default_path() -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    return METRICS_DIR / "experiments.jsonl"


def log_experiment(
    model_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    split: str = "validation",
    path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one experiment record to the JSONL log. Returns the record."""
    record: dict[str, Any] = {
        "experiment_id": uuid.uuid4().hex[:12],
        "model_name": model_name,
        "parameters": parameters,
        "features_version": FEATURES_VERSION,
        "metrics": metrics,
        "split": split,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        record.update(extra)

    path = Path(path) if path else _default_path()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def load_experiments(path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(path) if path else _default_path()
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
