"""Experiment-tracking (MLOps) logger tests."""
from __future__ import annotations

import json

from src.evaluation.experiments import load_experiments, log_experiment


def test_log_and_load(tmp_path):
    path = tmp_path / "experiments.jsonl"
    rec = log_experiment(
        "XGBoost", {"n_estimators": 100}, {"f1": 0.5}, split="validation", path=path
    )
    assert rec["experiment_id"]
    assert rec["model_name"] == "XGBoost"
    assert rec["features_version"]
    assert rec["timestamp"]

    rows = load_experiments(path)
    assert len(rows) == 1
    assert rows[0]["metrics"]["f1"] == 0.5
    assert json.loads(path.read_text())


def test_missing_file_returns_empty(tmp_path):
    assert load_experiments(tmp_path / "nope.jsonl") == []
