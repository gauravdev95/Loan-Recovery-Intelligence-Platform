"""Run the whole reproducible pipeline end-to-end.

    prepare (download-if-missing -> validate -> clean -> engineer)
        -> train (compare/tune/select/serialize)
        -> evaluate (honest test evaluation + SHAP sample)

Usage
-----
    python scripts/run_pipeline.py            # full run with tuning
    python scripts/run_pipeline.py --no-tune  # skip hyperparameter tuning
    python scripts/run_pipeline.py --no-train # only prepare data
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _run(script: str, extra: list[str] | None = None) -> int:
    print(f"\n========== RUN {script} ==========")
    cmd = [sys.executable, str(ROOT / "scripts" / script)] + (extra or [])
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    rc = _run("download_data.py")
    if rc != 0:
        print("Dataset unavailable; cannot continue (no fake data will be used).")
        return rc

    rc = _run("prepare_data.py")
    if rc != 0:
        return rc

    if args.no_train:
        return 0

    train_extra = ["--no-tune"] if args.no_tune else []
    rc = _run("train.py", train_extra)
    if rc != 0:
        return rc

    rc = _run("evaluate.py")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
