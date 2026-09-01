"""Dataset loading helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CSV_USE_COLS, RAW_DATA_PATH


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Give-Me-Some-Credit CSV, keeping only expected columns.

    The source CSV has a leading unnamed row-id column (empty header). We
    read only the columns we care about so the dataframe starts clean.
    """
    path = Path(path) if path else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    df = pd.read_csv(path)
    keep = [c for c in df.columns if c in CSV_USE_COLS]
    return df[keep].copy()


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    """Save a dataframe to CSV, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
