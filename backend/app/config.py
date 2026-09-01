"""Backend configuration — reads environment variables with safe defaults."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the repo root (containing the `src` package) is importable whether we
# run uvicorn from the repo root or from backend/.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class Settings:
    """Simple env-driven settings (no external cfg dependency at startup)."""

    def __init__(self) -> None:
        self.app_name = os.environ.get("APP_NAME", "Loan Recovery Intelligence API")
        self.api_version = os.environ.get("API_VERSION", "1.0.0")

        # Database: prefer DATABASE_URL, else a SQLite fallback for local dev
        # and tests (production uses PostgreSQL via Docker Compose).
        self.database_url = os.environ.get(
            "DATABASE_URL",
            os.environ.get("SQLITE_URL", "sqlite:///./loan_recovery_local.db"),
        )

        # CORS origins, comma-separated.
        raw_origins = os.environ.get("CORS_ORIGINS", "")
        self.cors_origins = (
            [o.strip() for o in raw_origins.split(",") if o.strip()]
            if raw_origins
            else ["http://localhost:5173", "http://localhost:3000"]
        )

        base = Path(os.environ.get("MODEL_BASE_DIR", str(_ROOT)))
        self.model_path = Path(os.environ.get("MODEL_PATH", base / "models/best_model.joblib"))
        self.preprocessor_path = Path(
            os.environ.get("PREPROCESSOR_PATH", base / "models/preprocessing.joblib")
        )
        self.feature_metadata_path = Path(
            os.environ.get("FEATURE_METADATA_PATH", base / "models/feature_metadata.json")
        )
        self.model_metadata_path = Path(
            os.environ.get("MODEL_METADATA_PATH", base / "models/model_metadata.json")
        )

        self.log_level = os.environ.get("LOG_LEVEL", "INFO")


settings = Settings()
