"""Create tables / seed helpers on startup.

We use ``create_all`` for simplicity at this scale; production could migrate
to Alembic. The backend container waits for PostgreSQL (Alembic not required).
"""
from __future__ import annotations

import logging

from backend.app.database.db import engine
from backend.app.database.models import Base

logger = logging.getLogger(__name__)


def init_db() -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured.")
