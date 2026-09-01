"""FastAPI application entrypoint.

Run (dev):          uvicorn backend.app.main:app --reload
Run (prod/container): uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.database.init_db import init_db
from backend.app.ml.loader import ModelNotLoadedError, model_service
from backend.app.routes import borrowers, explain, health, model, predict

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    try:
        model_service.load()
    except ModelNotLoadedError as exc:
        logger.warning("Starting without model artifacts: %s", exc)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    description=(
        "ML-driven borrower default-risk prediction, explainability and "
        "recovery prioritization (educational/prototype decision support)."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root ping + route registration.
app.include_router(health.router)
app.include_router(model.router)
app.include_router(predict.router)
app.include_router(borrowers.router)
app.include_router(explain.router)


@app.get("/")
def root() -> dict:
    return {
        "service": settings.app_name,
        "version": settings.api_version,
        "docs": "/docs",
    }


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})
