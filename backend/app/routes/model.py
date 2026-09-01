"""Model information and metrics endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.ml.loader import ModelNotLoadedError, model_service
from backend.app.schemas import ModelInfoResponse

router = APIRouter(prefix="/api/v1/model", tags=["model"])


@router.get("/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")
    info = model_service.model_info()
    return ModelInfoResponse(**info)


@router.get("/metrics")
def model_metrics():
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return model_service.model_metrics()
