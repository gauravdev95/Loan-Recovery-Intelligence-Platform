"""Prediction endpoints (single + batch)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.database.db import get_db
from backend.app.ml.loader import ModelNotLoadedError, model_service
from backend.app.schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from backend.app.services.prediction_service import predict_and_store

router = APIRouter(prefix="/api/v1", tags=["predict"])


def _ensure_loaded():
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest, db: Session = Depends(get_db)
) -> PredictionResponse:
    _ensure_loaded()
    try:
        return predict_and_store(
            service=model_service,
            db=db,
            borrower_id=payload.borrower_id,
            raw=payload.to_raw_dict(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    payload: list[BatchPredictionItem], db: Session = Depends(get_db)
) -> BatchPredictionResponse:
    _ensure_loaded()
    results = []
    for item in payload:
        try:
            result = predict_and_store(
                service=model_service,
                db=db,
                borrower_id=item.borrower_id,
                raw=item.request.to_raw_dict(),
            )
            results.append(result)
        except (ValueError, ModelNotLoadedError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BatchPredictionResponse(predictions=results, n=len(results))
