"""SHAP explanation endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.ml.loader import ModelNotLoadedError, model_service
from backend.app.schemas import (
    ExplanationFactor,
    ExplanationResponse,
    PredictionRequest,
)

router = APIRouter(prefix="/api/v1", tags=["explain"])


@router.post("/explain", response_model=ExplanationResponse)
def explain(payload: PredictionRequest) -> ExplanationResponse:
    """Explain a prediction for a raw borrower feature set."""
    try:
        result = model_service.explain(payload.to_raw_dict(), top_k=5)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ExplanationResponse(
        borrower_id=result.get("borrower_id"),
        default_probability=result["default_probability"],
        risk_level=result["risk_level"],
        recovery_priority=result["recovery_priority"],
        expected_value=result.get("expected_value"),
        top_factors=[ExplanationFactor(**f) for f in result["top_factors"]],
    )


@router.post("/explain/{borrower_id}")
def explain_by_id(
    borrower_id: str, payload: PredictionRequest
) -> ExplanationResponse:
    """Explain for a borrower id (uses the payload features)."""
    return explain(
        PredictionRequest(**payload.model_dump(), borrower_id=borrower_id)
    )
