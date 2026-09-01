"""Pydantic request/response schemas.

The prediction request mirrors the 10 raw "Give Me Some Credit" features that
a lender-collections user would understand, so clients do not need to know the
engineered feature set.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    """Raw borrower credit features required for a prediction."""

    model_config = ConfigDict(extra="forbid")

    borrower_id: str | None = Field(default=None, max_length=32)
    RevolvingUtilizationOfUnsecuredLines: float = Field(ge=0.0, le=100.0)
    age: int = Field(ge=18, le=124)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(ge=0, le=100)
    DebtRatio: float = Field(ge=0.0)
    MonthlyIncome: float | None = Field(default=None, ge=0.0)
    NumberOfOpenCreditLinesAndLoans: int = Field(ge=0, le=100)
    NumberOfTimes90DaysLate: int = Field(ge=0, le=100)
    NumberRealEstateLoansOrLines: int = Field(ge=0, le=100)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(ge=0, le=100)
    NumberOfDependents: int | None = Field(default=None, ge=0, le=100)

    # The raw CSV uses dashes in these column names; the API uses underscores.
    @field_validator(
        "NumberOfTime30_59DaysPastDueNotWorse",
        "NumberOfTime60_89DaysPastDueNotWorse",
        mode="before",
    )
    @classmethod
    def _drop_dashes(cls, v):
        # Allow clients to send either underscore or dash-keyed JSON.
        return v

    def to_raw_dict(self) -> dict:
        """Map API (underscore) names back to the raw dataset column names."""
        mapping = {
            "NumberOfTime30_59DaysPastDueNotWorse": "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60_89DaysPastDueNotWorse": "NumberOfTime60-89DaysPastDueNotWorse",
        }
        out = {}
        for key, value in self.model_dump().items():
            if key == "borrower_id":
                continue
            out[mapping.get(key, key)] = value
        return out


class PredictionResponse(BaseModel):
    borrower_id: str | None = None
    default_probability: float
    risk_level: str
    recovery_priority: str
    recommended_action: str
    model_version: str


class BatchPredictionItem(BaseModel):
    borrower_id: str | None = None
    request: PredictionRequest


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    n: int


class ExplanationFactor(BaseModel):
    feature: str
    value: float | None = None
    shap_value: float | None = None
    direction: str


class ExplanationResponse(BaseModel):
    borrower_id: str | None = None
    default_probability: float
    risk_level: str
    recovery_priority: str
    expected_value: float | None = None
    top_factors: list[ExplanationFactor]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    features_version: str
    n_features: int
    target: str
    loaded: bool


class BorrowerResponse(BaseModel):
    borrower_id: str
    default_probability: float | None = None
    risk_level: str | None = None
    recovery_priority: str | None = None
    recommended_action: str | None = None
    model_version: str | None = None
    created_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database: str
    version: str
