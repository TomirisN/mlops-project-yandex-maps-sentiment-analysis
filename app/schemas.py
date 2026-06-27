from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.sentiment_enums import SentimentEnum


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    true_rating: Optional[int] = Field(None, ge=1, le=5)

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Текст отзыва не может быть пустым")
        return v.strip()


class SentimentResponse(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    confidence: float = Field(..., ge=0, le=1)
    sentiment: SentimentEnum
    prediction_id: Optional[int] = None
    is_anomaly: bool = False


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class DriftItemResponse(BaseModel):
    drift_type: str
    detected: bool
    score: float
    details: str


class DriftStatusResponse(BaseModel):
    checked_at: str
    data_drift: DriftItemResponse
    target_drift: DriftItemResponse
    concept_drift: DriftItemResponse
    any_drift_detected: bool
    report_path: Optional[str] = None
    notifications: List[str] = []


class PredictionItemResponse(BaseModel):
    id: int
    text: str
    rating: int
    confidence: float
    sentiment: str
    created_at: str
    is_anomaly: bool
    true_rating: Optional[int] = None


class PredictionsListResponse(BaseModel):
    total: int
    items: List[PredictionItemResponse]


class RetrainResponse(BaseModel):
    status: str
    message: str
    started_at: Optional[str] = None


class ExperimentRunResponse(BaseModel):
    run_id: str
    run_name: str
    status: str
    accuracy: Optional[float] = None
    start_time: Optional[int] = None


class ExperimentsResponse(BaseModel):
    experiment_name: str
    runs: List[ExperimentRunResponse]
