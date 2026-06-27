import time

from fastapi import APIRouter, Depends

from app.config import config
from app.core.model_manager import ModelManager
from app.dependency import get_model_manager, get_prediction_store
from app.monitoring.metrics import PREDICTION_LATENCY, record_prediction
from app.monitoring.prediction_store import PredictionStore
from app.schemas import HealthResponse, ReviewRequest, SentimentResponse

router = APIRouter()


@router.get("/")
async def root():
    return {"service": "Sentiment API", "version": "1.0.0"}


@router.get("/health", response_model=HealthResponse)
async def health(manager: ModelManager = Depends(get_model_manager)):
    return HealthResponse(
        status="healthy" if manager.is_loaded else "degraded",
        model_loaded=manager.is_loaded,
    )


@router.post("/predict", response_model=SentimentResponse)
async def predict(
    review: ReviewRequest,
    manager: ModelManager = Depends(get_model_manager),
    store: PredictionStore = Depends(get_prediction_store),
):
    start = time.perf_counter()
    rating, confidence = manager.predict(review.text)
    sentiment = manager.get_sentiment_label(rating)
    is_anomaly = confidence < config.ANOMALY_CONFIDENCE_THRESHOLD

    prediction_id = store.add(
        text=review.text,
        rating=rating,
        confidence=confidence,
        sentiment=sentiment,
        is_anomaly=is_anomaly,
        true_rating=review.true_rating,
    )
    record_prediction(rating, confidence, sentiment, is_anomaly)
    PREDICTION_LATENCY.observe(time.perf_counter() - start)

    return SentimentResponse(
        rating=rating,
        confidence=confidence,
        sentiment=sentiment,
        prediction_id=prediction_id,
        is_anomaly=is_anomaly,
    )
