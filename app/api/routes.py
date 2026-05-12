from fastapi import APIRouter, Depends
from app.schemas import ReviewRequest, SentimentResponse, HealthResponse
from app.dependency import get_model_manager
from app.core.model_manager import ModelManager

router = APIRouter()

@router.get("/")
async def root():
    return {"service": "Sentiment API", "version": "1.0.0"}

@router.get("/health", response_model=HealthResponse)
async def health(manager: ModelManager = Depends(get_model_manager)):
    return HealthResponse(
        status="healthy" if manager.is_loaded else "degraded",
        model_loaded=manager.is_loaded
    )

@router.post("/predict", response_model=SentimentResponse)
async def predict(review: ReviewRequest, manager: ModelManager = Depends(get_model_manager)):
    rating, confidence = manager.predict(review.text)
    sentiment = manager.get_sentiment_label(rating)
    return SentimentResponse(rating=rating, confidence=confidence, sentiment=sentiment)