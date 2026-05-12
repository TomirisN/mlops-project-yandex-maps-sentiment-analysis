from pydantic import BaseModel, Field, field_validator
from app.sentiment_enums import SentimentEnum

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Текст отзыва не может быть пустым')
        return v.strip()

class SentimentResponse(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    confidence: float = Field(..., ge=0, le=1)
    sentiment: SentimentEnum

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool