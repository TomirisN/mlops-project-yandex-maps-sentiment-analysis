from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
import numpy as np

app = FastAPI(
    title="Sentiment Analysis API",
    description="Анализ тональности отзывов Яндекс Карт",
    version="1.0.0"
)

# Загрузка модели
model = None
MODEL_PATH = "models/model.pkl"

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Модель загружена из {MODEL_PATH}")
    else:
        print(f"❌ Модель не найдена")

class ReviewRequest(BaseModel):
    text: str = Field(..., example="Отличное место, очень понравилось!")

class SentimentResponse(BaseModel):
    rating: int
    confidence: float
    sentiment: str

@app.get("/")
async def root():
    return {"message": "API работает", "model_loaded": model is not None}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=SentimentResponse)
async def predict(review: ReviewRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    pred = model.predict([review.text])[0] + 1
    proba = model.predict_proba([review.text])[0]
    confidence = float(np.max(proba))
    
    if pred <= 2:
        sentiment = "negative"
    elif pred == 3:
        sentiment = "neutral"
    else:
        sentiment = "positive"
    
    return SentimentResponse(rating=int(pred), confidence=confidence, sentiment=sentiment)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)