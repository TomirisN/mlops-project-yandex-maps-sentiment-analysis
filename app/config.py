import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
    
    # MLflow настройки
    USE_MLFLOW = os.getenv("USE_MLFLOW", "True").lower() == "true"
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "yandex_maps_sentiment")
    MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")
    
    API_TITLE = "Sentiment Analysis API"
    API_DESCRIPTION = "Анализ тональности отзывов Яндекс Карт"
    API_VERSION = "1.0.0"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))

config = Config()