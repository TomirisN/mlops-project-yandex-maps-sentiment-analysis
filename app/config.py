import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = "models/model.pkl"
    API_TITLE = "Sentiment Analysis API"
    API_DESCRIPTION = "Анализ тональности отзывов Яндекс Карт (оценки 1-5)"
    API_VERSION = "1.0.0"
    HOST = "0.0.0.0"
    PORT = 8000

config = Config()