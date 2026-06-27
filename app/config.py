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

    # Мониторинг
    PREDICTIONS_DB_PATH = os.getenv(
        "PREDICTIONS_DB_PATH",
        str(BASE_DIR / "data" / "monitoring" / "predictions.db"),
    )
    REFERENCE_DATA_PATH = os.getenv(
        "REFERENCE_DATA_PATH",
        str(BASE_DIR / "data" / "reference" / "reference_sample.csv"),
    )
    DRIFT_REPORTS_DIR = os.getenv(
        "DRIFT_REPORTS_DIR",
        str(BASE_DIR / "reports" / "drift"),
    )
    DRIFT_DATA_THRESHOLD = float(os.getenv("DRIFT_DATA_THRESHOLD", "0.5"))
    DRIFT_TARGET_THRESHOLD = float(os.getenv("DRIFT_TARGET_THRESHOLD", "0.5"))
    DRIFT_CONCEPT_THRESHOLD = float(os.getenv("DRIFT_CONCEPT_THRESHOLD", "0.15"))
    DRIFT_MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "30"))
    ANOMALY_CONFIDENCE_THRESHOLD = float(
        os.getenv("ANOMALY_CONFIDENCE_THRESHOLD", "0.4")
    )

    TRAIN_SCRIPT_PATH = os.getenv(
        "TRAIN_SCRIPT_PATH",
        str(BASE_DIR / "src" / "core" / "train_mlflow.py"),
    )
    MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", MLFLOW_TRACKING_URI)
    AUTO_RETRAIN_ON_DRIFT = os.getenv("AUTO_RETRAIN_ON_DRIFT", "true").lower() == "true"


config = Config()
