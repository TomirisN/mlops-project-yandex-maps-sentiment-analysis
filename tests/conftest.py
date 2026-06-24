"""Общие фикстуры для pytest.

Тесты не требуют запущенного MLflow, MinIO или файла models/model.pkl:
подставляется маленькая обученная модель в памяти.
"""
import os

# Важно: переменные окружения до импорта app, чтобы lifespan не ходил в MLflow.
os.environ.setdefault("USE_MLFLOW", "false")
os.environ.setdefault("MODEL_PATH", "models/model.pkl")

import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.core.model_manager import ModelManager
from app.dependency import get_model_manager
from app.main import app


def _build_tiny_model() -> Pipeline:
    """Минимальная модель для тестов — обучается за доли секунды."""
    texts = [
        "ужасно плохо отвратительно",
        "всё плохо разочарование",
        "нормально средне",
        "хорошо понравилось",
        "отлично супер рекомендую",
    ]
    labels = [0, 1, 2, 3, 4]  # рейтинги 1–5 в формате обучения (0–4)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline


@pytest.fixture
def loaded_model_manager() -> ModelManager:
    manager = ModelManager(model_path="models/model.pkl")
    manager.model = _build_tiny_model()
    manager._is_loaded = True
    return manager


@pytest.fixture
def client(loaded_model_manager: ModelManager) -> TestClient:
    """HTTP-клиент FastAPI с подменённой загруженной моделью."""
    app.dependency_overrides[get_model_manager] = lambda: loaded_model_manager
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
