"""Тесты drift detection и monitoring API."""

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("USE_MLFLOW", "false")
os.environ.setdefault("DRIFT_MIN_SAMPLES", "5")

from app.monitoring.drift import DriftDetector
from app.monitoring.prediction_store import PredictionStore


@pytest.fixture
def temp_monitoring_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "reference.csv"
        ref_path.write_text(
            "text,rating,confidence\n"
            '"Отлично всем рекомендую",5,0.9\n'
            '"Ужасно плохо",1,0.8\n'
            '"Нормально",3,0.6\n'
            '"Хорошо",4,0.7\n'
            '"Супер",5,0.85\n',
            encoding="utf-8",
        )
        reports_dir = Path(tmp) / "reports"
        db_path = Path(tmp) / "predictions.db"
        yield ref_path, reports_dir, db_path


def test_drift_detector_insufficient_data(temp_monitoring_dirs):
    ref_path, reports_dir, db_path = temp_monitoring_dirs
    store = PredictionStore(str(db_path))
    detector = DriftDetector(
        reference_path=str(ref_path),
        reports_dir=str(reports_dir),
        min_samples=10,
    )

    status = detector.check(store)
    assert status.any_drift_detected is False
    assert "Недостаточно данных" in status.data_drift.details


def test_drift_detector_with_predictions(temp_monitoring_dirs):
    ref_path, reports_dir, db_path = temp_monitoring_dirs
    store = PredictionStore(str(db_path))
    detector = DriftDetector(
        reference_path=str(ref_path),
        reports_dir=str(reports_dir),
        min_samples=5,
        data_threshold=0.99,
        target_threshold=0.99,
        concept_threshold=0.99,
    )

    for i in range(6):
        store.add(
            text=f"Тестовый отзыв номер {i} с достаточным количеством слов",
            rating=3,
            confidence=0.7,
            sentiment="neutral",
        )

    status = detector.check(store)
    assert status.checked_at
    assert status.data_drift.drift_type == "data"


def test_drift_check_endpoint(client: TestClient):
    for i in range(6):
        client.post(
            "/api/v1/predict",
            json={"text": f"Отзыв для drift теста номер {i}, всё хорошо"},
        )

    response = client.post("/api/v1/drift/check")
    assert response.status_code == 200
    data = response.json()
    assert "data_drift" in data
    assert "target_drift" in data
    assert "concept_drift" in data


def test_predictions_list_endpoint(client: TestClient):
    client.post("/api/v1/predict", json={"text": "Проверка списка предсказаний"})
    response = client.get("/api/v1/predictions?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert len(data["items"]) >= 1


def test_metrics_endpoint(client: TestClient):
    client.post("/api/v1/predict", json={"text": "Метрики prometheus тест"})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "sentiment_predictions_total" in response.text


def test_retrain_endpoint(client: TestClient):
    response = client.post("/api/v1/retrain")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"running", "idle"}


def test_ui_pages(client: TestClient):
    for path in ("/ui", "/ui/inference", "/ui/predictions", "/ui/experiments"):
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
