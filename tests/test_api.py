"""Тесты REST API sentiment analysis."""


def test_health_returns_200_and_model_loaded(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["status"] == "healthy"


def test_predict_with_text_returns_rating_1_to_5(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": "Отличное место, всё понравилось, обязательно вернусь!"},
    )

    assert response.status_code == 200
    data = response.json()
    assert 1 <= data["rating"] <= 5
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["sentiment"] in {"negative", "neutral", "positive"}


def test_predict_empty_text_returns_422(client):
    response = client.post("/api/v1/predict", json={"text": ""})
    print(response.status_code, response.json())  # pytest tests/ -s
    assert response.status_code == 422

    assert response.status_code == 422
