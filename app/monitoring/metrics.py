"""Prometheus-метрики для inference и мониторинга дрейфа."""

from prometheus_client import Counter, Gauge, Histogram, Info

PREDICTIONS_TOTAL = Counter(
    "sentiment_predictions_total",
    "Общее число предсказаний",
    ["sentiment"],
)

PREDICTION_CONFIDENCE = Histogram(
    "sentiment_prediction_confidence",
    "Распределение уверенности модели",
    buckets=(0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

PREDICTION_RATING = Counter(
    "sentiment_prediction_rating_total",
    "Распределение предсказанных рейтингов",
    ["rating"],
)

ANOMALIES_TOTAL = Counter(
    "sentiment_anomalies_total",
    "Число предсказаний с флагом аномалии",
)

DRIFT_DETECTED = Gauge(
    "sentiment_drift_detected",
    "Флаг обнаружения дрейфа (1=да, 0=нет)",
    ["drift_type"],
)

DRIFT_SCORE = Gauge(
    "sentiment_drift_score",
    "Оценка дрейфа по типу (0-1)",
    ["drift_type"],
)

MODEL_INFO = Info(
    "sentiment_model",
    "Информация о загруженной модели",
)

RETRAIN_STATUS = Gauge(
    "sentiment_retrain_status",
    "Статус переобучения: 0=idle, 1=running, 2=success, 3=failed",
)

PREDICTION_LATENCY = Histogram(
    "sentiment_prediction_latency_seconds",
    "Латентность предсказания в секундах",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


def record_prediction(
    rating: int, confidence: float, sentiment: str, is_anomaly: bool
) -> None:
    PREDICTIONS_TOTAL.labels(sentiment=sentiment).inc()
    PREDICTION_RATING.labels(rating=str(rating)).inc()
    PREDICTION_CONFIDENCE.observe(confidence)
    if is_anomaly:
        ANOMALIES_TOTAL.inc()


def set_drift_metrics(drift_type: str, detected: bool, score: float) -> None:
    DRIFT_DETECTED.labels(drift_type=drift_type).set(1 if detected else 0)
    DRIFT_SCORE.labels(drift_type=drift_type).set(score)


def set_model_info(loaded: bool, source: str) -> None:
    MODEL_INFO.info({"loaded": str(loaded).lower(), "source": source})


def set_retrain_status(status_code: int) -> None:
    RETRAIN_STATUS.set(status_code)
