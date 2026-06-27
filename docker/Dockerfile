FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ ./app/
COPY configs/ ./configs/
COPY data/reference/ ./data/reference/
COPY scripts/ ./scripts/

RUN mkdir -p models data/monitoring reports/drift

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV USE_MLFLOW=true
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_MODEL_NAME=yandex_maps_sentiment
ENV MLFLOW_MODEL_STAGE=Production
ENV MODEL_PATH=/app/models/model.pkl
ENV REFERENCE_DATA_PATH=/app/data/reference/reference_sample.csv
ENV DRIFT_REPORTS_DIR=/app/reports/drift
ENV PREDICTIONS_DB_PATH=/app/data/monitoring/predictions.db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
