FROM python:3.11-slim

WORKDIR /app

# Системные зависимости (для scikit-learn / scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Код приложения
COPY app/ ./app/
COPY configs/ ./configs/

# Папка для локального fallback модели
RUN mkdir -p models

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# По умолчанию — через MLflow (переопределяется в compose)
ENV USE_MLFLOW=true
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_MODEL_NAME=yandex_maps_sentiment
ENV MLFLOW_MODEL_STAGE=Production
ENV MODEL_PATH=/app/models/model.pkl

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]