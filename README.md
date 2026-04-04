# mlops-project-yandex-maps-sentiment-analysis

Проект для курса MLOps в ходе обучения в магистратуре ИТМО.
По данным Яндекс Карт.

Ссылка на датасет: https://www.kaggle.com/datasets/kyakovlev/yandex-geo-reviews-dataset-2023

# Структура проекта

## Структура проекта

- `cookiecutter-template/`
  - `cookiecutter.json`
  - `{{cookiecutter.project_slug}}/`
    - `.github/workflows/`
      - `ci-cd.yml` – CI/CD (линтеры, тесты, сборка, деплой)
    - `docker/`
      - `Dockerfile` – для API-сервиса
      - `docker-compose.yml` – MLflow, Prometheus, Grafana, БД
    - `k8s/`
      - `deployment.yaml`
      - `service.yaml`
      - `ingress.yaml`
      - `argocd-application.yaml` – для Argo CD
    - `src/`
      - `api/`
        - `__init__.py`
        - `app.py` – FastAPI + UI (инференс, таблица, кнопка, эксперименты)
        - `routers/` – (можно всё в одном файле, если не много роутов)
      - `core/`
        - `inference.py` – логика предсказания
        - `train.py` – обучение (вызывается по кнопке или CI)
        - `drift.py` – расчёт дрейфа
      - `monitoring/`
        - `metrics.py` – метрики для Prometheus
        - `reports/` – папка для сгенерированных отчётов
    - `data/` – (DVC)
      - `raw/`
      - `processed/`
    - `tests/`
      - `test_api.py`
      - `test_drift.py`
    - `.gitignore`
    - `.dvcignore`
    - `requirements.txt`
    - `requirements-dev.txt`
    - `dvc.yaml` – (опционально)
    - `Makefile`
    - `README.md`