# {{ cookiecutter.project_name }}

{{ cookiecutter.description }}

## Быстрый старт

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- API docs: http://127.0.0.1:8000/docs
- Metrics: http://127.0.0.1:8000/metrics

## MLOps

```bash
dvc init
dvc repro
docker compose -f docker/docker-compose.yml up -d
```

MLflow model name: `{{ cookiecutter.mlflow_model_name }}`

## Тесты

```bash
pytest tests/ -v
```
