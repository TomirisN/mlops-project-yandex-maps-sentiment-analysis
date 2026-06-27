# {{ cookiecutter.project_name }}

{{ cookiecutter.description }}

## Быстрый старт

```bash
python -m venv venv
pip install -r requirements.txt
dvc pull
dvc repro
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## MLOps

- MLflow model: `{{ cookiecutter.mlflow_model_name }}`
- Docker registry: `{{ cookiecutter.docker_registry }}`

Автор: {{ cookiecutter.author_name }}
