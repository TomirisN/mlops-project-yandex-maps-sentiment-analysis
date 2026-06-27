# Contributing

## Git Flow

См. [GITFLOW.md](GITFLOW.md): ветки `main` / `develop` / `feature/*`.

## Conventional Commits

Все коммиты — в формате Conventional Commits:

```
feat(scope): описание на русском или английском
fix(api): исправлена загрузка модели
docs: обновлён README
ci: deploy в GHCR при push в main
```

## Разработка

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-ci.txt
pytest tests/ -v
flake8 app/ tests/
black --check app/ tests/
```

## Pull Request

1. Создайте ветку от `develop`
2. Убедитесь, что CI зелёный (lint + test + docker build)
3. PR в `develop`; после ревью — merge в `main` для деплоя

## DVC

```powershell
dvc pull
dvc repro
```
