# Руководство: локальный запуск и защита проекта

Пошаговая инструкция для **идеальной** локальной демонстрации MLOps-проекта.

---

## Что вы показываете на защите

| Требование курса | Где показать |
|------------------|--------------|
| Датасет + модель | `data/raw/`, MLflow Models v1–v4 |
| Git + DVC | `dvc pull`, `.dvc/config`, MinIO `:9001` |
| Cookiecutter | `cookiecutter cookiecutter-template/` |
| MLflow | http://127.0.0.1:5000 |
| CI/CD | GitHub Actions → зелёный workflow |
| FastAPI + Docker | http://127.0.0.1:8000/docs |
| Drift (3 типа) | UI Dashboard → «Проверить дрейф» |
| Evidently отчёты | `reports/drift/*.html` |
| Prometheus + Grafana | `:9090`, `:3000` |
| Web UI (6 элементов) | `/ui/*` — инференс, таблица, drift, retrain, эксперименты, уведомления |
| Kubernetes | `kubectl get pods -n mlops-sentiment` |
| Argo CD | `scripts/install_argocd.ps1` → UI sync |

---

## Шаг 0. Первичная настройка (один раз)

```powershell
cd C:\my_work_repo\mlops-project-yandex-maps-sentiment-analysis
.\scripts\setup_local.ps1
```

Скрипт:
- создаёт venv и ставит зависимости;
- переносит `model.pkl` → `models/model.pkl`;
- скачивает данные через DVC (если MinIO доступен);
- создаёт reference sample для drift;
- запускает `pytest`.

**Если нет датасета:** поднимите MinIO и выполните `dvc pull`:

```powershell
cd docker
docker compose up -d minio minio-init
cd ..
dvc pull
```

**Если нет mlruns/:** обучите модель (шаг 2) или скопируйте папку `mlruns` от коллеги.

---

## Шаг 1. Обучение модели и MLflow Registry

### Терминал 1 — MLflow Server

```powershell
.\geo_reviews_venv\Scripts\Activate.ps1
python -m mlflow server --host 127.0.0.1 --port 5000
```

### Терминал 2 — обучение

```powershell
.\geo_reviews_venv\Scripts\Activate.ps1
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python src\core\train_mlflow.py
```

Результат:
- новая версия в Registry `yandex_maps_sentiment`;
- автоматически переводится в **Production**;
- `models/model.pkl` обновлён.

Откройте: http://127.0.0.1:5000 → **Models** → `yandex_maps_sentiment`.

---

## Шаг 2. Локальный API без Docker

```powershell
.\geo_reviews_venv\Scripts\Activate.ps1
$env:USE_MLFLOW="true"
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

| URL | Назначение |
|-----|------------|
| http://127.0.0.1:8000/ui | Главный дашборд |
| http://127.0.0.1:8000/ui/inference | Инференс |
| http://127.0.0.1:8000/ui/predictions | Таблица предсказаний |
| http://127.0.0.1:8000/ui/experiments | Эксперименты MLflow |
| http://127.0.0.1:8000/docs | Swagger |
| http://127.0.0.1:8000/metrics | Prometheus-метрики |

### Проверка predict

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/predict `
  -H "Content-Type: application/json" `
  -d '{"text": "Отличное место, всё понравилось!"}'
```

---

## Шаг 3. Полный стек Docker Compose (рекомендуется для защиты)

```powershell
cd docker
docker compose build
docker compose up -d
docker compose ps
```

Все сервисы должны быть **healthy** / **running**.

| Сервис | URL | Логин |
|--------|-----|-------|
| Web UI | http://127.0.0.1:8000/ui | — |
| MLflow | http://127.0.0.1:5000 | — |
| Prometheus | http://127.0.0.1:9090 | — |
| Grafana | http://127.0.0.1:3000 | admin / admin |
| MinIO | http://127.0.0.1:9001 | minioadmin / minioadmin |

### Сценарий демо в UI (10 минут)

1. **Инференс** — введите 10+ отзывов (разная тональность).
2. **Predictions** — убедитесь, что записи появились в таблице.
3. **Dashboard** → «Проверить дрейф» — data/target/concept drift + HTML-отчёт.
4. **Dashboard** → «Запустить переобучение» — дождитесь `success`, модель перезагрузится автоматически.
5. **Experiments** — список run'ов из MLflow.
6. **Grafana** — дашборд «MLOps Sentiment Monitoring» (нужны predict'ы).
7. **MLflow** — Models → новая версия в Production.

Остановка:

```powershell
docker compose down
```

---

## Шаг 4. Тесты и линтер (как в CI)

```powershell
.\geo_reviews_venv\Scripts\Activate.ps1
pip install -r requirements-ci.txt
pytest tests/ -v
flake8 app/ tests/
black --check app/ tests/
```

---

## Шаг 5. Cookiecutter

```powershell
pip install cookiecutter
cookiecutter cookiecutter-template/ --no-input
```

Покажите сгенерированный проект — шаблон для воспроизводимого старта MLOps-проекта.

---

## Шаг 6. Kubernetes (Minikube)

```powershell
# Добавьте minikube в PATH (если нужно)
$env:PATH = "$env:USERPROFILE\.bin;$env:PATH"

.\scripts\deploy_minikube.ps1
kubectl get pods -n mlops-sentiment
```

Ожидаемый вывод — 4 pod'а в статусе **Running**:
- `sentiment-api`
- `mlflow`
- `prometheus`
- `grafana`

### URL в Minikube

```powershell
minikube ip
# UI: http://<IP>:30080/ui
# Grafana: http://<IP>:30300
# Prometheus: http://<IP>:30090
```

Или:

```powershell
minikube service sentiment-api -n mlops-sentiment --url
```

---

## Шаг 7. Argo CD (GitOps)

```powershell
.\scripts\install_argocd.ps1
```

В другом терминале:

```powershell
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

Откройте https://localhost:8080 → Application `mlops-sentiment` → **Sync**.

---

## Шаг 8. DVC + MinIO

```powershell
dvc status
dvc pull
dvc repro
```

MinIO Console: http://127.0.0.1:9001 — bucket `dvc-store-mlops-project`.

---

## Чеклист перед защитой

```
[ ] setup_local.ps1 прошёл без ошибок
[ ] pytest — все тесты зелёные
[ ] docker compose up — все сервисы healthy
[ ] MLflow: модели v1–v4, Production актуален
[ ] UI: predict → drift → retrain → success
[ ] Grafana: графики после predict'ов
[ ] kubectl get pods — всё Running
[ ] GitHub Actions — CI зелёный
[ ] Cookiecutter — генерация работает
[ ] Argo CD — application synced (опционально)
```

---

## Частые проблемы

| Проблема | Решение |
|----------|---------|
| MLflow пустой в Docker | `docker compose build mlflow --no-cache && docker compose up -d mlflow` |
| Grafana «No data» | Сделайте 5–10 predict'ов, подождите 15 сек |
| Drift «Норма» при 0 predict | Нужно ≥10 предсказаний (`DRIFT_MIN_SAMPLES=10`) |
| Retrain failed: dataset not found | `dvc pull`, проверьте `data/raw/geo-reviews-dataset-2023.csv` |
| `minikube` не найден | `$env:PATH = "$env:USERPROFILE\.bin;$env:PATH"` |
| Experiments пустые | MLflow должен быть доступен по HTTP (`http://mlflow:5000` в Docker) |

---

## Порядок выступления (рекомендация)

1. Архитектура (схема из README).
2. `docker compose ps` — весь стек.
3. UI: predict → drift → retrain.
4. MLflow Models.
5. Grafana + Prometheus.
6. `pytest` + GitHub Actions.
7. `kubectl get pods` + Argo CD.
8. Cookiecutter + DVC.

Удачи на защите!
