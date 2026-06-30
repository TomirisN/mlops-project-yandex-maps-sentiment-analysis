# mlops-project-yandex-maps-sentiment-analysis

MLOps-проект для анализа тональности отзывов **Яндекс Карт** (магистратура ИТМО).

**Датасет:** [Yandex Geo Reviews Dataset 2023](https://www.kaggle.com/datasets/kyakovlev/yandex-geo-reviews-dataset-2023)

**Задача:** по тексту отзыва предсказать рейтинг (1–5) и тональность (`negative` / `neutral` / `positive`).

---

## Структура проекта

```
├── app/                          # FastAPI-сервис
│   ├── main.py                   # Точка входа, /metrics, lifespan
│   ├── api/
│   │   ├── routes.py             # /predict, /health
│   │   ├── monitoring_routes.py  # drift, predictions, retrain, experiments
│   │   └── ui_routes.py          # Веб UI (/ui/*)
│   ├── monitoring/
│   │   ├── drift.py              # Data/target/concept drift + Evidently
│   │   ├── metrics.py            # Prometheus-метрики
│   │   ├── prediction_store.py   # SQLite-хранилище предсказаний
│   │   └── retrain_service.py    # Фоновое переобучение
│   ├── templates/                # Jinja2-шаблоны UI
│   └── static/                   # CSS/JS
├── src/core/
│   ├── train_fast.py             # TF-IDF + LR (локально)
│   ├── train_mlflow.py           # Обучение + MLflow Registry
│   └── train.py                  # BERT (RuBERT)
├── data/
│   ├── raw/                      # Датасет (DVC)
│   └── reference/                # Reference sample для drift
├── reports/drift/                # HTML-отчёты Evidently
├── docker/
│   ├── docker-compose.yml        # MinIO + MLflow + API + Prometheus + Grafana
│   ├── prometheus.yml
│   └── grafana/                  # Dashboards и provisioning
├── k8s/                          # Kubernetes + Argo CD
├── scripts/init_reference_data.py
├── tests/
└── .github/workflows/ci-cd.yml
```

---

## Быстрый старт (локально)

**Автоматическая настройка (рекомендуется):**

```powershell
.\scripts\setup_local.ps1
```

Подробное руководство для защиты: **[docs/DEMO.md](docs/DEMO.md)**

### 1. Окружение (вручную)

```powershell
python -m venv geo_reviews_venv
.\geo_reviews_venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Данные

```powershell
dvc pull
python scripts\init_reference_data.py
```

### 3. Обучение

```powershell
python src\core\train_mlflow.py
```

### 4. MLflow (терминал 1)

```powershell
python -m mlflow server --host 127.0.0.1 --port 5000
```

### 5. API + UI (терминал 2)

```powershell
$env:USE_MLFLOW="true"
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```


| URL                                  | Назначение                    |
| ------------------------------------ | --------------------------------------- |
| http://127.0.0.1:8000/ui             | Веб-дашборд                   |
| http://127.0.0.1:8000/ui/inference   | Страница инференса     |
| http://127.0.0.1:8000/ui/predictions | Таблица предсказаний |
| http://127.0.0.1:8000/ui/experiments | Эксперименты MLflow         |
| http://127.0.0.1:8000/docs           | Swagger API                             |
| http://127.0.0.1:8000/metrics        | Prometheus-метрики               |

---

## Docker (полный стек)

Поднимает **MinIO**, **MLflow**, **FastAPI**, **Prometheus**, **Grafana**.

```powershell
cd docker
docker compose build
docker compose up -d
docker compose ps
```


| Сервис    | URL                        | Логин                  |
| --------------- | -------------------------- | --------------------------- |
| Web UI          | http://127.0.0.1:8000/ui   | —                          |
| FastAPI Swagger | http://127.0.0.1:8000/docs | —                          |
| MLflow          | http://127.0.0.1:5000      | —                          |
| Prometheus      | http://127.0.0.1:9090      | —                          |
| Grafana         | http://127.0.0.1:3000      | `admin` / `admin`           |
| MinIO Console   | http://127.0.0.1:9001      | `minioadmin` / `minioadmin` |

Остановка: `docker compose down`

---

## Мониторинг дрейфа

Система отслеживает три типа дрейфа:


| Тип            | Что сравнивается                                                                                |
| ----------------- | -------------------------------------------------------------------------------------------------------------- |
| **Data drift**    | Длина текста, число слов (KS-тест)                                                     |
| **Target drift**  | Распределение предсказанных рейтингов vs reference                          |
| **Concept drift** | Сдвиг accuracy (если есть`true_rating`) или средней уверенности модели |

### API

```powershell
# Проверить дрейф
curl -X POST http://127.0.0.1:8000/api/v1/drift/check

# Статус последней проверки
curl http://127.0.0.1:8000/api/v1/drift/status

# Список HTML-отчётов Evidently
curl http://127.0.0.1:8000/api/v1/drift/reports
```

Отчёты сохраняются в `reports/drift/drift_report_*.html`.

В веб UI:

- баннер с уведомлениями о дрейфе на всех страницах;
- кнопка **«Проверить дрейф»** на дашборде;
- ссылка на последний Evidently-отчёт.

---

## Prometheus + Grafana

Метрики на `/metrics`:

- `sentiment_predictions_total` — число предсказаний по тональности
- `sentiment_prediction_confidence` — гистограмма уверенности
- `sentiment_drift_detected` / `sentiment_drift_score` — дрейф
- `sentiment_anomalies_total` — аномалии (низкая уверенность)
- `sentiment_retrain_status` — статус переобучения

В Grafana импортирован дашборд **MLOps Sentiment Monitoring** (provisioning из `docker/grafana/`).

---

## Веб UI


| Страница  | Функции                                                                                                                               |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `/ui`             | Дашборд: drift-метрики, уведомления, кнопки «Проверить дрейф» и «Переобучение» |
| `/ui/inference`   | Форма инференса, опциональный`true_rating` для concept drift                                                    |
| `/ui/predictions` | Таблица последних предсказаний с флагами аномалий                                                |
| `/ui/experiments` | Список MLflow-экспериментов                                                                                               |

### Переобучение

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/retrain
curl http://127.0.0.1:8000/api/v1/retrain/status
```

Запускает `src/core/train_mlflow.py` в фоне. После успеха модель автоматически регистрируется в MLflow (**Production**) и перезагружается в API без рестарта контейнера.

---

## Kubernetes + Argo CD

### Minikube (ручной деплой)

```powershell
minikube start

# Собрать образы в Minikube
minikube docker-env | Invoke-Expression
docker build -f docker/Dockerfile -t mlops-api:latest .
docker build -f docker/Dockerfile.mlflow -t mlops-mlflow:latest docker/

# Применить манифесты
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/prometheus-deployment.yaml
kubectl apply -f k8s/grafana-deployment.yaml
```

Доступ к сервисам:

```powershell
minikube service sentiment-api -n mlops-sentiment --url
minikube service grafana -n mlops-sentiment --url
```

### Argo CD (GitOps)

1. Установите Argo CD в кластер.
2. Замените `repoURL` в `k8s/argocd-application.yaml` на URL вашего репозитория.
3. Примените Application:

```powershell
kubectl apply -f k8s/argocd-application.yaml
```

Argo CD будет автоматически синхронизировать `k8s/` при изменениях в `main`.

---

## Model Registry — версии моделей

Для лабораторной в MLflow должно быть **несколько версий** одной модели — чтобы сравнивать метрики и переводить лучшую в Production.

Сейчас в реестре `yandex_maps_sentiment`:


| Версия | Стадия   | Accuracy  | Описание                                     |
| ------------ | -------------- | --------- | ---------------------------------------------------- |
| **v1**       | Staging        | 0.568     | Базовая: TF-IDF 5k, unigrams                  |
| **v2**       | **Production** | **0.575** | Улучшенная: TF-IDF 10k, bi-grams           |
| **v3**       | Staging        | 0.562     | Эксперимент с 15k features (хуже v2) |

**Где смотреть в MLflow:** http://127.0.0.1:5000 → **Models** → `yandex_maps_sentiment` → вкладки Versions / Stages.

**Где на диске:** `mlruns/models/yandex_maps_sentiment/version-N/` + артефакты в `mlruns/<experiment_id>/<run_id>/artifacts/sentiment_model`.

**Локальная копия лучшей модели:** `models/model.pkl` (сейчас = v2).

### Добавить новые версии

```powershell
python src\core\train_model_versions.py --configs improved-v2 best-v3
# или все конфиги:
python src\core\train_model_versions.py --configs all
```

Список версий в терминале:

```powershell
python scripts\list_model_versions.py
```

После смены Production перезапустите API:

```powershell
# Docker
cd docker; docker compose restart api

# Локально — Ctrl+C и снова uvicorn
```

---


| Метод | Путь               | Описание                                |
| ---------- | ---------------------- | ----------------------------------------------- |
| POST       | `/api/v1/predict`      | Предсказание тональности |
| GET        | `/api/v1/health`       | Статус модели                       |
| GET        | `/api/v1/predictions`  | Последние предсказания     |
| POST       | `/api/v1/drift/check`  | Запуск проверки дрейфа      |
| GET        | `/api/v1/drift/status` | Статус дрейфа                       |
| POST       | `/api/v1/retrain`      | Запуск переобучения           |
| GET        | `/api/v1/experiments`  | Список MLflow-экспериментов  |
| GET        | `/metrics`             | Prometheus-метрики                       |

### POST /api/v1/predict

```json
{
  "text": "Отличное место, всем рекомендую!",
  "true_rating": 5
}
```

Ответ:

```json
{
  "rating": 5,
  "confidence": 0.73,
  "sentiment": "positive",
  "prediction_id": 42,
  "is_anomaly": false
}
```

---

## Переменные окружения


| Переменная           | По умолчанию               | Описание                                           |
| ------------------------------ | ------------------------------------- | ---------------------------------------------------------- |
| `USE_MLFLOW`                   | `true`                                | Загрузка модели из MLflow                  |
| `MLFLOW_TRACKING_URI`          | `http://localhost:5000`               | Адрес MLflow                                          |
| `MLFLOW_MODEL_NAME`            | `yandex_maps_sentiment`               | Имя модели                                        |
| `MLFLOW_MODEL_STAGE`           | `Production`                          | Стадия модели                                  |
| `REFERENCE_DATA_PATH`          | `data/reference/reference_sample.csv` | Reference для drift                                     |
| `DRIFT_MIN_SAMPLES`            | `30`                                  | Мин. предсказаний для drift              |
| `ANOMALY_CONFIDENCE_THRESHOLD` | `0.4`                                 | Порог аномалии                                |
| `PREDICTIONS_DB_PATH`          | `data/monitoring/predictions.db`      | SQLite БД                                                |
| `AUTO_RETRAIN_ON_DRIFT`        | `true`                                | Автозапуск переобучения при drift |

---

## Git Flow и Conventional Commits

См. [docs/GITFLOW.md](docs/GITFLOW.md) и [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

- **main** — production, деплой в K8s / GHCR
- **develop** — интеграция фич
- **feature/\*** → PR в develop → PR в main
- Коммиты: `feat:`, `fix:`, `docs:`, `ci:` и т.д.

---

## DVC pipeline

```powershell
dvc pull
dvc repro          # prepare_reference + train
dvc dag            # граф pipeline
```

Файлы: `dvc.yaml`, `params.yaml`.

---

## Cookiecutter

```powershell
pip install cookiecutter
cookiecutter cookiecutter-template/
```

Создаёт новый проект из шаблона. См. [cookiecutter-template/README.md](cookiecutter-template/README.md).

---

## CI/CD

GitHub Actions (`.github/workflows/ci-cd.yml`):


| Job         | Когда      | Что делает                                                |
| ----------- | --------------- | ------------------------------------------------------------------ |
| **test**    | push/PR         | pytest                                                             |
| **lint**    | push/PR         | flake8 + black (`app/`, `tests/`)                                  |
| **docker**  | push/PR         | сборка образа API                                      |
| **publish** | push в**main** | push образов API + MLflow в GHCR                           |
| **deploy**  | push в**main** | `kubectl apply -f k8s/` (если задан secret `KUBE_CONFIG`) |

---

## Kubernetes / Minikube / Argo CD

**Docker Compose** — отладка локально. **Kubernetes** — целевой production.

```powershell
# Minikube (полный деплой одной командой)
.\scripts\deploy_minikube.ps1
```

Подробнее: [k8s/README.md](k8s/README.md)

**Argo CD:** `kubectl apply -f k8s/argocd-application.yaml` (замените `YOUR_ORG` в repoURL).

---

## Архитектура

```
┌──────────┐   dvc pull    ┌──────────┐
│  MinIO   │ ◄───────────► │   CSV    │
└──────────┘               └────┬─────┘
                              │ train_mlflow.py
┌──────────┐   load model ┌───▼──────┐   metrics   ┌────────────┐
│ FastAPI  │ ◄────────── │  MLflow  │ ──────────► │ Prometheus │
│  :8000   │             └──────────┘             └─────┬──────┘
│  /ui     │                                            │
└────┬─────┘                                            ▼
     │ drift reports                              ┌──────────┐
     ▼                                            │ Grafana  │
┌──────────┐                                      │  :3000   │
│ Evidently│                                      └──────────┘
│ reports/ │
└──────────┘
```

---

## Чек-лист лабораторной (все пункты)


| № | Требование              | Статус | Где                                |
| -- | --------------------------------- | ------------ | ------------------------------------- |
| 1  | Датасет + модель     | ✅           | `data/raw/`, `src/core/`              |
| 2  | Git + conventional commits + DVC  | ✅           | `docs/GITFLOW.md`, `dvc.yaml`         |
| 3  | Cookiecutter                      | ✅           | `cookiecutter-template/`              |
| 4  | MLflow                            | ✅           | `:5000`, `mlruns/`                    |
| 5  | CI/CD (lint, test, build, deploy) | ✅           | `.github/workflows/ci-cd.yml`         |
| 6  | FastAPI + Docker + Compose        | ✅           | `app/`, `docker/`                     |
| 7  | Drift + Prometheus + Grafana      | ✅           | `app/monitoring/`, `:9090`, `:3000`   |
| 8  | Evidently-отчёты            | ✅           | `reports/drift/`                      |
| 9  | Web UI                            | ✅           | `/ui/*`                               |
| 10 | Argo CD + Kubernetes              | ✅           | `k8s/`, `scripts/deploy_minikube.ps1` |
| 11 | README                            | ✅           | этот файл                     |

**MLOps-цикл:** данные (DVC) → predict (UI) → drift → автопереобучение (`AUTO_RETRAIN_ON_DRIFT`) → MLflow → restart API → Grafana.
