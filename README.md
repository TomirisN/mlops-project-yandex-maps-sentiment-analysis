# mlops-project-yandex-maps-sentiment-analysis

MLOps-проект для анализа тональности отзывов **Яндекс Карт** (магистратура ИТМО).

**Датасет:** [Yandex Geo Reviews Dataset 2023](https://www.kaggle.com/datasets/kyakovlev/yandex-geo-reviews-dataset-2023)

**Задача:** по тексту отзыва предсказать рейтинг (1–5) и тональность (`negative` / `neutral` / `positive`).

---

## Структура проекта

```
├── app/                    # FastAPI-сервис (inference)
│   ├── main.py
│   ├── api/routes.py
│   └── core/model_manager.py
├── src/core/
│   ├── train_fast.py       # обучение TF-IDF + LR (локально)
│   └── train_mlflow.py     # обучение + логирование в MLflow
├── data/raw/               # датасет (DVC, не в git)
├── models/                 # model.pkl (локальный fallback)
├── mlruns/                 # эксперименты MLflow
├── docker/
│   ├── Dockerfile          # образ API
│   ├── Dockerfile.mlflow   # образ MLflow server
│   └── docker-compose.yml  # API + MLflow + MinIO
├── configs/config.yaml
├── requirements.txt        # полные зависимости (обучение)
├── requirements-api.txt    # зависимости для Docker API
└── .dvc/                   # DVC remote → MinIO
```

---

## Быстрый старт (локально, без Docker)

### 1. Окружение и зависимости

```powershell
python -m venv geo_reviews_venv
.\geo_reviews_venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Данные (DVC + MinIO)

```powershell
# MinIO должен быть запущен (см. раздел Docker)
dvc pull
```

Файл появится в `data/raw/geo-reviews-dataset-2023.csv`.

### 3. Обучение

```powershell
python src\core\train_fast.py
# или с MLflow:
python src\core\train_mlflow.py
```

### 4. MLflow (терминал 1)

```powershell
python -m mlflow server --host 127.0.0.1 --port 5000
```

UI: http://127.0.0.1:5000

После `train_mlflow.py` переведите модель в **Production**:

```powershell
python -c "from mlflow.tracking import MlflowClient; c=MlflowClient('http://127.0.0.1:5000'); c.transition_model_version_stage(name='yandex_maps_sentiment', version=1, stage='Production'); print('OK')"
```

### 5. FastAPI (терминал 2)

```powershell
$env:USE_MLFLOW="true"
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```


| URL                                  | Назначение      |
| -------------------------------------- | --------------------------- |
| http://127.0.0.1:8000/docs           | Swagger UI                |
| http://127.0.0.1:8000/api/v1/health  | Статус модели |
| http://127.0.0.1:8000/api/v1/predict | Предсказание  |

> **Ошибка `[WinError 10048]` на порту 8000** — порт занят. Остановите другой uvicorn или Docker-контейнер `sentiment-api`: `docker stop sentiment-api`

---

## Docker (полный стек)

Поднимает **MinIO** (DVC), **MLflow** (трекинг моделей), **FastAPI** (inference).

### Предварительные условия

1. Установлен [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Локально есть папка `mlruns/` с зарегистрированной моделью `yandex_maps_sentiment` в стадии **Production**
3. Порты **8000**, **5000**, **9000**, **9001** свободны (остановите локальные `uvicorn` / `mlflow server`)
4. Если был отдельный контейнер `minio`, остановите его: `docker stop minio && docker rm minio`
   Compose использует volume `minio-data` с вашими DVC-данными

### Запуск

```powershell
cd docker

# Сборка образов
docker compose build

# Запуск всех сервисов
docker compose up -d

# Статус
docker compose ps

# Логи
docker compose logs -f api
```

### Проверка


| Сервис      | URL                                 | Логин                  |
| ------------------- | ------------------------------------- | ----------------------------- |
| FastAPI (Swagger) | http://127.0.0.1:8000/docs          | —                          |
| Health            | http://127.0.0.1:8000/api/v1/health | `model_loaded: true`        |
| MLflow UI         | http://127.0.0.1:5000               | —                          |
| MinIO Console     | http://127.0.0.1:9001               | `minioadmin` / `minioadmin` |

### DVC с MinIO в Docker

```powershell
# credentials в .dvc/config.local
dvc remote modify minio endpointurl http://127.0.0.1:9000 --local
dvc pull
```

Бакет `dvc-store-mlops-project` создаётся автоматически сервисом `minio-init`.

### Остановка

```powershell
docker compose down
# с удалением volumes (осторожно — удалит данные MinIO):
docker compose down -v
```

### Переобучение с Docker-стеком

```powershell
# MLflow в compose уже на :5000
python src\core\train_mlflow.py
# Перевести новую версию в Production (UI или Python)
docker compose restart api
```

---

## Архитектура Docker

```
┌─────────────┐     dvc pull/push     ┌─────────────┐
│    MinIO    │ ◄──────────────────► │  CSV (DVC)  │
│  :9000      │                       └──────┬──────┘
└─────────────┘                              │
                                             ▼
                                      train_mlflow.py
                                             │
┌─────────────┐     load model      ┌────────▼──────┐
│   FastAPI   │ ◄────────────────── │    MLflow     │
│   :8000     │                     │    :5000      │
└─────────────┘                     │  (mlruns/)    │
                                    └───────────────┘
```

- **MinIO** — хранилище данных для DVC
- **MLflow** — эксперименты и Model Registry
- **FastAPI** — HTTP API для предсказаний (`USE_MLFLOW=true`)

---

## API

### POST /api/v1/predict

```json
{
  "text": "Отличное место, всем рекомендую!"
}
```

Ответ:

```json
{
  "rating": 5,
  "confidence": 0.73,
  "sentiment": "positive"
}
```

### GET /api/v1/health

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## Переменные окружения (API)


| Переменная  | По умолчанию | Описание                                   |
| ----------------------- | ------------------------- | ---------------------------------------------------- |
| `USE_MLFLOW`          | `true`                  | Загружать модель из MLflow        |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | Адрес MLflow (в Docker:`http://mlflow:5000`) |
| `MLFLOW_MODEL_NAME`   | `yandex_maps_sentiment` | Имя модели в реестре              |
| `MLFLOW_MODEL_STAGE`  | `Production`            | Стадия модели                          |
| `MODEL_PATH`          | `models/model.pkl`      | Fallback, если MLflow недоступен     |

---

## Что ещё в плане (курс MLOps)

- [ ] CI/CD (GitHub Actions: lint, test, build, deploy)
- [ ] Drift-мониторинг (Evidently)
- [ ] Prometheus + Grafana
- [ ] Веб UI
- [ ] Kubernetes + Argo CD
