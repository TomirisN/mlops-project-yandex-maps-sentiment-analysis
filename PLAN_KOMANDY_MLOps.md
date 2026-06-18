# План работы команды MLOps-проекта
## Yandex Maps Sentiment Analysis

**Команда:** Томирис Намозова + Полина Домбровская  
**Дедлайн:** 27 июня 2026  
**Дата составления:** 18 июня 2026  

---

## Оглавление

1. [О чём этот проект простыми словами](#1-о-чём-этот-проект-простыми-словами)
2. [Что уже сделано](#2-что-уже-сделано)
3. [Что осталось сделать](#3-что-осталось-сделать)
4. [Какие программы и сервисы мы используем](#4-какие-программы-и-сервисы-мы-используем)
5. [Как запустить проект сейчас (напоминание)](#5-как-запустить-проект-сейчас-напоминание)
6. [Подробный план по дням](#6-подробный-план-по-дням)
7. [Задачи Томирис — подробно](#7-задачи-томирис--подробно)
8. [Задачи Полины — подробно](#8-задачи-полины--подробно)
9. [Как работать с Git вдвоём](#9-как-работать-с-git-вдвоём)
10. [Сценарий для защиты (демо)](#10-сценарий-для-защиты-демо)
11. [Если что-то не успеваем](#11-если-что-то-не-успеваем)
12. [Чеклист перед сдачей 27.06](#12-чеклист-перед-сдачей-2706)

---

## 1. О чём этот проект простыми словами

Мы делаем **систему**, которая:

1. Берёт **отзывы с Яндекс Карт** (текст + оценка 1–5).
2. **Обучает модель**, которая по тексту предсказывает рейтинг и тональность (негатив / нейтрал / позитив).
3. Отдаёт предсказания через **веб-сервис (API)** — как «калькулятор тональности».
4. **Следит**, не ухудшилась ли модель со временем (**дрейф данных**).
5. Позволяет **переобучить** модель и **обновить** её в сервисе.
6. Показывает всё это в **веб-интерфейсе** и **графиках мониторинга**.

Это и есть **MLOps** — не просто «обучили модель в ноутбуке», а **весь цикл в production**.

### Схема «как всё связано»

```
┌──────────┐    dvc pull     ┌─────────────┐
│  MinIO   │ ──────────────► │  CSV-файл   │  ← данные
└──────────┘                 └──────┬──────┘
                                    │
                             train_mlflow.py
                                    │
                                    ▼
┌──────────┐   загрузка модели  ┌──────────┐
│ FastAPI  │ ◄───────────────── │  MLflow  │  ← эксперименты и модели
│  :8000   │                    │  :5000   │
└────┬─────┘                    └──────────┘
     │
     ├──► Веб UI (страницы в браузере)
     ├──► Prometheus (метрики) ──► Grafana (графики)
     └──► Drift (Evidently) ──► отчёты + флаги «аномалия»
```

---

## 2. Что уже сделано

| Что | Статус | Где в проекте |
|-----|--------|---------------|
| Датасет выбран | ✅ | Kaggle, DVC, MinIO |
| Модель TF-IDF + Logistic Regression | ✅ | `train_fast.py`, `train_mlflow.py` |
| FastAPI + Swagger | ✅ | папка `app/`, http://localhost:8000/docs |
| MLflow (трекинг, реестр моделей) | ✅ | `train_mlflow.py`, Docker `mlops-mlflow` |
| Docker Compose (API + MLflow + MinIO) | ✅ | `docker/docker-compose.yml` |
| README с инструкцией запуска | ✅ | `README.md` |
| DVC remote на MinIO | ✅ | `.dvc/config` |

**Итого:** база работает. Можно обучить модель и получить предсказание через API.

---

## 3. Что осталось сделать

| Что требует курс | Статус | Приоритет |
|------------------|--------|-----------|
| CI/CD (линтер, тесты, сборка Docker) | ❌ пустой файл | 🔴 Высокий |
| Тесты (`pytest`) | ❌ пустые файлы | 🔴 Высокий |
| Drift (data / target / concept) | ❌ пустой `drift.py` | 🔴 Высокий |
| Prometheus + Grafana | ❌ не подключены | 🔴 Высокий |
| Отчёты о дрейфе (HTML) | ❌ | 🔴 Высокий |
| Веб UI (6 страниц) | ❌ пустой `src/api/app.py` | 🔴 Высокий |
| Kubernetes + Argo CD | ❌ пустые yaml | 🟡 Средний |
| Cookiecutter шаблон | ❌ только json | 🟢 Низкий |
| Git flow + conventional commits | ⚠️ описано, не настроено | 🟡 Средний |
| `dvc.yaml` (пайплайн) | ❌ | 🟡 Средний |

---

## 4. Какие программы и сервисы мы используем

### Уже используем

| Сервис | Что это | Зачем нам | Где открыть |
|--------|---------|-----------|-------------|
| **Git / GitHub** | Хранение кода | Версии, работа вдвоём, CI | github.com |
| **DVC** | Версионирование данных | Большой CSV не в git | команда `dvc pull` |
| **MinIO** | Файловое хранилище (как S3) | Хранит датасет для DVC | http://localhost:9001 |
| **MLflow** | Журнал экспериментов | Метрики, версии моделей | http://localhost:5000 |
| **FastAPI** | Python веб-сервис | API для предсказаний | http://localhost:8000/docs |
| **Docker** | Контейнеры | Одинаковый запуск у всех | Docker Desktop |
| **Docker Compose** | Несколько контейнеров сразу | MinIO + MLflow + API | `docker compose up` |

### Нужно добавить

| Сервис | Что это простыми словами | Зачем | Как будет выглядеть |
|--------|--------------------------|-------|---------------------|
| **GitHub Actions** | Робот в GitHub | При push/MR запускает линтер, тесты, сборку Docker | Вкладка Actions в репозитории, зелёная галочка |
| **pytest** | Тесты Python | Проверяют, что API не сломался | `pytest` в терминале → все тесты passed |
| **flake8 / black** | Проверка стиля кода | Линтер в CI | CI не падает на этапе lint |
| **Evidently** | Библиотека для drift | Сравнивает «старые» и «новые» данные | HTML-отчёт в папке `reports/` |
| **Prometheus** | Сборщик метрик | Считает: сколько predict, latency, drift | Endpoint `/metrics` на API |
| **Grafana** | Графики | Красивые дашборды | http://localhost:3000 |
| **Kubernetes (Minikube)** | Оркестратор контейнеров | Запуск в «мини-облаке» на ноутбуке | `kubectl get pods` |
| **Argo CD** | CD в Kubernetes | Автодеплой при изменении в git | UI Argo CD, статус Synced |

---

## 5. Как запустить проект сейчас (напоминание)

### Шаг 1. Docker

```powershell
cd docker
docker compose up -d
docker compose ps
```

**Ожидаемый результат:** контейнеры `mlops-api`, `mlops-mlflow`, `mlops-minio` в статусе Up/healthy.

### Шаг 2. Данные

```powershell
dvc pull
```

**Ожидаемый результат:** файл `data/raw/geo-reviews-dataset-2023.csv` появился на диске.

### Шаг 3. Обучение (если модели ещё нет)

```powershell
pip install mlflow==2.15.1 querystring_parser "setuptools<81"
python src\core\train_mlflow.py
```

Перевести модель в Production (через UI http://localhost:5000 или Python).

### Шаг 4. Проверка API

- http://localhost:8000/api/v1/health → `"model_loaded": true`
- http://localhost:8000/docs → POST predict работает

### ⚠️ Важно для Полины

Версия MLflow **клиента** и **сервера** должна совпадать: **2.15.1**.  
Иначе `train_mlflow.py` падает на `log_model`.

---

## 6. Подробный план по дням

| Дата | День | Томирис | Полина | Результат дня |
|------|------|---------|--------|---------------|
| 18–19.06 | 1 | Git flow, ветка develop, начало CI | Проверка train + Docker у себя | Оба поднимают проект одинаково |
| 20.06 | 2 | `test_api.py`, черновик `ci-cd.yml` | Начало `drift.py` | API тестируется автоматически |
| 21.06 | 3 | CI: lint + test + docker build | `dvc.yaml`, `test_drift.py` | Зелёный CI на GitHub |
| 22.06 | 4 | API: `/drift/status`, хранение predict | Drift + HTML-отчёты | Drift считается, отчёт на диске |
| 23.06 | 5 | Prometheus `/metrics`, Grafana в compose | Связать drift → метрика | Графики в Grafana |
| 24.06 | 6 | `POST /retrain`, роуты для UI | UI: инференс + таблица predict | Страницы в браузере |
| 25.06 | 7 | UI: подключение роутов в main.py | UI: drift, retrain, эксперименты | Все 6 страниц UI |
| 26.06 | 8 | K8s yaml + Argo CD | Тест в Minikube | `kubectl get pods` — Running |
| 27.06 | 9 | README финал, MR в main | Репетиция демо | Сдача |

---

## 7. Задачи Томирис — подробно

### Задача T1. CI/CD (GitHub Actions)

**О чём речь:** при каждом Push или Merge Request GitHub сам запускает проверки. Не нужно вручную помнить «а мы тесты запускали?».

**Что сделать:**
- Заполнить файл `.github/workflows/ci-cd.yml`
- Этапы pipeline:
  1. **lint** — `flake8 app/ src/` и `black --check`
  2. **test** — `pytest tests/`
  3. **build** — `docker compose build` в папке docker
  4. **deploy** (при merge в main) — опционально push образа в GitHub Container Registry

**Какими сервисами пользоваться:**
- GitHub (репозиторий)
- GitHub Actions (встроено, бесплатно для учебных проектов)

**Ожидаемый результат:**
- Открываете GitHub → вкладка **Actions** → видите зелёный workflow ✅
- При плохом коде — красный крест ❌, merge заблокирован

**Как проверить:**
```powershell
git push origin feature/ci-tests
# Смотреть Actions на GitHub
```

---

### Задача T2. Тесты API (`tests/test_api.py`)

**О чём речь:** маленькие программы, которые автоматически проверяют API без браузера.

**Что написать (минимум):**
1. `test_health` — GET `/api/v1/health` возвращает 200 и `model_loaded`
2. `test_predict_positive` — POST predict с хорошим отзывом → rating 1–5
3. `test_predict_empty_text` — пустой текст → ошибка 422

**Какими сервисами пользоваться:**
- `pytest` (библиотека Python)
- `httpx` или `TestClient` из FastAPI

**Ожидаемый результат:**
```powershell
pytest tests/test_api.py -v
# ===== 3 passed =====
```

---

### Задача T3. Prometheus + Grafana

**О чём речь:** 
- **Prometheus** — «счётчик», который раз в N секунд спрашивает API «сколько запросов было?»
- **Grafana** — рисует графики из данных Prometheus

**Что сделать:**
1. Файл `src/monitoring/metrics.py` — счётчики:
   - `predictions_total` — сколько predict сделано
   - `prediction_latency_seconds` — время ответа
   - `drift_detected` — 0 или 1
2. В `app/main.py` — endpoint `GET /metrics`
3. В `docker-compose.yml` добавить сервисы `prometheus` и `grafana`
4. Файл `docker/prometheus.yml` — куда стучаться (target: api:8000)
5. Простой dashboard в Grafana

**Ожидаемый результат:**
- http://localhost:8000/metrics — текст с метриками
- http://localhost:3000 — Grafana, график predict/sec

**Как проверить:**
- Сделать 5 predict через Swagger
- В Grafana видно, что счётчик вырос

---

### Задача T4. API для UI и переобучения

**О чём речь:** веб-страницам нужны данные с бэкенда.

**Новые endpoints:**

| Метод | URL | Зачем |
|-------|-----|-------|
| GET | `/api/v1/predictions` | Таблица последних предсказаний |
| GET | `/api/v1/drift/status` | Флаги аномалий для UI |
| POST | `/api/v1/retrain` | Кнопка «переобучить» |
| GET | `/ui` | Главная страница интерфейса |

**Хранение предсказаний:** простой SQLite файл `data/predictions.db` или JSON-файл.

**Переобучение:** `POST /retrain` запускает `train_mlflow.py` в фоне (subprocess).

**Ожидаемый результат:**
- `curl http://localhost:8000/api/v1/predictions` → JSON со списком
- `curl -X POST http://localhost:8000/api/v1/retrain` → `{"status": "started"}`

---

### Задача T5. Kubernetes + Argo CD

**О чём речь:**
- **Kubernetes** — запускает контейнеры на «кластере» (Minikube = кластер на вашем ПК)
- **Argo CD** — следит за git и обновляет то, что в кластере

**Что заполнить:**
- `k8s/deployment.yaml` — описание контейнера API (образ, порты, env)
- `k8s/service.yaml` — доступ к API внутри кластера
- `k8s/ingress.yaml` — доступ снаружи (опционально)
- `k8s/argocd-application.yaml` — связь Argo CD с вашим репо

**Какими сервисами пользоваться:**
- Minikube (`minikube start`)
- kubectl (`kubectl apply -f k8s/`)
- Argo CD (ставится в Minikube)

**Ожидаемый результат:**
```powershell
kubectl get pods
# mlops-api-xxx   1/1   Running
minikube service mlops-api --url
# открывается predict
```

---

### Задача T6. Cookiecutter + README

**О чём речь:** Cookiecutter — шаблон, из которого можно «вырастить» копию проекта. Для курса достаточно минимального шаблона.

**Ожидаемый результат:**
- Папка `cookiecutter-template/{{cookiecutter.project_slug}}/` с базовыми файлами
- README описывает всё: clone → docker up → dvc pull → train → открыть UI

---

## 8. Задачи Полины — подробно

### Задача P1. Drift — дрейф данных (`src/core/drift.py`)

**О чём речь:** модель обучалась на одних данных. Если приходят **другие** отзывы (другие слова, другие оценки) — модель может ошибаться чаще. Это называется **дрейф**.

**Три вида (требование курса):**

| Вид | Что сравниваем | Пример |
|-----|----------------|--------|
| **Data drift** | Распределение входных данных (текст, длина) | Раньше отзывы короткие, теперь длинные |
| **Target drift** | Распределение оценок (1–5) | Раньше много 5★, теперь много 1★ |
| **Concept drift** | Связь текст→оценка изменилась | «Норм» раньше = 3★, теперь = 5★ |

**Какими сервисами пользоваться:**
- Библиотека **Evidently** (`pip install evidently`)
- Референсные данные: часть CSV из `data/raw/`
- Текущие данные: новая выборка или последние predict

**Что сделать:**
1. Функция `calculate_drift(reference_df, current_df)` 
2. Сохранение HTML-отчёта в `src/monitoring/reports/drift_report_YYYYMMDD.html`
3. Возврат флагов: `data_drift: true/false`, `target_drift: true/false`

**Ожидаемый результат:**
```powershell
python -c "from src.core.drift import run_drift_check; run_drift_check()"
# Создан файл src/monitoring/reports/drift_report_20260622.html
# В консоли: data_drift=True, target_drift=False
```

Открываете HTML в браузере — видите графики и таблицы Evidently.

---

### Задача P2. Тесты drift (`tests/test_drift.py`)

**О чём речь:** проверить, что drift-модуль не падает на тестовых данных.

**Минимум:**
- `test_drift_runs_without_error` — на маленькой выборке
- `test_drift_report_created` — файл отчёта появился

**Ожидаемый результат:** `pytest tests/test_drift.py` → passed

---

### Задача P3. DVC pipeline (`dvc.yaml`)

**О чём речь:** описать цепочку «данные → обучение → метрики» в одном файле.

**Пример структуры:**
```yaml
stages:
  train:
    cmd: python src/core/train_mlflow.py
    deps:
      - data/raw/geo-reviews-dataset-2023.csv
      - src/core/train_mlflow.py
    outs:
      - models/model.pkl
      - metrics.json
```

**Как пользоваться:** `dvc repro` — запускает pipeline.

**Ожидаемый результат:** одна команда вместо ручного запуска train.

---

### Задача P4. Веб UI — 6 страниц

**О чём речь:** не Swagger, а **нормальные страницы в браузере** для пользователя/преподавателя.

**Технология (простая):** HTML-шаблоны Jinja2 + немного CSS. FastAPI умеет отдавать HTML.

**6 обязательных страниц:**

| № | Страница | Что на ней | Откуда данные |
|---|----------|------------|---------------|
| 1 | **Инференс** | Поле ввода текста + кнопка «Предсказать» + результат | POST `/api/v1/predict` |
| 2 | **Таблица предсказаний** | Последние 20 predict: текст, rating, время | GET `/api/v1/predictions` |
| 3 | **Флаги аномалий** | Блок: «Data drift: ДА/НЕТ», «Target drift: ДА/НЕТ» | GET `/api/v1/drift/status` |
| 4 | **Кнопка переобучения** | Кнопка «Запустить переобучение» + статус | POST `/api/v1/retrain` |
| 5 | **Эксперименты** | Ссылка или iframe на MLflow UI | http://localhost:5000 |
| 6 | **Уведомления о дрейфе** | Красный баннер вверху, если drift=true | GET `/api/v1/drift/status` |

**Где писать код:**
- Шаблоны: `app/templates/` (создать папку)
- Роуты: `app/api/ui_routes.py` или расширить `routes.py`
- Статика (CSS): `app/static/style.css`

**Ожидаемый результат:**
- http://localhost:8000/ui — главная с навигацией
- Все 6 разделов открываются и работают
- Можно показать преподавателю без Swagger

---

### Задача P5. Демо-сценарий для защиты

**О чём речь:** 5–7 минутный сценарий, что показывать на защите.

**Пример сценария:**
1. `docker compose up` — подняли стек
2. Открыли UI → сделали predict
3. Показали таблицу предсказаний
4. Запустили drift check → показали отчёт и красный флаг
5. Нажали «переобучить» → MLflow → новая версия
6. Grafana — графики метрик
7. kubectl get pods — Kubernetes

**Ожидаемый результат:** документ `docs/DEMO.md` с пошаговым текстом.

---

## 9. Как работать с Git вдвоём

### Ветки

```
main          ← только готовый код (защита)
  ↑
develop       ← сливаем фичи друг друга
  ↑
feature/имя   ← каждая задача в своей ветке
```

### Порядок работы

1. `git checkout develop && git pull`
2. `git checkout -b feature/drift` (или `feature/ci`)
3. Делаете задачу, коммиты:
   ```
   feat(drift): add evidently data drift check
   test(drift): add basic drift tests
   ```
4. `git push origin feature/drift`
5. Создаёте **Pull Request** в `develop` на GitHub
6. Вторая человека делает **review**
7. CI зелёный → merge
8. В конце проекта: PR `develop` → `main`

### Conventional Commits — примеры

| Тип | Пример |
|-----|--------|
| feat | `feat(api): add /metrics endpoint for prometheus` |
| fix | `fix(mlflow): pin client version to 2.15.1` |
| test | `test(api): add health and predict tests` |
| ci | `ci: add github actions workflow` |
| docs | `docs: add team plan and demo script` |

---

## 10. Сценарий для защиты (демо)

### Подготовка (за 10 минут до защиты)

```powershell
cd docker
docker compose up -d
docker compose ps   # всё healthy
```

### Показ (7 минут)

| Минута | Что показать | Что сказать |
|--------|--------------|-------------|
| 0–1 | README, схема архитектуры | «Проект анализирует отзывы Яндекс Карт» |
| 1–2 | http://localhost:8000/ui — predict | «Пользователь вводит текст, получает rating» |
| 2–3 | Таблица предсказаний + MLflow | «Все predict логируются, эксперименты в MLflow» |
| 3–4 | Drift report + красный флаг | «Evidently обнаружил data drift» |
| 4–5 | Кнопка retrain → MLflow новая версия | «Переобучение и обновление модели» |
| 5–6 | Grafana | «Prometheus собирает метрики, Grafana визуализирует» |
| 6–7 | kubectl / Argo CD | «Деплой в Kubernetes через Argo CD» |

---

## 11. Если что-то не успеваем

### Обязательно к 27.06 (без этого не сдавать)

- [ ] CI с lint + test + docker build
- [ ] Drift (data + target) + HTML отчёт
- [ ] Prometheus `/metrics` + Grafana в compose
- [ ] UI: хотя бы инференс + таблица + retrain + drift-флаг
- [ ] K8s deployment (хотя бы ручной `kubectl apply`)
- [ ] README

### Можно упростить

| Задача | Упрощение |
|--------|-----------|
| Cookiecutter | Только `cookiecutter.json` + пояснение в README |
| Argo CD | Манифест Application + ручной sync на демо |
| Concept drift | Proxy: «accuracy упала на 10%» |
| BERT | Не трогать, оставить TF-IDF |
| Airflow | Не делать (не в обязательных этапах) |

---

## 12. Чеклист перед сдачей 27.06

### Технический

- [ ] `docker compose up -d` — все сервисы healthy
- [ ] `dvc pull` — данные скачиваются
- [ ] `train_mlflow.py` — обучение проходит
- [ ] http://localhost:8000/api/v1/health — model_loaded: true
- [ ] http://localhost:8000/ui — все страницы открываются
- [ ] Drift report генерируется
- [ ] Grafana показывает метрики
- [ ] `pytest` — все тесты passed
- [ ] GitHub Actions — зелёный
- [ ] `kubectl get pods` — Running (если K8s)

### Документация

- [ ] README актуален
- [ ] CONTRIBUTING с conventional commits
- [ ] План команды (этот документ) в репозитории

### Команда

- [ ] Оба клонировали репо и подняли проект с нуля
- [ ] Демо отрепетировано
- [ ] MR develop → main сделан

---

## Контакты и договорённости

| Вопрос | Решение |
|--------|---------|
| Кто мержит в main? | Обе смотрят PR, мержит автор PR после approve |
| Синхронизация | Короткий созвон раз в 2 дня (18, 20, 22, 24, 26 июня) |
| Блокеры | Писать в чат сразу, не ждать 2 дня |
| Версия MLflow | Строго **2.15.1** у всех |

---

*Документ создан для внутреннего использования командой. При изменении плана — обновляйте этот файл и делайте commit `docs: update team plan`.*
