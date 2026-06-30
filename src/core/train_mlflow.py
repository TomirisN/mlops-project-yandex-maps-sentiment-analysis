import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment_analysis")
MODEL_REGISTRY_NAME = os.getenv("MLFLOW_MODEL_NAME", "yandex_maps_sentiment")
DATASET_PATH = os.getenv(
    "TRAIN_DATASET_PATH", "data/raw/geo-reviews-dataset-2023.csv"
)
MAX_ROWS = int(os.getenv("TRAIN_MAX_ROWS", "50000"))
SAMPLES_PER_CLASS = int(os.getenv("TRAIN_SAMPLES_PER_CLASS", "3000"))
AUTO_PROMOTE = os.getenv("MLFLOW_AUTO_PROMOTE", "true").lower() == "true"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"🚀 Загрузка данных из {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH, nrows=MAX_ROWS)
df = df.dropna(subset=['text', 'rating'])
df = df[df['rating'] >= 1]
df['rating'] = df['rating'].astype(int) - 1

# Балансировка классов
print("⚖️ Балансировка классов...")
balanced_dfs = []
samples_per_class = SAMPLES_PER_CLASS

for rating in range(5):
    class_df = df[df['rating'] == rating]
    if len(class_df) >= samples_per_class:
        sampled = class_df.sample(n=samples_per_class, random_state=42)
    else:
        sampled = class_df
    balanced_dfs.append(sampled)
    print(f"  Класс {rating+1}: {len(sampled)} отзывов")

df_balanced = pd.concat(balanced_dfs)
print(f"✅ После балансировки: {len(df_balanced)} отзывов")

# Подготовка данных
X = df_balanced['text'].fillna('').tolist()
y = df_balanced['rating'].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
print("🔄 Обучение модели...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Точность: {acc:.3f}")

# Регистрация в MLflow
print("📦 Регистрация модели в MLflow...")
with mlflow.start_run() as run:
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("samples_per_class", samples_per_class)
    mlflow.log_metric("accuracy", acc)
    
    # Логируем модель
    mlflow.sklearn.log_model(model, "sentiment_model")
    
    # Регистрируем в Model Registry
    model_uri = f"runs:/{run.info.run_id}/sentiment_model"
    registered = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)

    if AUTO_PROMOTE:
        client = MlflowClient(tracking_uri=TRACKING_URI)
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=registered.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"✅ Версия {registered.version} переведена в Production")

    print("✅ Модель зарегистрирована в MLflow")
    print(f"   Run ID: {run.info.run_id}")

# Сохраняем локально
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
print("✅ Модель сохранена локально в models/model.pkl")

with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": float(acc),
        "samples": len(df_balanced),
        "balanced": True,
        "mlflow_run_id": run.info.run_id
    }, f)

print("✅ Готово!")
ui_hint = TRACKING_URI if TRACKING_URI.startswith("http") else "http://localhost:5000"
print(f"🌐 Откройте MLflow UI: {ui_hint}")