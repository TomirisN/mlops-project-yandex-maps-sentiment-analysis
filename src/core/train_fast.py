# src/core/train_fast.py - ЛЕГКАЯ МОДЕЛЬ ДЛЯ CPU
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

print("🚀 Загрузка данных...")

# Загрузка данных
df = pd.read_csv('data/raw/geo-reviews-dataset-2023.csv', nrows=5000)  # 5000 строк для скорости
df = df.dropna(subset=['text', 'rating'])
df = df[df['rating'] >= 1]
df['rating'] = df['rating'].astype(int) - 1  # 1-5 -> 0-4

print(f"✅ Загружено {len(df)} отзывов")

# Разделение
X = df['text'].fillna('').tolist()
y = df['rating'].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение
print("🔄 Обучение модели...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Точность: {acc:.3f}")

# Сохранение
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
print("✅ Модель сохранена в models/model.pkl")

# Сохраняем метрики для DVC
import json
with open('metrics.json', 'w') as f:
    json.dump({"accuracy": float(acc), "samples": len(df)}, f)