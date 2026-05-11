# src/core/train_fast.py - ЛЕГКАЯ МОДЕЛЬ ДЛЯ CPU
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
# src/core/train_fast.py - ЛЕГКАЯ МОДЕЛЬ С БАЛАНСИРОВКОЙ
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json

print("🚀 Загрузка данных...")

# Загрузка данных
df = pd.read_csv('data/raw/geo-reviews-dataset-2023.csv', nrows=50000)
df = df.dropna(subset=['text', 'rating'])
df = df[df['rating'] >= 1]
df['rating'] = df['rating'].astype(int) - 1  # 1-5 -> 0-4

print(f"✅ Загружено {len(df)} отзывов")

# ПОКАЗЫВАЕМ РАСПРЕДЕЛЕНИЕ ДО БАЛАНСИРОВКИ
print("\n📊 Распределение до балансировки:")
for rating in range(5):
    count = len(df[df['rating'] == rating])
    print(f"  Оценка {rating+1}: {count} ({count/len(df)*100:.1f}%)")

# БАЛАНСИРОВКА: берем одинаковое количество для каждого класса
balanced_dfs = []
samples_per_class = 3000  # по 3000 отзывов на каждую оценку

for rating in range(5):
    class_df = df[df['rating'] == rating]
    if len(class_df) >= samples_per_class:
        sampled = class_df.sample(n=samples_per_class, random_state=42)
    else:
        sampled = class_df  # если меньше, берем все
    balanced_dfs.append(sampled)
    print(f"  Класс {rating+1}: взято {len(sampled)} отзывов")

df_balanced = pd.concat(balanced_dfs)
print(f"\n✅ После балансировки: {len(df_balanced)} отзывов")

# Разделение на train/test
X = df_balanced['text'].fillna('').tolist()
y = df_balanced['rating'].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")

# Обучение модели
print("\n🔄 Обучение модели...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Точность: {acc:.3f}")

# Детальный отчет
print("\n📋 Детальный отчет по каждому классу:")
print(classification_report(y_test, y_pred, 
                          target_names=['1⭐', '2⭐', '3⭐', '4⭐', '5⭐']))

# Сохранение модели
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
print("\n✅ Модель сохранена в models/model.pkl")

# Сохраняем метрики
with open('metrics.json', 'w') as f:
    json.dump({
        "accuracy": float(acc), 
        "samples": len(df_balanced),
        "balanced": True
    }, f)

print("✅ Метрики сохранены в metrics.json")