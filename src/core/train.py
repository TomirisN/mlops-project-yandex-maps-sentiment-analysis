# src/core/train_bert.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import warnings
import mlflow
import mlflow.pytorch
import joblib
import yaml
import os
from pathlib import Path
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================
# 1. АРГУМЕНТЫ КОМАНДНОЙ СТРОКИ
# ============================================
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/config.yaml")
parser.add_argument("--data-path", type=str, default="data/raw/geo-reviews-dataset-2023.csv")
parser.add_argument("--max-samples", type=int, default=100000)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--max-len", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-5)
args = parser.parse_args()

# Загрузка конфига
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# ============================================
# 2. ФУНКЦИИ ДЛЯ ОБРАБОТКИ ДАННЫХ
# ============================================
def clean_text(text):
    """Простая очистка текста"""
    if not isinstance(text, str):
        return ""
    text = ' '.join(text.split())
    return text

def extract_rubrics(rubrics):
    """Извлекает список категорий из строки"""
    if not isinstance(rubrics, str):
        return []
    try:
        if rubrics.startswith('[') and rubrics.endswith(']'):
            rubrics = rubrics.strip('[]').replace("'", "").replace('"', '')
            return [r.strip() for r in rubrics.split(',')]
    except:
        pass
    return [rubrics] if rubrics else []

def load_data(data_path, max_samples):
    """Загрузка данных из DVC"""
    print("\n" + "="*60)
    print("ЗАГРУЗКА ДАННЫХ ИЗ DVC")
    print("="*60)
    
    # Проверяем существование файла
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Данные не найдены: {data_path}. Запустите 'dvc pull'")
    
    data = pd.read_csv(data_path)
    print(f"Оригинальный размер: {len(data)} строк")
    
    # Берём max_samples строк
    df = data.iloc[:max_samples].copy()
    print(f"Используем {len(df)} строк")
    
    return df

def prepare_data(df):
    """Подготовка данных для обучения"""
    print("\n" + "="*60)
    print("ОБРАБОТКА ДАННЫХ")
    print("="*60)
    
    # Удаляем пропуски
    df = df.dropna(subset=['text', 'rating'])
    print(f"После удаления пропусков: {len(df)} строк")
    
    # Очищаем текст
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Преобразуем rating в int и удаляем оценки 0
    df['rating'] = df['rating'].astype(int)
    df = df[df['rating'] >= 1]
    print(f"После удаления оценок 0: {len(df)} строк")
    
    # Распределение оценок
    print("\nРаспределение оценок:")
    for r in sorted(df['rating'].unique()):
        count = len(df[df['rating'] == r])
        print(f"  {r}: {count} ({count/len(df)*100:.1f}%)")
    
    # Обработка рубрик (опционально)
    df['rubrics_list'] = df['rubrics'].apply(extract_rubrics)
    df['main_category'] = df['rubrics_list'].apply(lambda x: x[0] if len(x) > 0 else 'unknown')
    
    return df

# ============================================
# 3. DATASET КЛАСС
# ============================================
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

# ============================================
# 4. ФУНКЦИИ ОБУЧЕНИЯ
# ============================================
def train_epoch(model, loader, optimizer, device):
    """Обучает модель одну эпоху"""
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device):
    """Оценивает модель"""
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    # Преобразуем обратно из 0-4 в 1-5
    predictions = [p + 1 for p in predictions]
    actuals = [a + 1 for a in actuals]

    accuracy = accuracy_score(actuals, predictions)
    f1 = f1_score(actuals, predictions, average='weighted')

    return accuracy, f1, predictions

# ============================================
# 5. ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# ============================================
def main():
    # Проверка GPU
    print("\n" + "="*60)
    print("ПРОВЕРКА GPU")
    print("="*60)
    print(f"CUDA доступен: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("⚠️ GPU не найден, обучение будет на CPU (медленно)")
    
    # 1. Загружаем данные
    df = load_data(args.data_path, args.max_samples)
    df = prepare_data(df)
    
    # 2. Подготовка данных для BERT
    print("\n" + "="*60)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ BERT")
    print("="*60)
    
    texts = df['clean_text'].tolist()
    labels = df['rating'].tolist()
    
    # Разделение на train/val/test (70/15/15)
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels,
        test_size=0.15,
        random_state=42,
        stratify=labels
    )
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=0.176,
        random_state=42,
        stratify=train_val_labels
    )
    
    print(f"Train: {len(train_texts)} строк")
    print(f"Validation: {len(val_texts)} строк")
    print(f"Test: {len(test_texts)} строк")
    
    # Преобразуем метки из 1-5 в 0-4
    train_labels = [l - 1 for l in train_labels]
    val_labels = [l - 1 for l in val_labels]
    test_labels = [l - 1 for l in test_labels]
    
    # 3. Токенизация
    print("\n" + "="*60)
    print("ТОКЕНИЗАЦИЯ")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    
    def tokenize_texts(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=args.max_len,
            return_tensors='pt'
        )
    
    train_encodings = tokenize_texts(train_texts)
    val_encodings = tokenize_texts(val_texts)
    test_encodings = tokenize_texts(test_texts)
    
    # 4. Создание Dataset и DataLoader
    train_dataset = ReviewDataset(train_encodings, train_labels)
    val_dataset = ReviewDataset(val_encodings, val_labels)
    test_dataset = ReviewDataset(test_encodings, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 5. Создание модели
    print("\n" + "="*60)
    print("СОЗДАНИЕ МОДЕЛИ")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        'DeepPavlov/rubert-base-cased',
        num_labels=5
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # 6. MLflow логирование
    mlflow.set_experiment("sentiment-bert")
    
    with mlflow.start_run() as run:
        # Логируем параметры
        mlflow.log_params({
            "model_name": "DeepPavlov/rubert-base-cased",
            "max_samples": args.max_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "learning_rate": args.lr,
            "num_classes": 5,
            "train_size": len(train_texts),
            "val_size": len(val_texts),
            "test_size": len(test_texts)
        })
        
        # 7. Обучение
        print("\n" + "="*60)
        print("НАЧАЛО ОБУЧЕНИЯ")
        print("="*60)
        
        best_val_f1 = 0
        history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
        
        for epoch in range(args.epochs):
            print(f"\nЭпоха {epoch+1}/{args.epochs}")
            print("-" * 40)
            
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_acc, val_f1, _ = evaluate(model, val_loader, device)
            
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            # Логируем метрики
            mlflow.log_metrics({
                f"train_loss_epoch_{epoch}": train_loss,
                f"val_accuracy_epoch_{epoch}": val_acc,
                f"val_f1_epoch_{epoch}": val_f1
            })
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
            print(f"Val F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                # Сохраняем модель в формате для MLflow
                mlflow.pytorch.log_model(model, "bert_model")
                # И в локальную папку для FastAPI
                Path("models").mkdir(exist_ok=True)
                torch.save(model.state_dict(), 'models/best_model.pt')
                # Сохраняем токенизатор
                tokenizer.save_pretrained('models/tokenizer')
                print(f"✅ Сохранена лучшая модель! (F1: {val_f1:.4f})")
        
        # 8. Тестирование
        print("\n" + "="*60)
        print("ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ")
        print("="*60)
        
        # Загружаем лучшую модель
        model.load_state_dict(torch.load('models/best_model.pt'))
        test_acc, test_f1, test_preds = evaluate(model, test_loader, device)
        
        # Логируем финальные метрики
        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_f1_weighted": test_f1
        })
        
        print(f"\nРЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
        print(f"Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
        print(f"Weighted F1: {test_f1:.4f}")
        
        # Сохраняем историю
        history_df = pd.DataFrame(history)
        history_df.to_csv('models/training_history.csv', index=False)
        
        # Сохраняем метрики для DVC
        metrics_dict = {
            "best_val_f1": float(best_val_f1),
            "test_accuracy": float(test_acc),
            "test_f1_weighted": float(test_f1)
        }
        
        with open('metrics.json', 'w') as f:
            import json
            json.dump(metrics_dict, f)
    
    print("\n✅ Обучение завершено!")
    print(f"📊 Результаты сохранены в MLflow")
    print(f"🤖 Модель сохранена в 'models/best_model.pt'")
    print(f"📝 Метрики сохранены в 'metrics.json'")

if __name__ == "__main__":
    main()