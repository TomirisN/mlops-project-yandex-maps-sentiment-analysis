"""Обучение v4: 3-class sentiment с accuracy ~0.8 и адаптером для API (рейтинги 1/3/5)."""

import argparse
import json
import os
import re
from typing import Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from app.core.rating_adapter import ThreeClassToRatingEstimator
from src.core.mlflow_utils import ensure_experiment, setup_mlflow_tracking
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rating_to_3class(rating: int) -> int:
    if rating <= 2:
        return 0  # negative
    if rating == 3:
        return 1  # neutral
    return 2  # positive


def load_data(nrows: int, samples_per_class: int) -> Tuple[list, list, list]:
    df = pd.read_csv("data/raw/geo-reviews-dataset-2023.csv", nrows=nrows)
    df = df.dropna(subset=["text", "rating"])
    df = df[df["rating"].between(1, 5)]
    df["full_text"] = (
        df["name_ru"].fillna("") + " " + df["rubrics"].fillna("") + " " + df["text"].fillna("")
    ).map(clean_text)
    df = df[df["full_text"].str.len() >= 15]
    df["y3"] = df["rating"].astype(int).map(rating_to_3class)

    parts = []
    for label in (0, 1, 2):
        class_df = df[df["y3"] == label]
        n = min(samples_per_class, len(class_df))
        if n:
            parts.append(class_df.sample(n=n, random_state=42))

    balanced = pd.concat(parts)
    return balanced["full_text"].tolist(), balanced["y3"].tolist(), balanced["rating"].astype(int).tolist()


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=25000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    max_df=0.95,
                ),
            ),
            ("clf", ThreeClassToRatingEstimator(base_estimator=ComplementNB(alpha=0.1))),
        ]
    )


def mapped_5class_accuracy(model, X_test, true_ratings) -> float:
  mapped_true = [1 if r <= 2 else 3 if r == 3 else 5 for r in true_ratings]
  pred_ratings = [int(p) + 1 for p in model.predict(X_test)]
  return accuracy_score(mapped_true, pred_ratings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default="yandex_maps_sentiment")
    parser.add_argument("--nrows", type=int, default=150000)
    parser.add_argument("--samples-per-class", type=int, default=8000)
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    tracking_uri = setup_mlflow_tracking(args.tracking_uri)
    print(f"MLflow tracking: {tracking_uri}")
    ensure_experiment("sentiment_analysis")
    client = MlflowClient(tracking_uri)

    print("Loading data...")
    X, y3, y_rating = load_data(args.nrows, args.samples_per_class)
    X_train, X_test, y_train, y_test, _, y_rating_test = train_test_split(
        X, y3, y_rating, test_size=0.2, random_state=42, stratify=y3
    )

    print(f"Samples: {len(X)} | train: {len(X_train)} | test: {len(X_test)}")
    model = build_model()
    print("Training high-accuracy v4 (3-class sentiment)...")
    model.fit(X_train, y_train)

    # Оценка: 3-class на тесте
    y_pred3_idx = model.named_steps["clf"].base_estimator.predict(
        model.named_steps["tfidf"].transform(X_test)
    )
    acc3 = accuracy_score(y_test, y_pred3_idx)
    f1_3 = f1_score(y_test, y_pred3_idx, average="macro")
    acc5_mapped = mapped_5class_accuracy(model, X_test, y_rating_test)

    print(f"accuracy (3-class): {acc3:.4f}")
    print(f"f1_macro (3-class): {f1_3:.4f}")
    print(f"accuracy (mapped 1/3/5): {acc5_mapped:.4f}")

    with mlflow.start_run(run_name="high-accuracy-v4") as run:
        mlflow.log_param("model_type", "ComplementNB+TF-IDF")
        mlflow.log_param("config_name", "high-accuracy-v4")
        mlflow.log_param("task", "3class_sentiment")
        mlflow.log_param("features", "word+char+name+rubrics")
        mlflow.log_param("samples_per_class", args.samples_per_class)
        mlflow.log_param("nrows", args.nrows)
        mlflow.log_metric("accuracy", acc3)
        mlflow.log_metric("f1_macro", f1_3)
        mlflow.log_metric("accuracy_5class_mapped", acc5_mapped)
        mlflow.set_tag(
            "description",
            "3-class sentiment (neg/neu/pos), API отдаёт рейтинги 1/3/5",
        )

        mlflow.sklearn.log_model(model, "sentiment_model")
        result = mlflow.register_model(f"runs:/{run.info.run_id}/sentiment_model", args.model_name)
        version = int(result.version)
        print(f"Registered {args.model_name} v{version}")

    if not args.no_promote:
        for mv in client.search_model_versions(f"name='{args.model_name}'"):
            ver = int(mv.version)
            if ver == version:
                client.transition_model_version_stage(
                    args.model_name, ver, "Production", archive_existing_versions=True
                )
                print(f"v{ver} -> Production")
            else:
                client.transition_model_version_stage(args.model_name, ver, "Staging")
                print(f"v{ver} -> Staging")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": acc3,
                    "f1_macro": f1_3,
                    "accuracy_5class_mapped": acc5_mapped,
                    "config_name": "high-accuracy-v4",
                    "mlflow_run_id": run.info.run_id,
                    "mlflow_version": version,
                    "task": "3class_sentiment",
                },
                f,
                indent=2,
            )

    print("\nDone. Restart API to load new Production model.")
    print(f"MLflow: {args.tracking_uri}")


if __name__ == "__main__":
    main()
