"""Обучение нескольких версий модели для сравнения в MLflow Model Registry.

Создаёт v2, v3, ... с улучшенными гиперпараметрами и переводит лучшую в Production.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class ModelConfig:
    name: str
    description: str
    max_features: int
    ngram_range: Tuple[int, int]
    samples_per_class: int
    nrows: int
    c: float
    max_iter: int


MODEL_CONFIGS: List[ModelConfig] = [
    ModelConfig(
        name="baseline-v1",
        description="Базовая модель: TF-IDF 5k, unigrams (как v1)",
        max_features=5000,
        ngram_range=(1, 1),
        samples_per_class=3000,
        nrows=50000,
        c=1.0,
        max_iter=1000,
    ),
    ModelConfig(
        name="improved-v2",
        description="Улучшенная: TF-IDF 10k, bi-grams, больше данных",
        max_features=10000,
        ngram_range=(1, 2),
        samples_per_class=4000,
        nrows=80000,
        c=2.0,
        max_iter=1500,
    ),
    ModelConfig(
        name="best-v3",
        description="Лучшая: TF-IDF 15k, bi-grams, 5k/class, C=4",
        max_features=15000,
        ngram_range=(1, 2),
        samples_per_class=5000,
        nrows=100000,
        c=4.0,
        max_iter=2000,
    ),
]


def load_balanced_data(nrows: int, samples_per_class: int) -> Tuple[List[str], List[int]]:
    df = pd.read_csv("data/raw/geo-reviews-dataset-2023.csv", nrows=nrows)
    df = df.dropna(subset=["text", "rating"])
    df = df[df["rating"] >= 1]
    df["rating"] = df["rating"].astype(int) - 1

    balanced = []
    for rating in range(5):
        class_df = df[df["rating"] == rating]
        n = min(samples_per_class, len(class_df))
        if n > 0:
            balanced.append(class_df.sample(n=n, random_state=42))

    df_balanced = pd.concat(balanced)
    X = df_balanced["text"].fillna("").tolist()
    y = df_balanced["rating"].tolist()
    return X, y


def train_one(config: ModelConfig, register: bool, model_name: str) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Training: {config.name}")
    print(f"  {config.description}")
    print(f"{'='*60}")

    X, y = load_balanced_data(config.nrows, config.samples_per_class)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=config.max_features,
                    ngram_range=config.ngram_range,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=config.max_iter,
                    class_weight="balanced",
                    C=config.c,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 macro: {f1_macro:.4f}")
    print(f"  Samples:  {len(X)}")

    run_id = None
    version = None

    with mlflow.start_run(run_name=config.name) as run:
        run_id = run.info.run_id
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("config_name", config.name)
        mlflow.log_param("max_features", config.max_features)
        mlflow.log_param("ngram_range", str(config.ngram_range))
        mlflow.log_param("samples_per_class", config.samples_per_class)
        mlflow.log_param("nrows", config.nrows)
        mlflow.log_param("C", config.c)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.set_tag("description", config.description)

        mlflow.sklearn.log_model(model, "sentiment_model")

        if register:
            model_uri = f"runs:/{run_id}/sentiment_model"
            result = mlflow.register_model(model_uri, model_name)
            version = int(result.version)
            print(f"  Registered: {model_name} v{version}")

    return {
        "config_name": config.name,
        "run_id": run_id,
        "version": version,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "samples": len(X),
        "model": model,
    }


def promote_best(client: MlflowClient, model_name: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = max(results, key=lambda r: (r["accuracy"], r["f1_macro"]))
    print(f"\nBest model: {best['config_name']} (v{best['version']}, accuracy={best['accuracy']:.4f})")

    versions = client.search_model_versions(f"name='{model_name}'")
    for mv in versions:
        ver = int(mv.version)
        if ver == best["version"]:
            client.transition_model_version_stage(model_name, ver, "Production", archive_existing_versions=True)
            print(f"  v{ver} -> Production")
        else:
            client.transition_model_version_stage(model_name, ver, "Staging")
            print(f"  v{ver} -> Staging")

    return best


def main():
    parser = argparse.ArgumentParser(description="Train multiple model versions for MLflow Registry")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--model-name", default="yandex_maps_sentiment")
    parser.add_argument("--experiment", default="sentiment_analysis")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=["improved-v2", "best-v3"],
        help="Config names to train (default: improved-v2 best-v3). Use 'all' for all configs.",
    )
    parser.add_argument("--no-promote", action="store_true", help="Do not change Production/Staging stages")
    parser.add_argument("--no-register", action="store_true", help="Log runs only, skip registry")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    client = MlflowClient(args.tracking_uri)

    if "all" in args.configs:
        selected = MODEL_CONFIGS
    else:
        names = set(args.configs)
        selected = [c for c in MODEL_CONFIGS if c.name in names]
        if not selected:
            raise ValueError(f"No configs matched: {args.configs}. Available: {[c.name for c in MODEL_CONFIGS]}")

    results = []
    for config in selected:
        result = train_one(config, register=not args.no_register, model_name=args.model_name)
        results.append(result)

    if not args.no_register and not args.no_promote and results:
        best = promote_best(client, args.model_name, results)

        os.makedirs("models", exist_ok=True)
        joblib.dump(best["model"], "models/model.pkl")

        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": best["accuracy"],
                    "f1_macro": best["f1_macro"],
                    "samples": best["samples"],
                    "config_name": best["config_name"],
                    "mlflow_run_id": best["run_id"],
                    "mlflow_version": best["version"],
                },
                f,
                indent=2,
            )
        print(f"\nBest model saved to models/model.pkl")
        print("Restart API to load new Production model from MLflow.")

    print("\nDone! Open MLflow UI -> Models -> yandex_maps_sentiment")


if __name__ == "__main__":
    main()
