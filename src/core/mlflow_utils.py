"""Единая настройка MLflow tracking URI (file store на диске проекта)."""

import os
from pathlib import Path

import mlflow


def setup_mlflow_tracking(explicit_uri: str | None = None) -> str:
    """file:// store: на хосте — ./mlruns, в Docker — /mlflow/mlruns (общий volume с API)."""
    uri = explicit_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        if Path("/mlflow/mlruns").exists():
            uri = "file:///mlflow/mlruns"
        else:
            root = Path(__file__).resolve().parents[2]
            uri = (root / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    return uri


def ensure_experiment(name: str) -> None:
    mlflow.set_experiment(name)
