"""Удаляет все версии модели из MLflow Registry (артефакты runs не трогает)."""
import os
from pathlib import Path

from mlflow.tracking import MlflowClient

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "yandex_maps_sentiment")


def _tracking_uri() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    if Path("/mlflow/mlruns").exists():
        return "file:///mlflow/mlruns"
    return (Path(__file__).resolve().parents[1] / "mlruns").resolve().as_uri()


URI = _tracking_uri()
client = MlflowClient(URI)
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    print(f"No versions for {MODEL_NAME}")
else:
    for mv in versions:
        client.delete_model_version(MODEL_NAME, mv.version)
        print(f"Deleted v{mv.version}")
