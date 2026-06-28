"""Проверка версий модели в MLflow Registry."""
import os
from pathlib import Path

from mlflow.tracking import MlflowClient


def _tracking_uri() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    if Path("/mlflow/mlruns").exists():
        return "file:///mlflow/mlruns"
    return (Path(__file__).resolve().parents[1] / "mlruns").resolve().as_uri()


uri = _tracking_uri()
client = MlflowClient(uri)

print(f"Tracking URI: {uri}\n")
for mv in client.search_model_versions("name='yandex_maps_sentiment'"):
    print(f"v{mv.version} | {mv.current_stage or 'None':12} | run={mv.run_id[:12]} | source={mv.source}")

prod = client.get_latest_versions("yandex_maps_sentiment", stages=["Production"])
if prod:
    print(f"\nProduction: v{prod[0].version}")
