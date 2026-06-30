"""Перевести указанную версию модели в Production (остальные — Staging/Archived)."""

import argparse
import os
from pathlib import Path

from mlflow.tracking import MlflowClient


def tracking_uri() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    if Path("/mlflow/mlruns").exists():
        return "file:///mlflow/mlruns"
    return (Path(__file__).resolve().parents[1] / "mlruns").resolve().as_uri()


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote MLflow model version to Production")
    parser.add_argument("--version", type=int, required=True, help="Номер версии, напр. 4")
    parser.add_argument(
        "--model-name", default="yandex_maps_sentiment", help="Имя в Model Registry"
    )
    args = parser.parse_args()

    uri = tracking_uri()
    client = MlflowClient(uri)
    target = args.version

    versions = client.search_model_versions(f"name='{args.model_name}'")
    if not any(int(mv.version) == target for mv in versions):
        raise SystemExit(f"Версия v{target} не найдена для {args.model_name}")

    print(f"Tracking URI: {uri}")
    print(f"Переводим v{target} -> Production...")

    client.transition_model_version_stage(
        args.model_name,
        str(target),
        "Production",
        archive_existing_versions=True,
    )

    for mv in versions:
        ver = int(mv.version)
        if ver == target:
            continue
        stage = mv.current_stage or "None"
        if stage == "Production":
            client.transition_model_version_stage(args.model_name, str(ver), "Archived")
            print(f"  v{ver} -> Archived")
        elif stage not in ("Staging", "Archived"):
            client.transition_model_version_stage(args.model_name, str(ver), "Staging")
            print(f"  v{ver} -> Staging")

    prod = client.get_latest_versions(args.model_name, stages=["Production"])
    if prod:
        print(f"\nГотово. Production: v{prod[0].version}")
    else:
        print("\nОшибка: Production не назначен")


if __name__ == "__main__":
    main()
