from pathlib import Path
from typing import List

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.config import config
from app.dependency import get_drift_detector, get_prediction_store, get_retrain_service
from app.monitoring.drift import DriftDetector, DriftStatus
from app.monitoring.prediction_store import PredictionStore, PredictionRecord
from app.monitoring.retrain_service import RetrainService
from app.schemas import (
    DriftItemResponse,
    DriftStatusResponse,
    ExperimentRunResponse,
    ExperimentsResponse,
    PredictionItemResponse,
    PredictionsListResponse,
    RetrainResponse,
)

router = APIRouter()


def _drift_status_to_response(status: DriftStatus) -> DriftStatusResponse:
    return DriftStatusResponse(
        checked_at=status.checked_at,
        data_drift=DriftItemResponse(**status.data_drift.__dict__),
        target_drift=DriftItemResponse(**status.target_drift.__dict__),
        concept_drift=DriftItemResponse(**status.concept_drift.__dict__),
        any_drift_detected=status.any_drift_detected,
        report_path=status.report_path,
        notifications=status.notifications,
    )


def _prediction_to_response(record: PredictionRecord) -> PredictionItemResponse:
    return PredictionItemResponse(
        id=record.id,
        text=record.text,
        rating=record.rating,
        confidence=record.confidence,
        sentiment=record.sentiment,
        created_at=record.created_at,
        is_anomaly=record.is_anomaly,
        true_rating=record.true_rating,
    )


@router.get("/drift/status", response_model=DriftStatusResponse)
async def drift_status(detector: DriftDetector = Depends(get_drift_detector)):
    if detector.last_status is None:
        raise HTTPException(
            status_code=404,
            detail="Дрейф ещё не проверялся. Вызовите POST /drift/check",
        )
    return _drift_status_to_response(detector.last_status)


@router.post("/drift/check", response_model=DriftStatusResponse)
async def drift_check(
    detector: DriftDetector = Depends(get_drift_detector),
    store: PredictionStore = Depends(get_prediction_store),
    retrain_service: RetrainService = Depends(get_retrain_service),
):
    status = detector.check(store)

    if (
        config.AUTO_RETRAIN_ON_DRIFT
        and status.any_drift_detected
        and not retrain_service.is_running()
    ):
        retrain_state = retrain_service.start()
        status.notifications.append(
            f"🔄 Автопереобучение запущено: {retrain_state.message}"
        )

    return _drift_status_to_response(status)


@router.get("/drift/reports")
async def drift_reports(detector: DriftDetector = Depends(get_drift_detector)) -> dict:
    reports = detector.list_reports()
    return {"reports": [Path(path).name for path in reports]}


@router.get("/drift/reports/{filename}")
async def drift_report_file(filename: str):
    report_path = Path(config.DRIFT_REPORTS_DIR) / filename
    if not report_path.exists() or not report_path.suffix == ".html":
        raise HTTPException(status_code=404, detail="Отчёт не найден")
    return FileResponse(report_path, media_type="text/html")


@router.get("/predictions", response_model=PredictionsListResponse)
async def list_predictions(
    limit: int = 50,
    store: PredictionStore = Depends(get_prediction_store),
):
    items = store.list_recent(limit=limit)
    return PredictionsListResponse(
        total=store.count(),
        items=[_prediction_to_response(item) for item in items],
    )


@router.post("/retrain", response_model=RetrainResponse)
async def retrain(service: RetrainService = Depends(get_retrain_service)):
    state = service.start()
    return RetrainResponse(
        status=state.status,
        message=state.message,
        started_at=state.started_at,
    )


@router.get("/retrain/status", response_model=RetrainResponse)
async def retrain_status(service: RetrainService = Depends(get_retrain_service)):
    state = service.state
    return RetrainResponse(
        status=state.status,
        message=state.message,
        started_at=state.started_at,
    )


@router.get("/experiments", response_model=ExperimentsResponse)
async def list_experiments() -> ExperimentsResponse:
    runs: List[ExperimentRunResponse] = []
    experiment_name = "sentiment_analysis"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            exp_resp = await client.get(
                f"{config.MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/get-by-name",
                params={"experiment_name": experiment_name},
            )
            if exp_resp.status_code != 200:
                return ExperimentsResponse(experiment_name=experiment_name, runs=[])

            experiment_id = exp_resp.json()["experiment"]["experiment_id"]
            runs_resp = await client.get(
                f"{config.MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/search",
                json={
                    "experiment_ids": [experiment_id],
                    "max_results": 20,
                    "order_by": ["attributes.start_time DESC"],
                },
            )
            if runs_resp.status_code != 200:
                return ExperimentsResponse(experiment_name=experiment_name, runs=[])

            for run in runs_resp.json().get("runs", []):
                metrics = {m["key"]: float(m["value"]) for m in run.get("metrics", [])}
                runs.append(
                    ExperimentRunResponse(
                        run_id=run["info"]["run_id"],
                        run_name=run["info"].get("run_name")
                        or run["info"]["run_id"][:8],
                        status=run["info"]["status"],
                        accuracy=metrics.get("accuracy"),
                        start_time=run["info"].get("start_time"),
                    )
                )
    except Exception:
        return ExperimentsResponse(experiment_name=experiment_name, runs=[])

    return ExperimentsResponse(experiment_name=experiment_name, runs=runs)
