from typing import Optional

from fastapi import HTTPException

from app.core.model_manager import ModelManager
from app.monitoring.drift import DriftDetector
from app.monitoring.prediction_store import PredictionStore
from app.monitoring.retrain_service import RetrainService

_model_manager: Optional[ModelManager] = None
_prediction_store: Optional[PredictionStore] = None
_drift_detector: Optional[DriftDetector] = None
_retrain_service: Optional[RetrainService] = None


def set_model_manager(manager: ModelManager) -> None:
    global _model_manager
    _model_manager = manager


def get_model_manager() -> ModelManager:
    if _model_manager is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return _model_manager


def set_prediction_store(store: PredictionStore) -> None:
    global _prediction_store
    _prediction_store = store


def get_prediction_store() -> PredictionStore:
    if _prediction_store is None:
        raise HTTPException(status_code=503, detail="Prediction store not initialized")
    return _prediction_store


def set_drift_detector(detector: DriftDetector) -> None:
    global _drift_detector
    _drift_detector = detector


def get_drift_detector() -> DriftDetector:
    if _drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    return _drift_detector


def set_retrain_service(service: RetrainService) -> None:
    global _retrain_service
    _retrain_service = service


def get_retrain_service() -> RetrainService:
    if _retrain_service is None:
        raise HTTPException(status_code=503, detail="Retrain service not initialized")
    return _retrain_service
