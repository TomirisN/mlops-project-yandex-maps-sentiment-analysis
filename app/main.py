from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

from app.api.monitoring_routes import router as monitoring_router
from app.api.routes import router
from app.api.ui_routes import router as ui_router
from app.config import config
from app.core.model_manager import ModelManager
from app.dependency import (
    set_drift_detector,
    set_model_manager,
    set_prediction_store,
    set_retrain_service,
)
from app.monitoring.drift import DriftDetector
from app.monitoring.metrics import set_model_info
from app.monitoring.prediction_store import PredictionStore
from app.monitoring.retrain_service import RetrainService


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application...")

    prediction_store = PredictionStore(config.PREDICTIONS_DB_PATH)
    drift_detector = DriftDetector(
        reference_path=config.REFERENCE_DATA_PATH,
        reports_dir=config.DRIFT_REPORTS_DIR,
        data_threshold=config.DRIFT_DATA_THRESHOLD,
        target_threshold=config.DRIFT_TARGET_THRESHOLD,
        concept_threshold=config.DRIFT_CONCEPT_THRESHOLD,
        min_samples=config.DRIFT_MIN_SAMPLES,
    )
    retrain_service = RetrainService(
        train_script=config.TRAIN_SCRIPT_PATH,
        project_root=str(config.BASE_DIR),
    )

    set_prediction_store(prediction_store)
    set_drift_detector(drift_detector)
    set_retrain_service(retrain_service)

    model_manager = ModelManager(config.MODEL_PATH)
    model_source = "local"

    if config.USE_MLFLOW:
        print(
            f"Trying to load model from MLflow: {config.MLFLOW_MODEL_NAME} ({config.MLFLOW_MODEL_STAGE})"
        )
        success = model_manager.load_from_mlflow(
            config.MLFLOW_TRACKING_URI,
            config.MLFLOW_MODEL_NAME,
            config.MLFLOW_MODEL_STAGE,
        )
        if success:
            model_source = "mlflow"
        else:
            print("MLflow load failed, falling back to local model...")
            model_manager.load_model()
    else:
        print("Loading model from local file...")
        model_manager.load_model()

    set_model_manager(model_manager)
    set_model_info(model_manager.is_loaded, model_source)

    yield
    print("Shutting down...")


app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(router, prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")
app.include_router(ui_router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/metrics", make_asgi_app())


@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis API",
        "docs": "/docs",
        "api": "/api/v1",
        "ui": "/ui",
        "metrics": "/metrics",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=True)
