from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config import config
from app.core.model_manager import ModelManager
from app.api.routes import router
from app.dependency import set_model_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting application...")
    
    model_manager = ModelManager(config.MODEL_PATH)
    
    # Пробуем загрузить из MLflow, если настроено
    if config.USE_MLFLOW:
        print(f"📦 Trying to load model from MLflow: {config.MLFLOW_MODEL_NAME} ({config.MLFLOW_MODEL_STAGE})")
        success = model_manager.load_from_mlflow(
            config.MLFLOW_TRACKING_URI,
            config.MLFLOW_MODEL_NAME,
            config.MLFLOW_MODEL_STAGE
        )
        if not success:
            print("⚠️ MLflow load failed, falling back to local model...")
            model_manager.load_model()
    else:
        print("📁 Loading model from local file...")
        model_manager.load_model()
    
    set_model_manager(model_manager)
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API", "docs": "/docs", "api": "/api/v1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=True)