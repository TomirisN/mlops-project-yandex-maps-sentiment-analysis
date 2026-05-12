from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config import config
from app.core.model_manager import ModelManager
from app.api.routes import router
from app.dependency import set_model_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting...")
    manager = ModelManager(config.MODEL_PATH)
    manager.load_model()
    set_model_manager(manager)
    yield
    print("Shutting down...")

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=True)