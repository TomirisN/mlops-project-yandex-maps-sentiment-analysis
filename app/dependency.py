from fastapi import HTTPException
from typing import Optional
from app.core.model_manager import ModelManager

_model_manager: Optional[ModelManager] = None

def set_model_manager(manager: ModelManager):
    global _model_manager
    _model_manager = manager

def get_model_manager() -> ModelManager:
    if _model_manager is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return _model_manager