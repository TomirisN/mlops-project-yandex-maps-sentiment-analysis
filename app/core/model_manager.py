import joblib
import numpy as np
import os
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_path: str = "models/model.pkl"):
        self.model_path = model_path
        self.model = None
        self._is_loaded = False
        
    def load_model(self) -> bool:
        """Загружает модель из локального файла"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self._is_loaded = True
                logger.info(f"✅ Model loaded from {self.model_path}")
                return True
            else:
                logger.error(f"❌ Model not found: {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Load error: {e}")
            return False
    
    def load_from_mlflow(self, tracking_uri: str, model_name: str, stage: str = "Production") -> bool:
        """Загружает модель из MLflow Model Registry"""
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            model_uri = f"models:/{model_name}/{stage}"
            self.model = mlflow.sklearn.load_model(model_uri)
            self._is_loaded = True
            logger.info(f"✅ Model loaded from MLflow: {model_uri}")
            return True
        except Exception as e:
            logger.error(f"❌ MLflow load error: {e}")
            return False
    
    def predict(self, text: str) -> Tuple[int, float]:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        pred_idx = self.model.predict([text])[0]
        rating = int(pred_idx + 1)
        
        proba = self.model.predict_proba([text])[0]
        confidence = float(np.max(proba))
        
        return rating, confidence
    
    def get_sentiment_label(self, rating: int) -> str:
        if rating <= 2:
            return "negative"
        elif rating == 3:
            return "neutral"
        return "positive"
    
    @property
    def is_loaded(self):
        return self._is_loaded