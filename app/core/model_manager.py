import joblib
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._is_loaded = False
        
    def load_model(self) -> bool:
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
    
    def predict(self, text: str):
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