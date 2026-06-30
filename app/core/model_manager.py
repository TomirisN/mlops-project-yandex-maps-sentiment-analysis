import joblib
import numpy as np
import os
import logging
import sys
from typing import Tuple

from app.core.rating_adapter import ThreeClassToRatingEstimator
from app.core.text_utils import clean_text

# Для загрузки v4, сохранённой из train_v4_high_accuracy.py или __main__
import types
import __main__

__main__.ThreeClassToRatingEstimator = ThreeClassToRatingEstimator
if "train_v4_high_accuracy" not in sys.modules:
    _stub = types.ModuleType("train_v4_high_accuracy")
    _stub.ThreeClassToRatingEstimator = ThreeClassToRatingEstimator
    sys.modules["train_v4_high_accuracy"] = _stub

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

    def load_from_mlflow(
        self, tracking_uri: str, model_name: str, stage: str = "Production"
    ) -> bool:
        """Загружает модель из MLflow Registry (run_id + прямой путь в Docker file store)."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            from pathlib import Path

            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri)
            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                raise RuntimeError(f"No {stage} version for {model_name}")

            run_id = versions[0].run_id
            run = client.get_run(run_id)
            exp_id = run.info.experiment_id

            # Docker + Windows bind mount: meta.yaml хранит C:/ пути, грузим по run_id
            docker_root = Path("/mlflow/mlruns")
            if docker_root.exists():
                model_dir = (
                    docker_root / exp_id / run_id / "artifacts" / "sentiment_model"
                )
                if model_dir.is_dir():
                    self.model = mlflow.sklearn.load_model(str(model_dir))
                    self._is_loaded = True
                    logger.info(
                        f"✅ Model loaded from MLflow: {model_name} {stage} "
                        f"(run={run_id[:12]}, path={model_dir})"
                    )
                    return True

            model_uri = f"runs:/{run_id}/sentiment_model"
            self.model = mlflow.sklearn.load_model(model_uri)
            self._is_loaded = True
            logger.info(
                f"✅ Model loaded from MLflow: {model_name} {stage} (run={run_id[:12]})"
            )
            return True
        except Exception as e:
            logger.error(f"❌ MLflow load error: {e}")
            return False

    def predict(self, text: str) -> Tuple[int, float]:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        cleaned = clean_text(text)
        pred_idx = self.model.predict([cleaned])[0]
        rating = int(pred_idx + 1)

        proba = self.model.predict_proba([cleaned])[0]
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
