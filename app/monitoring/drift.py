"""Расчёт data/target/concept drift и генерация Evidently-отчётов."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from app.monitoring.metrics import set_drift_metrics
from app.monitoring.prediction_store import PredictionStore

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    drift_type: str
    detected: bool
    score: float
    details: str


@dataclass
class DriftStatus:
    checked_at: str
    data_drift: DriftResult
    target_drift: DriftResult
    concept_drift: DriftResult
    report_path: Optional[str] = None
    notifications: List[str] = field(default_factory=list)

    @property
    def any_drift_detected(self) -> bool:
        return any(
            [
                self.data_drift.detected,
                self.target_drift.detected,
                self.concept_drift.detected,
            ]
        )


class DriftDetector:
    def __init__(
        self,
        reference_path: str,
        reports_dir: str,
        data_threshold: float = 0.5,
        target_threshold: float = 0.5,
        concept_threshold: float = 0.15,
        min_samples: int = 30,
    ):
        self.reference_path = Path(reference_path)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.data_threshold = data_threshold
        self.target_threshold = target_threshold
        self.concept_threshold = concept_threshold
        self.min_samples = min_samples
        self._last_status: Optional[DriftStatus] = None
        self._reference_df = self._load_reference()

    def _load_reference(self) -> pd.DataFrame:
        if self.reference_path.exists():
            df = pd.read_csv(self.reference_path)
            return self._enrich_features(df)

        logger.warning(
            "Reference data not found at %s, using synthetic reference",
            self.reference_path,
        )
        return self._enrich_features(self._synthetic_reference())

    @staticmethod
    def _synthetic_reference() -> pd.DataFrame:
        texts = [
            "Отличное место, всем рекомендую!",
            "Ужасный сервис, больше не приду.",
            "Нормально, ничего особенного.",
            "Хорошо, но есть куда расти.",
            "Прекрасная атмосфера и вкусная еда.",
            "Долго ждали заказ, разочарованы.",
            "Средний уровень, цена завышена.",
            "Лучшее заведение в районе!",
            "Грязно и неприятный персонал.",
            "Зашли случайно, остались довольны.",
        ] * 20
        ratings = [5, 1, 3, 4, 5, 2, 3, 5, 1, 4] * 20
        confidences = [0.85, 0.78, 0.55, 0.72, 0.91, 0.80, 0.60, 0.88, 0.75, 0.70] * 20
        return pd.DataFrame(
            {"text": texts, "rating": ratings, "confidence": confidences}
        )

    @staticmethod
    def _enrich_features(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["text"] = result["text"].fillna("").astype(str)
        result["text_length"] = result["text"].str.len()
        result["word_count"] = result["text"].str.split().str.len()
        if "rating" not in result.columns:
            result["rating"] = 3
        if "confidence" not in result.columns:
            result["confidence"] = 0.7
        return result

    def check(self, store: PredictionStore) -> DriftStatus:
        current_df = store.get_dataframe_rows(limit=500)
        checked_at = datetime.now(timezone.utc).isoformat()

        if len(current_df) < self.min_samples:
            status = DriftStatus(
                checked_at=checked_at,
                data_drift=DriftResult(
                    "data",
                    False,
                    0.0,
                    f"Недостаточно данных: {len(current_df)}/{self.min_samples}",
                ),
                target_drift=DriftResult("target", False, 0.0, "Недостаточно данных"),
                concept_drift=DriftResult("concept", False, 0.0, "Недостаточно данных"),
                notifications=["Накопите больше предсказаний для анализа дрейфа"],
            )
            self._last_status = status
            return status

        current_df = self._enrich_features(current_df)
        reference_df = self._reference_df

        data_result = self._check_data_drift(reference_df, current_df)
        target_result = self._check_target_drift(reference_df, current_df)
        concept_result = self._check_concept_drift(reference_df, current_df)

        report_path = self._generate_report(reference_df, current_df)

        notifications = []
        for result in (data_result, target_result, concept_result):
            set_drift_metrics(result.drift_type, result.detected, result.score)
            if result.detected:
                notifications.append(f"⚠️ {result.drift_type} drift: {result.details}")

        status = DriftStatus(
            checked_at=checked_at,
            data_drift=data_result,
            target_drift=target_result,
            concept_drift=concept_result,
            report_path=report_path,
            notifications=notifications,
        )
        self._last_status = status
        return status

    def _check_data_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> DriftResult:
        score = self._numeric_drift_score(
            reference["text_length"], current["text_length"]
        )
        word_score = self._numeric_drift_score(
            reference["word_count"], current["word_count"]
        )
        combined = max(score, word_score)
        detected = combined >= self.data_threshold
        details = f"score={combined:.3f} (text_length={score:.3f}, word_count={word_score:.3f})"
        return DriftResult("data", detected, combined, details)

    def _check_target_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> DriftResult:
        ref_dist = reference["rating"].value_counts(normalize=True).sort_index()
        cur_dist = current["rating"].value_counts(normalize=True).sort_index()
        all_ratings = sorted(set(ref_dist.index) | set(cur_dist.index))
        ref_vals = [ref_dist.get(r, 0.0) for r in all_ratings]
        cur_vals = [cur_dist.get(r, 0.0) for r in all_ratings]
        score = float(np.sum(np.abs(np.array(ref_vals) - np.array(cur_vals))) / 2)
        detected = score >= self.target_threshold
        details = f"score={score:.3f}, ref_mean={reference['rating'].mean():.2f}, cur_mean={current['rating'].mean():.2f}"
        return DriftResult("target", detected, score, details)

    def _check_concept_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> DriftResult:
        labeled = current.dropna(subset=["true_rating"])
        if len(labeled) >= 10:
            accuracy = float((labeled["true_rating"] == labeled["rating"]).mean())
            ref_accuracy = 0.7
            score = max(0.0, ref_accuracy - accuracy)
            detected = score >= self.concept_threshold
            details = f"accuracy={accuracy:.3f}, drop={score:.3f}"
            return DriftResult("concept", detected, score, details)

        ref_conf = float(reference["confidence"].mean())
        cur_conf = float(current["confidence"].mean())
        score = abs(ref_conf - cur_conf)
        detected = score >= self.concept_threshold
        details = (
            f"confidence_shift={score:.3f} (ref={ref_conf:.3f}, cur={cur_conf:.3f})"
        )
        return DriftResult("concept", detected, score, details)

    @staticmethod
    def _numeric_drift_score(reference: pd.Series, current: pd.Series) -> float:
        from scipy.stats import ks_2samp

        statistic, _ = ks_2samp(reference.astype(float), current.astype(float))
        return float(statistic)

    def _generate_report(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> Optional[str]:
        try:
            column_mapping = ColumnMapping(
                target="rating",
                numerical_features=["text_length", "word_count", "confidence"],
                categorical_features=["sentiment"],
                text_features=["text"],
            )
            report = Report(
                metrics=[
                    DataDriftPreset(),
                    TargetDriftPreset(),
                ]
            )
            report.run(
                reference_data=reference,
                current_data=current,
                column_mapping=column_mapping,
            )
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_file = self.reports_dir / f"drift_report_{timestamp}.html"
            report.save_html(str(report_file))

            summary_file = self.reports_dir / f"drift_summary_{timestamp}.json"
            summary = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "reference_samples": len(reference),
                "current_samples": len(current),
            }
            summary_file.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return str(report_file)
        except Exception as exc:
            logger.error("Failed to generate Evidently report: %s", exc)
            return None

    @property
    def last_status(self) -> Optional[DriftStatus]:
        return self._last_status

    def list_reports(self, limit: int = 10) -> List[str]:
        reports = sorted(self.reports_dir.glob("drift_report_*.html"), reverse=True)
        return [str(path) for path in reports[:limit]]
