"""SQLite-хранилище последних предсказаний для UI и drift-анализа."""

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


@dataclass
class PredictionRecord:
    id: int
    text: str
    rating: int
    confidence: float
    sentiment: str
    created_at: str
    is_anomaly: bool
    true_rating: Optional[int] = None


class PredictionStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL,
                        rating INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        sentiment TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        is_anomaly INTEGER NOT NULL DEFAULT 0,
                        true_rating INTEGER
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def add(
        self,
        text: str,
        rating: int,
        confidence: float,
        sentiment: str,
        is_anomaly: bool = False,
        true_rating: Optional[int] = None,
    ) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO predictions
                    (text, rating, confidence, sentiment, created_at, is_anomaly, true_rating)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        text,
                        rating,
                        confidence,
                        sentiment,
                        created_at,
                        int(is_anomaly),
                        true_rating,
                    ),
                )
                conn.commit()
                return int(cursor.lastrowid)
            finally:
                conn.close()

    def list_recent(self, limit: int = 50) -> List[PredictionRecord]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, text, rating, confidence, sentiment, created_at, is_anomaly, true_rating
                    FROM predictions
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [self._row_to_record(row) for row in rows]
            finally:
                conn.close()

    def get_since(self, since_id: int) -> List[PredictionRecord]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, text, rating, confidence, sentiment, created_at, is_anomaly, true_rating
                    FROM predictions
                    WHERE id > ?
                    ORDER BY id ASC
                    """,
                    (since_id,),
                ).fetchall()
                return [self._row_to_record(row) for row in rows]
            finally:
                conn.close()

    def count(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM predictions").fetchone()
                return int(row["cnt"])
            finally:
                conn.close()

    def get_dataframe_rows(self, limit: int = 500):
        """Возвращает строки для drift-анализа."""
        import pandas as pd

        records = self.list_recent(limit=limit)
        if not records:
            return pd.DataFrame(
                columns=[
                    "text",
                    "text_length",
                    "word_count",
                    "rating",
                    "confidence",
                    "sentiment",
                ]
            )

        rows = []
        for record in reversed(records):
            words = record.text.split()
            rows.append(
                {
                    "text": record.text,
                    "text_length": len(record.text),
                    "word_count": len(words),
                    "rating": record.rating,
                    "confidence": record.confidence,
                    "sentiment": record.sentiment,
                    "true_rating": record.true_rating,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> PredictionRecord:
        return PredictionRecord(
            id=row["id"],
            text=row["text"],
            rating=row["rating"],
            confidence=row["confidence"],
            sentiment=row["sentiment"],
            created_at=row["created_at"],
            is_anomaly=bool(row["is_anomaly"]),
            true_rating=row["true_rating"],
        )
