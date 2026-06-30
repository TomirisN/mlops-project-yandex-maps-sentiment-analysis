"""Фоновый запуск переобучения модели."""

import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.monitoring.metrics import set_retrain_status

logger = logging.getLogger(__name__)


@dataclass
class RetrainState:
    status: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    message: str = "idle"
    exit_code: Optional[int] = None


class RetrainService:
    def __init__(
        self,
        train_script: str,
        project_root: str,
        train_env: Optional[dict] = None,
        on_success=None,
    ):
        self.train_script = Path(train_script)
        self.project_root = Path(project_root)
        self.train_env = train_env or {}
        self.on_success = on_success
        self._lock = threading.Lock()
        self._state = RetrainState(status="idle")
        self._thread: Optional[threading.Thread] = None

    @property
    def state(self) -> RetrainState:
        with self._lock:
            return self._state

    def is_running(self) -> bool:
        with self._lock:
            return self._state.status == "running"

    def start(self) -> RetrainState:
        with self._lock:
            if self._state.status == "running":
                return self._state

            self._state = RetrainState(
                status="running",
                started_at=datetime.now(timezone.utc).isoformat(),
                message="Переобучение запущено",
            )
            set_retrain_status(1)

        self._thread = threading.Thread(target=self._run_training, daemon=True)
        self._thread.start()
        return self.state

    def _run_training(self) -> None:
        try:
            env = os.environ.copy()
            env.update(self.train_env)
            result = subprocess.run(
                [sys.executable, str(self.train_script)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=3600,
                env=env,
            )
            finished_at = datetime.now(timezone.utc).isoformat()
            with self._lock:
                if result.returncode == 0:
                    self._state = RetrainState(
                        status="success",
                        started_at=self._state.started_at,
                        finished_at=finished_at,
                        message="Модель переобучена и зарегистрирована в MLflow",
                        exit_code=0,
                    )
                    set_retrain_status(2)
                    if self.on_success:
                        try:
                            self.on_success()
                        except Exception as exc:
                            logger.exception(
                                "Model reload after retrain failed: %s", exc
                            )
                            self._state.message += f" (reload: {exc})"
                else:
                    self._state = RetrainState(
                        status="failed",
                        started_at=self._state.started_at,
                        finished_at=finished_at,
                        message=(
                            result.stderr[-500:]
                            if result.stderr
                            else "Ошибка переобучения"
                        ),
                        exit_code=result.returncode,
                    )
                    set_retrain_status(3)
        except Exception as exc:
            logger.exception("Retrain failed")
            with self._lock:
                self._state = RetrainState(
                    status="failed",
                    started_at=self._state.started_at,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    message=str(exc),
                )
                set_retrain_status(3)
