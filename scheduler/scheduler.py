"""
Фоновый поток-планировщик TinyAI (День 18).

BackgroundScheduler — поднимает daemon-поток, который каждые N секунд
проверяет таблицу scheduled_tasks и запускает задачи с истёкшим next_run.

Использование:
    from scheduler.scheduler import BackgroundScheduler

    sched = BackgroundScheduler(check_interval=30)
    sched.start()   # поток — daemon, завершается вместе с процессом
    ...
    sched.stop()
"""

import json
import threading
import traceback
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

from .db import get_connection, init_db
from .tasks import (
    run_chat_backup,
    run_chat_collector,
    run_reminder,
    run_summary_generator,
)

# Реестр функций задач: task_type → callable(payload) → str
TASK_FUNCTIONS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "chat_collector":    lambda payload: run_chat_collector(),
    "chat_backup":       lambda payload: run_chat_backup(),
    "summary_generator": lambda payload: run_summary_generator(),
    "reminder":          run_reminder,
}


class BackgroundScheduler:
    """
    Поток-планировщик на основе threading.Thread (daemon=True).

    Параметры:
      check_interval — как часто (секунды) проверять таблицу задач
    """

    def __init__(self, check_interval: int = 30) -> None:
        self._check_interval = check_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

    # ── Публичный интерфейс ───────────────────────────────────────────────

    def start(self) -> None:
        """Запускает фоновый поток. Инициализирует БД при первом запуске."""
        init_db()
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="TinyAI-Scheduler",
            daemon=True,
        )
        self._thread.start()
        print(
            f"[Scheduler] Запущен (check_interval={self._check_interval}с)",
            flush=True,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Останавливает планировщик, ожидает завершения потока."""
        self._stop_event.set()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        print("[Scheduler] Остановлен.", flush=True)

    def is_running(self) -> bool:
        return (
            self._running
            and self._thread is not None
            and self._thread.is_alive()
        )

    def run_once(self) -> None:
        """Выполняет один цикл проверки без запуска потока (для тестов/демо)."""
        init_db()
        self._check_and_run_tasks()

    # ── Внутренние методы ─────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Основной цикл: проверяет задачи, ждёт check_interval секунд."""
        while not self._stop_event.is_set():
            try:
                self._check_and_run_tasks()
            except Exception:
                # Не позволяем ни одной ошибке убить поток
                traceback.print_exc()
            self._stop_event.wait(self._check_interval)

    def _check_and_run_tasks(self) -> None:
        """Выбирает из БД просроченные задачи и последовательно выполняет их."""
        now = datetime.now().isoformat()
        with get_connection() as conn:
            due = conn.execute(
                """SELECT * FROM scheduled_tasks
                   WHERE is_active = 1 AND next_run <= ?
                   ORDER BY next_run""",
                (now,),
            ).fetchall()

        for task in due:
            self._execute_task(dict(task))

    def _execute_task(self, task: dict) -> None:
        """Выполняет задачу и обновляет записи в БД."""
        task_id = task["id"]
        task_type = task["task_type"]
        payload: Dict[str, Any] = json.loads(task.get("payload") or "{}")

        run_at = datetime.now().isoformat()
        status = "success"
        result_text = ""

        fn = TASK_FUNCTIONS.get(task_type)
        if fn is None:
            status = "error"
            result_text = f"Неизвестный тип задачи: {task_type}"
        else:
            try:
                result_text = fn(payload)
                print(
                    f"[Scheduler] ✓ {task['name']}: {result_text[:120]}",
                    flush=True,
                )
            except Exception:
                status = "error"
                result_text = traceback.format_exc(limit=4)
                print(
                    f"[Scheduler] ✗ {task['name']}: {result_text[:200]}",
                    flush=True,
                )

        interval = task["interval_seconds"]
        next_run = (
            datetime.now() + timedelta(seconds=interval)
        ).isoformat()

        with get_connection() as conn:
            conn.execute(
                """UPDATE scheduled_tasks
                   SET last_run = ?, next_run = ?
                   WHERE id = ?""",
                (run_at, next_run, task_id),
            )
            conn.execute(
                """INSERT INTO task_runs (task_id, run_at, status, result_text)
                   VALUES (?, ?, ?, ?)""",
                (task_id, run_at, status, result_text[:2000]),
            )
