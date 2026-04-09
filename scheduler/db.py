"""
SQLite-хранилище для планировщика TinyAI (День 18).

Схема:
  scheduled_tasks   — задачи по расписанию (тип, интервал, следующий запуск)
  task_runs         — история выполнений задач
  chat_snapshots    — снимки состояния истории чатов
  periodic_summaries— агрегированные сводки активности
  chat_backups      — записи о резервных копиях логов
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "scheduler_data" / "scheduler.db"


def get_connection() -> sqlite3.Connection:
    """Открывает соединение с SQLite (создаёт файл если отсутствует)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Создаёт таблицы если они не существуют."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                name             TEXT    NOT NULL UNIQUE,
                task_type        TEXT    NOT NULL,
                interval_seconds INTEGER NOT NULL,
                next_run         TEXT    NOT NULL,
                last_run         TEXT,
                payload          TEXT    DEFAULT '{}',
                is_active        INTEGER DEFAULT 1,
                created_at       TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS task_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id     INTEGER NOT NULL,
                run_at      TEXT    NOT NULL,
                status      TEXT    NOT NULL,
                result_text TEXT,
                FOREIGN KEY (task_id) REFERENCES scheduled_tasks(id)
            );

            CREATE TABLE IF NOT EXISTS chat_snapshots (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_at   TEXT    NOT NULL,
                log_file      TEXT    NOT NULL,
                message_count INTEGER NOT NULL,
                total_tokens  INTEGER NOT NULL,
                summary       TEXT
            );

            CREATE TABLE IF NOT EXISTS periodic_summaries (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                generated_at   TEXT    NOT NULL,
                period_start   TEXT    NOT NULL,
                period_end     TEXT    NOT NULL,
                summary_text   TEXT    NOT NULL,
                total_messages INTEGER NOT NULL,
                total_tokens   INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_backups (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                backup_at     TEXT    NOT NULL,
                backup_path   TEXT    NOT NULL,
                files_count   INTEGER NOT NULL,
                total_size_kb REAL    NOT NULL
            );
        """)


# ── Вспомогательные функции ───────────────────────────────────────────────────

def add_task(name: str, task_type: str, interval_seconds: int,
             payload: Optional[dict] = None, run_immediately: bool = True) -> int:
    """
    Добавляет задачу в планировщик.
    При run_immediately=True задача запустится при ближайшей проверке.
    Возвращает id созданной задачи.
    """
    now = datetime.now()
    next_run = now if run_immediately else (
        __import__("datetime").datetime.now() +
        __import__("datetime").timedelta(seconds=interval_seconds)
    )
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO scheduled_tasks
               (name, task_type, interval_seconds, next_run, payload, is_active, created_at)
               VALUES (?, ?, ?, ?, ?, 1, ?)""",
            (
                name,
                task_type,
                interval_seconds,
                now.isoformat() if run_immediately else next_run.isoformat(),
                json.dumps(payload or {}),
                now.isoformat(),
            ),
        )
        return cursor.lastrowid


def get_task_by_name(name: str) -> Optional[dict]:
    """Ищет задачу по имени."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM scheduled_tasks WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None


def deactivate_task(name: str) -> bool:
    """Деактивирует задачу (не удаляет, сохраняет историю)."""
    with get_connection() as conn:
        affected = conn.execute(
            "UPDATE scheduled_tasks SET is_active = 0 WHERE name = ?", (name,)
        ).rowcount
        return affected > 0


def get_latest_summary() -> Optional[dict]:
    """Возвращает последнюю сводку."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM periodic_summaries ORDER BY generated_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def get_recent_snapshots(limit: int = 20) -> list:
    """Возвращает последние снимки истории чата."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_snapshots ORDER BY snapshot_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_backups() -> list:
    """Возвращает все записи о резервных копиях."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_backups ORDER BY backup_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_task_runs(task_id: int, limit: int = 10) -> list:
    """Возвращает историю выполнений задачи."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY run_at DESC LIMIT ?",
            (task_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_tasks() -> list:
    """Возвращает все задачи."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM scheduled_tasks ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
