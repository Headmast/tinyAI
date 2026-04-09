"""
SQLite-хранилище контента для MCP-пайплайна (День 19).

Схема:
  content_items   — сгенерированный контент (посты, статьи)
  pipeline_runs   — история запусков пайплайна
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "pipeline_data" / "content.db"


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
            CREATE TABLE IF NOT EXISTS content_items (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                title         TEXT    NOT NULL,
                content       TEXT    NOT NULL,
                summary       TEXT,
                platform      TEXT    NOT NULL,
                format        TEXT    DEFAULT 'plain',
                tags          TEXT    DEFAULT '[]',
                source_query  TEXT,
                pipeline_run  TEXT,
                status        TEXT    DEFAULT 'draft',
                created_at    TEXT    NOT NULL,
                published_at  TEXT,
                word_count    INTEGER DEFAULT 0,
                char_count    INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id           TEXT    PRIMARY KEY,
                steps_json   TEXT    NOT NULL,
                results_json TEXT,
                status       TEXT    DEFAULT 'running',
                error        TEXT,
                started_at   TEXT    NOT NULL,
                finished_at  TEXT,
                total_ms     INTEGER
            );
        """)


# ── content_items ─────────────────────────────────────────────────────────────

def save_content(
    title: str,
    content: str,
    platform: str,
    *,
    summary: Optional[str] = None,
    fmt: str = "plain",
    tags: Optional[List[str]] = None,
    source_query: Optional[str] = None,
    pipeline_run: Optional[str] = None,
    status: str = "draft",
) -> int:
    """Сохраняет контент и возвращает ID записи."""
    now = datetime.now().isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO content_items
               (title, content, summary, platform, format, tags,
                source_query, pipeline_run, status, created_at,
                word_count, char_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                title,
                content,
                summary,
                platform,
                fmt,
                json.dumps(tags or [], ensure_ascii=False),
                source_query,
                pipeline_run,
                status,
                now,
                len(content.split()),
                len(content),
            ),
        )
        return cursor.lastrowid


def get_content(content_id: int) -> Optional[Dict[str, Any]]:
    """Загружает контент по ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM content_items WHERE id = ?", (content_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["tags"] = json.loads(d.get("tags") or "[]")
        return d


def list_content(
    limit: int = 20,
    platform: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Возвращает список контента с фильтрацией."""
    clauses = []
    params: list = []
    if platform:
        clauses.append("platform = ?")
        params.append(platform)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM content_items{where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d.get("tags") or "[]")
            result.append(d)
        return result


def update_status(content_id: int, status: str) -> bool:
    """Обновляет статус контента (draft → published → archived)."""
    extras = ""
    params: list = [status]
    if status == "published":
        extras = ", published_at = ?"
        params.append(datetime.now().isoformat())
    params.append(content_id)
    with get_connection() as conn:
        affected = conn.execute(
            f"UPDATE content_items SET status = ?{extras} WHERE id = ?",
            params,
        ).rowcount
        return affected > 0


# ── pipeline_runs ─────────────────────────────────────────────────────────────

def create_pipeline_run(run_id: str, steps: List[Dict[str, Any]]) -> None:
    """Регистрирует запуск пайплайна."""
    now = datetime.now().isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO pipeline_runs (id, steps_json, status, started_at)
               VALUES (?, ?, 'running', ?)""",
            (run_id, json.dumps(steps, ensure_ascii=False), now),
        )


def finish_pipeline_run(
    run_id: str,
    results: List[Dict[str, Any]],
    status: str = "completed",
    error: Optional[str] = None,
    total_ms: Optional[int] = None,
) -> None:
    """Завершает запуск пайплайна."""
    now = datetime.now().isoformat()
    with get_connection() as conn:
        conn.execute(
            """UPDATE pipeline_runs
               SET results_json = ?, status = ?, error = ?,
                   finished_at = ?, total_ms = ?
               WHERE id = ?""",
            (
                json.dumps(results, ensure_ascii=False),
                status,
                error,
                now,
                total_ms,
                run_id,
            ),
        )


def get_pipeline_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Загружает информацию о запуске пайплайна."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["steps_json"] = json.loads(d.get("steps_json") or "[]")
        d["results_json"] = json.loads(d.get("results_json") or "[]")
        return d


def list_pipeline_runs(limit: int = 20) -> List[Dict[str, Any]]:
    """Возвращает последние запуски пайплайна."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["steps_json"] = json.loads(d.get("steps_json") or "[]")
            d["results_json"] = json.loads(d.get("results_json") or "[]")
            result.append(d)
        return result
