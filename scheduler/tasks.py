"""
Реализации задач для планировщика TinyAI (День 18).

Задачи:
  run_chat_collector    — сканирует logs/ и создаёт снимки истории чата в БД
  run_chat_backup       — копирует все логи в chat_backups/<timestamp>/
  run_summary_generator — агрегирует снимки и записывает сводку в БД
  run_reminder          — фиксирует напоминание (возвращает текст)
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .db import get_connection

BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
BACKUPS_DIR = BASE_DIR / "chat_backups"


# ══════════════════════════════════════════════════════════════════════════════
# Сбор снимков истории чата
# ══════════════════════════════════════════════════════════════════════════════

def run_chat_collector() -> str:
    """
    Сканирует все conversation_*.json в logs/ и создаёт снимки в БД.
    Пропускает файлы у которых снимок с таким же числом сообщений уже есть.
    """
    if not LOGS_DIR.exists():
        return "Директория logs/ не найдена."

    snapshots_created = 0
    files_scanned = 0

    with get_connection() as conn:
        for log_file in sorted(LOGS_DIR.glob("conversation_*.json")):
            files_scanned += 1
            try:
                with open(log_file, encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, list):
                    continue

                message_count = len(data)
                total_tokens = sum(
                    m.get("usage", {}).get("total_tokens", 0) for m in data
                )

                # Пропускаем если снимок с таким размером уже есть
                existing = conn.execute(
                    "SELECT id FROM chat_snapshots WHERE log_file = ? AND message_count = ?",
                    (log_file.name, message_count),
                ).fetchone()

                if existing:
                    continue

                # Составляем краткое описание
                first_msg = (data[0].get("user_input", "") if data else "")[:80]
                last_msg = (data[-1].get("user_input", "") if data else "")[:80]
                summary = f"Первое: {first_msg} | Последнее: {last_msg}"

                conn.execute(
                    """INSERT INTO chat_snapshots
                       (snapshot_at, log_file, message_count, total_tokens, summary)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        datetime.now().isoformat(),
                        log_file.name,
                        message_count,
                        total_tokens,
                        summary,
                    ),
                )
                snapshots_created += 1

            except (json.JSONDecodeError, OSError):
                continue

    return (
        f"Сканировано файлов: {files_scanned}. "
        f"Новых снимков создано: {snapshots_created}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Резервное копирование логов
# ══════════════════════════════════════════════════════════════════════════════

def run_chat_backup() -> str:
    """
    Копирует все файлы из logs/ в chat_backups/backup_<timestamp>/.
    Записывает метаданные копии в таблицу chat_backups.
    """
    if not LOGS_DIR.exists():
        return "Директория logs/ не найдена, нечего копировать."

    files = list(LOGS_DIR.glob("*.json"))
    if not files:
        return "В директории logs/ нет файлов для резервного копирования."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUPS_DIR / f"backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    total_size = 0
    for f in files:
        shutil.copy2(f, backup_dir / f.name)
        total_size += f.stat().st_size

    total_size_kb = round(total_size / 1024, 1)

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO chat_backups
               (backup_at, backup_path, files_count, total_size_kb)
               VALUES (?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                str(backup_dir),
                len(files),
                total_size_kb,
            ),
        )

    return (
        f"Резервная копия создана: {backup_dir.name} "
        f"({len(files)} файлов, {total_size_kb} KB)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Генерация сводки
# ══════════════════════════════════════════════════════════════════════════════

def run_summary_generator() -> str:
    """
    Агрегирует данные из таблицы chat_snapshots и записывает сводку
    в таблицу periodic_summaries.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT log_file, message_count, total_tokens, snapshot_at
               FROM chat_snapshots
               ORDER BY snapshot_at DESC
               LIMIT 100"""
        ).fetchall()

    if not rows:
        return "Нет снимков истории для генерации сводки."

    total_messages = sum(r["message_count"] for r in rows)
    total_tokens = sum(r["total_tokens"] for r in rows)
    unique_files = len({r["log_file"] for r in rows})

    period_start = min(r["snapshot_at"] for r in rows)[:19]
    period_end = max(r["snapshot_at"] for r in rows)[:19]

    # Топ-файлов по числу сообщений (убираем дубликаты по имени)
    file_stats: Dict[str, int] = {}
    for r in rows:
        fname = r["log_file"]
        if file_stats.get(fname, 0) < r["message_count"]:
            file_stats[fname] = r["message_count"]

    top_files = sorted(file_stats.items(), key=lambda x: x[1], reverse=True)[:5]

    top_lines = "\n".join(
        f"    • {fname}: {cnt} сообщений" for fname, cnt in top_files
    )

    summary_text = (
        f"=== Сводка активности чата ===\n"
        f"  Файлов логов:      {unique_files}\n"
        f"  Всего сообщений:   {total_messages}\n"
        f"  Всего токенов:     {total_tokens:,}\n"
        f"  Период снимков:    {period_start} — {period_end}\n"
        f"\n  Топ файлов по активности:\n{top_lines}"
    )

    now = datetime.now().isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO periodic_summaries
               (generated_at, period_start, period_end,
                summary_text, total_messages, total_tokens)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (now, period_start, period_end, summary_text, total_messages, total_tokens),
        )

    return summary_text


# ══════════════════════════════════════════════════════════════════════════════
# Напоминание
# ══════════════════════════════════════════════════════════════════════════════

def run_reminder(payload: Dict[str, Any]) -> str:
    """Выводит напоминание из payload['message']."""
    message = payload.get("message", "Напоминание без текста!")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"⏰ [{ts}] НАПОМИНАНИЕ: {message}"
