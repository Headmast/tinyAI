"""
Scheduler package — День 18: Планировщик и фоновые задачи.

Модули:
  db         — SQLite-хранилище задач, снимков истории и резервных копий
  tasks      — реализации задач (сбор, резервное копирование, сводка, напоминания)
  scheduler  — фоновый поток-планировщик
"""

from .db import init_db
from .scheduler import BackgroundScheduler
from .tasks import run_chat_collector, run_chat_backup, run_summary_generator, run_reminder

__all__ = [
    "init_db",
    "BackgroundScheduler",
    "run_chat_collector",
    "run_chat_backup",
    "run_summary_generator",
    "run_reminder",
]
