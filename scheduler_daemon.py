"""
TinyAI Scheduler Daemon (День 18).

Запускает фоновый планировщик, который работает 24/7 и периодически:
  - Собирает снимки истории чатов (каждые 5 минут по умолчанию)
  - Создаёт резервные копии логов  (каждый час)
  - Генерирует сводку активности  (каждые 30 минут)

Использование:
  python scheduler_daemon.py              # запуск в режиме 24/7
  python scheduler_daemon.py --once       # один цикл и выход (отладка)
  python scheduler_daemon.py --interval 10 # проверять задачи каждые 10 секунд
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from scheduler.db import get_connection, init_db
from scheduler.scheduler import BackgroundScheduler

# ── Задачи по умолчанию ────────────────────────────────────────────────────
DEFAULT_TASKS = [
    {
        "name": "chat_collector_5min",
        "task_type": "chat_collector",
        "interval_seconds": 300,       # каждые 5 минут
        "description": "Сбор снимков истории чата",
    },
    {
        "name": "chat_backup_hourly",
        "task_type": "chat_backup",
        "interval_seconds": 3600,      # каждый час
        "description": "Резервное копирование логов",
    },
    {
        "name": "summary_generator_30min",
        "task_type": "summary_generator",
        "interval_seconds": 1800,      # каждые 30 минут
        "description": "Генерация сводки активности",
    },
]


def setup_default_tasks() -> None:
    """Создаёт задачи по умолчанию если они ещё не существуют."""
    init_db()
    now = datetime.now()

    with get_connection() as conn:
        for cfg in DEFAULT_TASKS:
            existing = conn.execute(
                "SELECT id, is_active FROM scheduled_tasks WHERE name = ?",
                (cfg["name"],),
            ).fetchone()

            if existing is None:
                conn.execute(
                    """INSERT INTO scheduled_tasks
                       (name, task_type, interval_seconds,
                        next_run, payload, is_active, created_at)
                       VALUES (?, ?, ?, ?, '{}', 1, ?)""",
                    (
                        cfg["name"],
                        cfg["task_type"],
                        cfg["interval_seconds"],
                        now.isoformat(),   # запустить сразу при старте
                        now.isoformat(),
                    ),
                )
                print(
                    f"[Daemon] + Создана задача: {cfg['name']} "
                    f"({cfg['description']}, каждые {cfg['interval_seconds']}с)",
                    flush=True,
                )
            else:
                print(
                    f"[Daemon]   Задача уже существует: {cfg['name']} "
                    f"(active={existing['is_active']})",
                    flush=True,
                )


def print_status() -> None:
    """Выводит текущий статус всех задач."""
    with get_connection() as conn:
        tasks = conn.execute(
            "SELECT name, task_type, last_run, next_run, is_active FROM scheduled_tasks"
        ).fetchall()

    print("\n[Daemon] Текущий статус задач:", flush=True)
    for t in tasks:
        status = "✓" if t["is_active"] else "✗"
        last = t["last_run"][:19] if t["last_run"] else "—"
        nxt = t["next_run"][:19] if t["next_run"] else "—"
        print(
            f"  {status} {t['name']}  last={last}  next={nxt}",
            flush=True,
        )
    print(flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TinyAI Scheduler Daemon — День 18"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Выполнить один цикл всех задач и выйти",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Интервал проверки задач в секундах (по умолчанию 30)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Показать статус задач и выйти",
    )
    args = parser.parse_args()

    print(
        f"[Daemon] TinyAI Scheduler Daemon v18 | "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )

    setup_default_tasks()

    if args.status:
        print_status()
        return

    scheduler = BackgroundScheduler(check_interval=args.interval)

    if args.once:
        print("[Daemon] Режим --once: выполняю один цикл...", flush=True)
        scheduler.run_once()
        print("[Daemon] Цикл завершён. Выход.", flush=True)
        return

    # ── Запуск в режиме 24/7 ─────────────────────────────────────────────
    scheduler.start()
    print_status()

    def _handle_signal(sig, _frame) -> None:
        print(f"\n[Daemon] Сигнал {sig} получен, завершаю...", flush=True)
        scheduler.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(
        "[Daemon] Работает 24/7. Нажмите Ctrl+C для остановки.",
        flush=True,
    )

    try:
        while scheduler.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()
