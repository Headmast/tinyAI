"""
Демонстрация День 18: Планировщик и фоновые задачи.

Что показывает демонстрация:
  1. Инициализация планировщика и SQLite-хранилища
  2. Добавление задач по расписанию
  3. Немедленный запуск задач (сбор истории, резервная копия, сводка)
  4. Диалог с MCPSchedulerAgent

Запуск:
  python demos/demo_day18_scheduler.py
  python demos/demo_day18_scheduler.py --no-llm   # без LLM (только планировщик)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv

from scheduler.db import init_db, get_all_tasks, get_recent_snapshots, get_all_backups
from scheduler.db import get_latest_summary, get_connection
from scheduler.scheduler import BackgroundScheduler

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# Раздел 1: Демонстрация планировщика без LLM
# ══════════════════════════════════════════════════════════════════════════════

def demo_scheduler_direct() -> None:
    """Прямая демонстрация планировщика и задач без LLM."""
    print("=" * 60)
    print("День 18: Планировщик и фоновые задачи")
    print("=" * 60)

    # Инициализируем БД
    init_db()
    print("\n[1] SQLite-хранилище инициализировано.")

    # Добавляем задачи по расписанию
    now = datetime.now()
    with get_connection() as conn:
        # Задача 1: сбор истории чата каждые 5 минут
        existing = conn.execute(
            "SELECT id FROM scheduled_tasks WHERE name = 'demo_chat_collector'",
        ).fetchone()
        if not existing:
            conn.execute(
                """INSERT INTO scheduled_tasks
                   (name, task_type, interval_seconds, next_run, payload, is_active, created_at)
                   VALUES (?, ?, ?, ?, '{}', 1, ?)""",
                (
                    "demo_chat_collector",
                    "chat_collector",
                    300,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )

        # Задача 2: резервное копирование каждый час
        existing = conn.execute(
            "SELECT id FROM scheduled_tasks WHERE name = 'demo_chat_backup'",
        ).fetchone()
        if not existing:
            conn.execute(
                """INSERT INTO scheduled_tasks
                   (name, task_type, interval_seconds, next_run, payload, is_active, created_at)
                   VALUES (?, ?, ?, ?, '{}', 1, ?)""",
                (
                    "demo_chat_backup",
                    "chat_backup",
                    3600,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )

        # Задача 3: сводка каждые 30 минут
        existing = conn.execute(
            "SELECT id FROM scheduled_tasks WHERE name = 'demo_summary_generator'",
        ).fetchone()
        if not existing:
            conn.execute(
                """INSERT INTO scheduled_tasks
                   (name, task_type, interval_seconds, next_run, payload, is_active, created_at)
                   VALUES (?, ?, ?, ?, '{}', 1, ?)""",
                (
                    "demo_summary_generator",
                    "summary_generator",
                    1800,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )

        # Задача 4: напоминание каждые 10 минут
        existing = conn.execute(
            "SELECT id FROM scheduled_tasks WHERE name = 'demo_reminder'",
        ).fetchone()
        if not existing:
            conn.execute(
                """INSERT INTO scheduled_tasks
                   (name, task_type, interval_seconds, next_run,
                    payload, is_active, created_at)
                   VALUES (?, ?, ?, ?, ?, 1, ?)""",
                (
                    "demo_reminder",
                    "reminder",
                    600,
                    now.isoformat(),
                    json.dumps({"message": "Проверь активность агентов TinyAI!"}),
                    now.isoformat(),
                ),
            )

    tasks = get_all_tasks()
    print(f"\n[2] Задач в планировщике: {len(tasks)}")
    for t in tasks:
        nxt = t["next_run"][:19] if t["next_run"] else "—"
        print(
            f"    • {t['name']}  ({t['task_type']}, "
            f"каждые {t['interval_seconds']}с)  следующий: {nxt}"
        )

    # Запускаем один цикл планировщика
    print("\n[3] Запускаю один цикл планировщика (run_once)...")
    sched = BackgroundScheduler(check_interval=10)
    sched.run_once()

    # Показываем результаты
    print("\n[4] Результаты после запуска:")

    snapshots = get_recent_snapshots(limit=10)
    print(f"\n  Снимки истории чата ({len(snapshots)} шт.):")
    for s in snapshots[:5]:
        ts = s["snapshot_at"][:19]
        print(
            f"    [{ts}] {s['log_file']}  "
            f"({s['message_count']} сообщений, {s['total_tokens']:,} токенов)"
        )

    backups = get_all_backups()
    print(f"\n  Резервные копии ({len(backups)} шт.):")
    for b in backups:
        ts = b["backup_at"][:19]
        bname = Path(b["backup_path"]).name
        print(
            f"    [{ts}] {bname} "
            f"({b['files_count']} файлов, {b['total_size_kb']} KB)"
        )

    summary = get_latest_summary()
    if summary:
        ts = summary["generated_at"][:19]
        print(f"\n  Последняя сводка [{ts}]:")
        for line in summary["summary_text"].split("\n"):
            print(f"    {line}")

    # Демонстрируем фоновый режим (3 секунды)
    print("\n[5] Запускаю фоновый планировщик на 3 секунды...")
    sched2 = BackgroundScheduler(check_interval=2)
    sched2.start()
    time.sleep(3)
    sched2.stop()
    print("    Фоновый планировщик остановлен.")


# ══════════════════════════════════════════════════════════════════════════════
# Раздел 2: Демонстрация с LLM-агентом
# ══════════════════════════════════════════════════════════════════════════════

DEMO_TASKS = [
    "Покажи мне все задачи в планировщике.",
    "Запусти немедленно задачу сбора истории чата (demo_chat_collector).",
    "Покажи последнюю сводку активности чатов.",
    "Какие резервные копии есть в системе?",
    (
        "Добавь новую задачу-напоминание с именем 'check_memory' "
        "с интервалом 3600 секунд и сообщением 'Время проверить память агентов!'"
    ),
    "Покажи историю выполнений задачи demo_chat_collector.",
]


def demo_with_llm() -> None:
    """Демонстрация MCPSchedulerAgent с LLM."""
    from openai import OpenAI
    from mcp_scheduler_agent import MCPSchedulerAgent

    api_key = os.getenv("CLOUD_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ CLOUD_API_KEY или OPENAI_API_KEY не найден в .env")
        sys.exit(1)

    base_url = os.getenv("CLOUD_BASE_URL", "https://foundation-models.api.cloud.ru/v1")
    model = os.getenv("CLOUD_MODEL", "zai-org/GLM-4.7")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    print("\n" + "=" * 60)
    print("День 18: MCPSchedulerAgent — диалог с LLM")
    print("=" * 60)

    with MCPSchedulerAgent(client=client, model=model, verbose=True) as agent:
        for i, task in enumerate(DEMO_TASKS, 1):
            print(f"\n{'─' * 60}")
            print(f"[Запрос {i}/{len(DEMO_TASKS)}] {task}")
            print("─" * 60)

            answer = agent.chat(task)
            print(f"\n[Ответ]:\n{answer}")
            time.sleep(1)

        # Итоги
        usage = agent.token_usage
        print(f"\n{'=' * 60}")
        print("Итоги демонстрации:")
        print(f"  Вызовов MCP-инструментов: {agent.tool_calls_total}")
        print(f"  Токенов (всего):           {usage['total_tokens']:,}")
        print(f"  Промпт / Ответ:            {usage['prompt_tokens']:,} / {usage['completion_tokens']:,}")


# ══════════════════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="День 18: Демо планировщика")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Только планировщик без LLM-агента",
    )
    args = parser.parse_args()

    demo_scheduler_direct()

    if not args.no_llm:
        demo_with_llm()
    else:
        print("\n[Режим --no-llm] LLM-агент пропущен.")


if __name__ == "__main__":
    main()
