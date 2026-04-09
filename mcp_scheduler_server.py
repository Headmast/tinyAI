"""
MCP-сервер планировщика TinyAI (День 18).

Протокол: Model Context Protocol (MCP) v2024-11-05
Транспорт: JSON-RPC 2.0 по stdout/stdin (stdio transport)

Доступные инструменты:
  - scheduler_add_task       — добавить задачу в планировщик
  - scheduler_list_tasks     — список всех задач с расписанием
  - scheduler_remove_task    — деактивировать задачу по имени
  - scheduler_get_summary    — последняя агрегированная сводка активности
  - scheduler_list_snapshots — снимки истории чата из БД
  - scheduler_list_backups   — список резервных копий
  - scheduler_trigger_task   — немедленно запустить задачу
  - scheduler_get_task_runs  — история выполнений задачи
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from scheduler.db import (
    add_task,
    deactivate_task,
    get_all_backups,
    get_all_tasks,
    get_latest_summary,
    get_recent_snapshots,
    get_task_by_name,
    get_task_runs,
    init_db,
    get_connection,
)
from scheduler.tasks import (
    run_chat_backup,
    run_chat_collector,
    run_reminder,
    run_summary_generator,
)
from scheduler.scheduler import TASK_FUNCTIONS

init_db()

# ══════════════════════════════════════════════════════════════════════════════
# Реестр инструментов
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "scheduler_add_task",
        "description": (
            "Добавляет периодическую задачу в планировщик. "
            "Типы задач: chat_collector (сбор снимков истории чата), "
            "chat_backup (резервное копирование логов), "
            "summary_generator (генерация сводки активности), "
            "reminder (напоминание с текстом)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Уникальное имя задачи (например 'daily_backup')",
                },
                "task_type": {
                    "type": "string",
                    "enum": [
                        "chat_collector",
                        "chat_backup",
                        "summary_generator",
                        "reminder",
                    ],
                    "description": "Тип задачи",
                },
                "interval_seconds": {
                    "type": "integer",
                    "description": "Интервал выполнения в секундах (минимум 60). Например: 3600 = каждый час.",
                },
                "payload": {
                    "type": "string",
                    "description": "JSON-строка с параметрами (для reminder: '{\"message\": \"текст\"}'). Опционально.",
                },
            },
            "required": ["name", "task_type", "interval_seconds"],
        },
    },
    {
        "name": "scheduler_list_tasks",
        "description": "Возвращает список всех задач в планировщике с расписанием и статусом.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "scheduler_remove_task",
        "description": "Деактивирует задачу по имени (история выполнений сохраняется).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Имя задачи для деактивации",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "scheduler_get_summary",
        "description": "Возвращает последнюю агрегированную сводку активности чатов.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "scheduler_list_snapshots",
        "description": "Возвращает последние снимки истории чата, собранные планировщиком.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Максимальное число записей (по умолчанию 20)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "scheduler_list_backups",
        "description": "Возвращает список всех резервных копий чат-логов.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "scheduler_trigger_task",
        "description": "Немедленно выполняет задачу по имени, не дожидаясь расписания.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Имя задачи для немедленного запуска",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "scheduler_get_task_runs",
        "description": "Возвращает историю последних выполнений задачи.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Имя задачи",
                },
                "limit": {
                    "type": "integer",
                    "description": "Максимальное число записей (по умолчанию 10)",
                },
            },
            "required": ["name"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Реализация инструментов
# ══════════════════════════════════════════════════════════════════════════════

def tool_scheduler_add_task(args: dict) -> str:
    name = args.get("name", "").strip()
    task_type = args.get("task_type", "")
    interval_seconds = args.get("interval_seconds", 0)
    payload_raw = args.get("payload", "{}")

    if not name:
        return "Ошибка: name обязателен."
    if task_type not in TASK_FUNCTIONS:
        return f"Ошибка: неизвестный task_type '{task_type}'."
    if interval_seconds < 60:
        return "Ошибка: interval_seconds должен быть >= 60."

    # Защита: имя не должно содержать опасных символов
    if not all(c.isalnum() or c in "_-" for c in name):
        return "Ошибка: name может содержать только буквы, цифры, _ и -."

    try:
        payload = json.loads(payload_raw) if payload_raw else {}
    except json.JSONDecodeError:
        return "Ошибка: payload должен быть валидным JSON."

    # Проверяем, нет ли уже задачи с таким именем
    existing = get_task_by_name(name)
    if existing:
        if existing["is_active"]:
            return f"Задача '{name}' уже существует и активна."
        # Реактивируем деактивированную
        with get_connection() as conn:
            conn.execute(
                """UPDATE scheduled_tasks
                   SET is_active = 1, interval_seconds = ?, next_run = ?, payload = ?
                   WHERE name = ?""",
                (
                    interval_seconds,
                    datetime.now().isoformat(),
                    json.dumps(payload),
                    name,
                ),
            )
        return f"Задача '{name}' реактивирована (интервал {interval_seconds}с)."

    task_id = add_task(name, task_type, interval_seconds, payload, run_immediately=True)
    return (
        f"Задача добавлена: '{name}' (тип={task_type}, "
        f"интервал={interval_seconds}с, id={task_id})."
    )


def tool_scheduler_list_tasks(args: dict) -> str:
    tasks = get_all_tasks()
    if not tasks:
        return "Задач в планировщике нет."

    lines = [f"Задач в планировщике: {len(tasks)}", ""]
    for t in tasks:
        status = "✓ активна" if t["is_active"] else "✗ неактивна"
        last = t["last_run"][:19] if t["last_run"] else "—"
        nxt = t["next_run"][:19] if t["next_run"] else "—"
        lines.append(
            f"  [{t['id']}] {t['name']}  ({t['task_type']})  {status}\n"
            f"       Интервал: {t['interval_seconds']}с | "
            f"Последний: {last} | Следующий: {nxt}"
        )
    return "\n".join(lines)


def tool_scheduler_remove_task(args: dict) -> str:
    name = args.get("name", "").strip()
    if not name:
        return "Ошибка: name обязателен."

    if deactivate_task(name):
        return f"Задача '{name}' деактивирована."
    return f"Задача '{name}' не найдена."


def tool_scheduler_get_summary(args: dict) -> str:
    summary = get_latest_summary()
    if not summary:
        return (
            "Сводки ещё нет. Запустите задачу summary_generator или подождите "
            "планировщика."
        )
    ts = summary["generated_at"][:19]
    return (
        f"Последняя сводка ({ts}):\n\n"
        f"{summary['summary_text']}\n\n"
        f"Всего сообщений: {summary['total_messages']}, "
        f"токенов: {summary['total_tokens']:,}"
    )


def tool_scheduler_list_snapshots(args: dict) -> str:
    limit = args.get("limit", 20)
    snapshots = get_recent_snapshots(limit=limit)
    if not snapshots:
        return "Снимков истории чата нет. Запустите chat_collector."

    lines = [f"Снимков истории чата: {len(snapshots)}", ""]
    for s in snapshots:
        ts = s["snapshot_at"][:19]
        lines.append(
            f"  [{ts}] {s['log_file']}  "
            f"({s['message_count']} сообщений, {s['total_tokens']:,} токенов)"
        )
    return "\n".join(lines)


def tool_scheduler_list_backups(args: dict) -> str:
    backups = get_all_backups()
    if not backups:
        return "Резервных копий нет. Запустите chat_backup."

    lines = [f"Резервных копий: {len(backups)}", ""]
    for b in backups:
        ts = b["backup_at"][:19]
        path = Path(b["backup_path"]).name
        lines.append(
            f"  [{ts}] {path}  "
            f"({b['files_count']} файлов, {b['total_size_kb']} KB)"
        )
    return "\n".join(lines)


def tool_scheduler_trigger_task(args: dict) -> str:
    name = args.get("name", "").strip()
    if not name:
        return "Ошибка: name обязателен."

    task = get_task_by_name(name)
    if not task:
        return f"Задача '{name}' не найдена."
    if not task["is_active"]:
        return f"Задача '{name}' деактивирована."

    task_type = task["task_type"]
    payload = json.loads(task.get("payload") or "{}")
    fn = TASK_FUNCTIONS.get(task_type)
    if fn is None:
        return f"Ошибка: неизвестный тип задачи '{task_type}'."

    try:
        result = fn(payload)
    except Exception as e:
        return f"Ошибка выполнения задачи '{name}': {e}"

    # Обновляем last_run и next_run
    run_at = datetime.now().isoformat()
    next_run = (
        datetime.now() + timedelta(seconds=task["interval_seconds"])
    ).isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE scheduled_tasks SET last_run = ?, next_run = ? WHERE name = ?",
            (run_at, next_run, name),
        )
        conn.execute(
            """INSERT INTO task_runs (task_id, run_at, status, result_text)
               VALUES (?, ?, 'success', ?)""",
            (task["id"], run_at, result[:2000]),
        )

    return f"Задача '{name}' выполнена:\n{result}"


def tool_scheduler_get_task_runs(args: dict) -> str:
    name = args.get("name", "").strip()
    limit = args.get("limit", 10)

    if not name:
        return "Ошибка: name обязателен."

    task = get_task_by_name(name)
    if not task:
        return f"Задача '{name}' не найдена."

    runs = get_task_runs(task["id"], limit=limit)
    if not runs:
        return f"Задача '{name}' ещё не запускалась."

    lines = [f"История выполнений задачи '{name}' (последние {len(runs)}):", ""]
    for r in runs:
        status_icon = "✓" if r["status"] == "success" else "✗"
        ts = r["run_at"][:19]
        result_preview = (r["result_text"] or "")[:120]
        lines.append(f"  {status_icon} [{ts}] {result_preview}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Диспетчер инструментов
# ══════════════════════════════════════════════════════════════════════════════

TOOL_HANDLERS = {
    "scheduler_add_task": tool_scheduler_add_task,
    "scheduler_list_tasks": tool_scheduler_list_tasks,
    "scheduler_remove_task": tool_scheduler_remove_task,
    "scheduler_get_summary": tool_scheduler_get_summary,
    "scheduler_list_snapshots": tool_scheduler_list_snapshots,
    "scheduler_list_backups": tool_scheduler_list_backups,
    "scheduler_trigger_task": tool_scheduler_trigger_task,
    "scheduler_get_task_runs": tool_scheduler_get_task_runs,
}


# ══════════════════════════════════════════════════════════════════════════════
# JSON-RPC / MCP dispatch
# ══════════════════════════════════════════════════════════════════════════════

def send(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def make_error(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_request(req: dict) -> None:
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "tinyai-scheduler-server",
                    "version": "18.0.0",
                },
            },
        })

    elif method == "notifications/initialized":
        pass  # уведомление, ответ не нужен

    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            send(make_error(req_id, -32601, f"Инструмент не найден: {tool_name}"))
            return
        try:
            result_text = handler(tool_args)
        except Exception as exc:
            send(make_error(req_id, -32603, f"Ошибка выполнения: {exc}"))
            return
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": [{"type": "text", "text": result_text}]},
        })

    else:
        if req_id is not None:
            send(make_error(req_id, -32601, f"Неизвестный метод: {method}"))


# ══════════════════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(
        f"[MCP scheduler] TinyAI Scheduler Server запущен (pid={os.getpid()})",
        file=sys.stderr,
    )

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as e:
            send(make_error(None, -32700, f"Parse error: {e}"))
            continue

        print(
            f"[MCP scheduler] ← {req.get('method', '?')} (id={req.get('id')})",
            file=sys.stderr,
        )
        handle_request(req)


if __name__ == "__main__":
    main()
