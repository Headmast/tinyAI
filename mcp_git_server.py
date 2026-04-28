#!/usr/bin/env python3
"""
MCP-сервер для работы с Git-контекстом проекта.

Протокол: Model Context Protocol (MCP) v2024-11-05
Транспорт: JSON-RPC 2.0 по stdout/stdin (stdio transport)

Доступные инструменты (tools):
  - git_current_branch  — текущая git-ветка
  - git_status          — изменённые файлы (git status --short)
  - git_diff_summary    — статистика изменений (git diff --stat)
  - git_log_short       — последние коммиты (git log --oneline)
  - git_list_files      — список отслеживаемых файлов

Запуск (через MCP-клиент):
    echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python mcp_git_server.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ══════════════════════════════════════════════════════════════════════════════
# Реестр инструментов
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "git_current_branch",
        "description": "Возвращает имя текущей git-ветки проекта.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "git_status",
        "description": "Возвращает список изменённых файлов (git status --short). Показывает modified, added, deleted файлы.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "git_diff_summary",
        "description": "Возвращает статистику изменений (git diff --stat): какие файлы изменены, сколько строк добавлено/удалено.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "staged": {
                    "type": "boolean",
                    "description": "Если true — показывает diff для staged (закоммиченных) изменений"
                }
            },
            "required": []
        }
    },
    {
        "name": "git_log_short",
        "description": "Возвращает последние N коммитов в формате oneline.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Количество коммитов (по умолчанию 10)"
                }
            },
            "required": []
        }
    },
    {
        "name": "git_list_files",
        "description": "Возвращает список отслеживаемых файлов проекта (git ls-files).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Фильтр по паттерну (например '*.py' для только Python-файлов)"
                }
            },
            "required": []
        }
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Реализация инструментов
# ══════════════════════════════════════════════════════════════════════════════

def _run_git(*args: str, timeout: int = 10) -> str:
    """Выполняет git-команду и возвращает stdout. Безопасно — без shell=True."""
    try:
        result = subprocess.run(
            ["git"] + list(args),
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            return f"Ошибка git: {stderr}" if stderr else f"git завершился с кодом {result.returncode}"
        return result.stdout.strip()
    except FileNotFoundError:
        return "Ошибка: git не найден в системе."
    except subprocess.TimeoutExpired:
        return "Ошибка: git команда превысила таймаут."


def tool_git_current_branch(args: dict) -> str:
    """Текущая ветка."""
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    return f"Текущая ветка: {branch}"


def tool_git_status(args: dict) -> str:
    """Статус рабочей директории."""
    output = _run_git("status", "--short")
    if not output:
        return "Рабочая директория чистая — нет изменений."
    count = len(output.strip().splitlines())
    return f"Изменённые файлы ({count}):\n{output}"


def tool_git_diff_summary(args: dict) -> str:
    """Статистика diff."""
    cmd_args = ["diff", "--stat"]
    if args.get("staged"):
        cmd_args.append("--staged")
    output = _run_git(*cmd_args)
    if not output:
        label = "staged" if args.get("staged") else "unstaged"
        return f"Нет {label} изменений."
    return f"Изменения:\n{output}"


def tool_git_log_short(args: dict) -> str:
    """Последние коммиты."""
    count = min(args.get("count", 10), 50)  # ограничение
    output = _run_git("log", f"--oneline", f"-{count}")
    if not output:
        return "История коммитов пуста."
    return f"Последние {count} коммитов:\n{output}"


def tool_git_list_files(args: dict) -> str:
    """Список отслеживаемых файлов."""
    pattern = args.get("pattern", "")
    if pattern:
        output = _run_git("ls-files", pattern)
    else:
        output = _run_git("ls-files")
    if not output:
        return "Нет отслеживаемых файлов."
    count = len(output.strip().splitlines())
    return f"Отслеживаемые файлы ({count}):\n{output}"


# ══════════════════════════════════════════════════════════════════════════════
# JSON-RPC / MCP dispatch
# ══════════════════════════════════════════════════════════════════════════════

TOOL_HANDLERS = {
    "git_current_branch": tool_git_current_branch,
    "git_status": tool_git_status,
    "git_diff_summary": tool_git_diff_summary,
    "git_log_short": tool_git_log_short,
    "git_list_files": tool_git_list_files,
}


def send(obj: dict) -> None:
    """Отправляет JSON-RPC сообщение в stdout (одна строка)."""
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def make_error(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_request(req: dict) -> None:
    """Обрабатывает входящий JSON-RPC запрос по правилам MCP-протокола."""
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {})

    # ── initialize ────────────────────────────────────────────────────────────
    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "tinyai-git-server",
                    "version": "1.0.0"
                }
            }
        })

    # ── notifications/initialized ─────────────────────────────────────────────
    elif method == "notifications/initialized":
        pass

    # ── tools/list ───────────────────────────────────────────────────────────
    elif method == "tools/list":
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS}
        })

    # ── tools/call ───────────────────────────────────────────────────────────
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
            "result": {
                "content": [{"type": "text", "text": result_text}]
            }
        })

    # ── неизвестный метод ─────────────────────────────────────────────────────
    else:
        if req_id is not None:
            send(make_error(req_id, -32601, f"Неизвестный метод: {method}"))


# ══════════════════════════════════════════════════════════════════════════════
# Главный цикл
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Читает JSON-RPC запросы из stdin и пишет ответы в stdout."""
    print(f"[MCP git server] TinyAI Git Server запущен (pid={os.getpid()})", file=sys.stderr)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as e:
            send(make_error(None, -32700, f"Parse error: {e}"))
            continue

        print(f"[MCP git server] ← {req.get('method', '?')} (id={req.get('id')})", file=sys.stderr)
        handle_request(req)


if __name__ == "__main__":
    main()
