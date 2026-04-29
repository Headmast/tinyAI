#!/usr/bin/env python3
"""
MCP-сервер для службы поддержки пользователей.

Протокол: Model Context Protocol (MCP) v2024-11-05
Транспорт: JSON-RPC 2.0 по stdout/stdin (stdio transport)

Доступные инструменты (tools):
  - get_user           — профиль пользователя по user_id
  - get_user_tickets   — тикеты пользователя
  - get_ticket         — полный тикет с сообщениями
  - search_tickets     — поиск тикетов по ключевому слову
  - get_active_tickets — все открытые/в работе тикеты

Запуск (через MCP-клиент):
    echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python mcp_support_server.py
"""

import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "support_data"

# ══════════════════════════════════════════════════════════════════════════════
# Загрузка данных
# ══════════════════════════════════════════════════════════════════════════════

def _load_json(filename: str) -> list:
    """Загружает JSON-файл из support_data/."""
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_users() -> list:
    return _load_json("users.json")


def _get_tickets() -> list:
    return _load_json("tickets.json")


# ══════════════════════════════════════════════════════════════════════════════
# Реестр инструментов
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "get_user",
        "description": "Возвращает профиль пользователя по user_id: имя, email, тариф, статус, дата регистрации.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Идентификатор пользователя (например 'user_1')"
                }
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "get_user_tickets",
        "description": "Возвращает список тикетов пользователя по user_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Идентификатор пользователя"
                }
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "get_ticket",
        "description": "Возвращает полный тикет с историей сообщений по ticket_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "Идентификатор тикета (например 'ticket_1')"
                }
            },
            "required": ["ticket_id"]
        }
    },
    {
        "name": "search_tickets",
        "description": "Поиск тикетов по ключевому слову в теме или сообщениях. Можно фильтровать по статусу.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Ключевое слово для поиска"
                },
                "status": {
                    "type": "string",
                    "description": "Фильтр по статусу: open, in_progress, resolved (необязательно)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_active_tickets",
        "description": "Возвращает все тикеты со статусом open или in_progress.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Реализация инструментов
# ══════════════════════════════════════════════════════════════════════════════

def tool_get_user(args: dict) -> str:
    """Профиль пользователя по user_id."""
    user_id = args.get("user_id", "")
    if not user_id:
        return "Ошибка: не указан user_id."
    users = _get_users()
    for user in users:
        if user["user_id"] == user_id:
            return json.dumps(user, ensure_ascii=False, indent=2)
    return f"Пользователь {user_id} не найден."


def tool_get_user_tickets(args: dict) -> str:
    """Тикеты пользователя."""
    user_id = args.get("user_id", "")
    if not user_id:
        return "Ошибка: не указан user_id."
    tickets = _get_tickets()
    user_tickets = [t for t in tickets if t["user_id"] == user_id]
    if not user_tickets:
        return f"У пользователя {user_id} нет тикетов."
    # Краткая информация (без полной переписки)
    summaries = []
    for t in user_tickets:
        summaries.append({
            "ticket_id": t["ticket_id"],
            "subject": t["subject"],
            "status": t["status"],
            "priority": t["priority"],
            "created_at": t["created_at"],
            "tags": t.get("tags", []),
            "messages_count": len(t.get("messages", [])),
        })
    return json.dumps(summaries, ensure_ascii=False, indent=2)


def tool_get_ticket(args: dict) -> str:
    """Полный тикет с историей сообщений."""
    ticket_id = args.get("ticket_id", "")
    if not ticket_id:
        return "Ошибка: не указан ticket_id."
    tickets = _get_tickets()
    for t in tickets:
        if t["ticket_id"] == ticket_id:
            return json.dumps(t, ensure_ascii=False, indent=2)
    return f"Тикет {ticket_id} не найден."


def tool_search_tickets(args: dict) -> str:
    """Поиск тикетов по ключевому слову."""
    query = args.get("query", "").lower()
    if not query:
        return "Ошибка: не указан запрос для поиска."
    status_filter = args.get("status", "")
    tickets = _get_tickets()

    results = []
    for t in tickets:
        if status_filter and t["status"] != status_filter:
            continue
        # Поиск в теме
        if query in t["subject"].lower():
            results.append(t)
            continue
        # Поиск в тегах
        if any(query in tag.lower() for tag in t.get("tags", [])):
            results.append(t)
            continue
        # Поиск в сообщениях
        for msg in t.get("messages", []):
            if query in msg.get("text", "").lower():
                results.append(t)
                break

    if not results:
        return f"Тикеты по запросу '{query}' не найдены."

    summaries = []
    for t in results:
        summaries.append({
            "ticket_id": t["ticket_id"],
            "user_id": t["user_id"],
            "subject": t["subject"],
            "status": t["status"],
            "priority": t["priority"],
            "tags": t.get("tags", []),
        })
    return json.dumps(summaries, ensure_ascii=False, indent=2)


def tool_get_active_tickets(args: dict) -> str:
    """Все открытые и в работе тикеты."""
    tickets = _get_tickets()
    active = [t for t in tickets if t["status"] in ("open", "in_progress")]
    if not active:
        return "Нет активных тикетов."
    summaries = []
    for t in active:
        summaries.append({
            "ticket_id": t["ticket_id"],
            "user_id": t["user_id"],
            "subject": t["subject"],
            "status": t["status"],
            "priority": t["priority"],
            "created_at": t["created_at"],
            "tags": t.get("tags", []),
        })
    return json.dumps(summaries, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# JSON-RPC / MCP dispatch
# ══════════════════════════════════════════════════════════════════════════════

TOOL_HANDLERS = {
    "get_user": tool_get_user,
    "get_user_tickets": tool_get_user_tickets,
    "get_ticket": tool_get_ticket,
    "search_tickets": tool_search_tickets,
    "get_active_tickets": tool_get_active_tickets,
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
                    "name": "tinyai-support-server",
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
    print(f"[MCP support server] TinyAI Support Server запущен (pid={os.getpid()})", file=sys.stderr)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            send(make_error(None, -32700, f"Parse error: {exc}"))
            continue
        handle_request(req)


if __name__ == "__main__":
    main()
