"""
MCP-сервер для управления локальными логами чатов и памяти TinyAI.

Протокол: Model Context Protocol (MCP) v2024-11-05
Транспорт: JSON-RPC 2.0 по stdout/stdin (stdio transport)

Как работает:
  1. Клиент запускает этот скрипт как подпроцесс
  2. Клиент отправляет JSON-RPC сообщения в stdin сервера
  3. Сервер отвечает JSON-RPC сообщениями в stdout
  4. Каждое сообщение — отдельная строка JSON (newline-delimited)

Доступные инструменты (tools):
  - list_logs        — список файлов логов разговоров
  - read_log         — прочитать конкретный лог
  - search_logs      — поискать текст по всем логам
  - list_memory      — список файлов памяти агента
  - read_memory      — прочитать файл памяти
  - get_usage_stats  — статистика использования токенов и стоимости
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

# ── Пути к данным ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
MEMORY_DIR = BASE_DIR / "memory_data"


# ══════════════════════════════════════════════════════════════════════════════
# Реестр инструментов
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "list_logs",
        "description": "Возвращает список всех файлов логов разговоров (conversations) с метаданными.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "description": "Необязательный фильтр по имени файла (подстрока)"
                }
            },
            "required": []
        }
    },
    {
        "name": "read_log",
        "description": "Читает содержимое файла лога разговора по имени файла.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Имя файла лога (например conversation_20260317_192513.json)"
                },
                "last_n": {
                    "type": "integer",
                    "description": "Вернуть только последние N сообщений (по умолчанию — все)"
                }
            },
            "required": ["filename"]
        }
    },
    {
        "name": "search_logs",
        "description": "Ищет заданный текст во всех логах разговоров. Возвращает совпадения с контекстом.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Текст для поиска (без учёта регистра)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Максимальное число результатов (по умолчанию 10)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_memory",
        "description": "Возвращает список файлов долгосрочной памяти агента с кратким содержимым.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "read_memory",
        "description": "Читает файл памяти агента по имени.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Имя файла памяти (например long_term.json)"
                }
            },
            "required": ["filename"]
        }
    },
    {
        "name": "get_usage_stats",
        "description": "Возвращает статистику использования: токены, стоимость, число разговоров.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


# ══════════════════════════════════════════════════════════════════════════════
# Реализация инструментов
# ══════════════════════════════════════════════════════════════════════════════

def tool_list_logs(args: dict) -> str:
    """Список файлов логов."""
    filter_str = args.get("filter", "").lower()
    files = sorted(LOGS_DIR.glob("*.json")) if LOGS_DIR.exists() else []

    results = []
    for f in files:
        if filter_str and filter_str not in f.name.lower():
            continue
        stat = f.stat()
        results.append({
            "filename": f.name,
            "size_kb": round(stat.st_size / 1024, 1),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        })

    lines = [f"Найдено файлов: {len(results)}", ""]
    for r in results:
        lines.append(f"  {r['filename']}  ({r['size_kb']} KB, изменён {r['modified']})")
    return "\n".join(lines)


def tool_read_log(args: dict) -> str:
    """Читает лог разговора."""
    filename = args.get("filename", "")
    last_n = args.get("last_n")

    # Защита от path traversal
    safe_path = (LOGS_DIR / Path(filename).name).resolve()
    if not str(safe_path).startswith(str(LOGS_DIR.resolve())):
        return "Ошибка: недопустимый путь к файлу."
    if not safe_path.exists():
        return f"Файл не найден: {filename}"

    with open(safe_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        messages = data
        if last_n:
            messages = messages[-last_n:]
        lines = [f"Лог: {filename}  ({len(data)} сообщений)", ""]
        for i, msg in enumerate(messages, 1):
            ts = msg.get("timestamp", "")[:19]
            user = msg.get("user_input", "")[:120]
            assistant = msg.get("assistant_response", "")[:200]
            tokens = msg.get("usage", {}).get("total_tokens", "?")
            lines.append(f"[{i}] {ts}  tokens={tokens}")
            lines.append(f"  User: {user}")
            lines.append(f"  Bot:  {assistant}")
            lines.append("")
    else:
        lines = [json.dumps(data, ensure_ascii=False, indent=2)]

    return "\n".join(lines)


def tool_search_logs(args: dict) -> str:
    """Поиск по всем логам."""
    query = args.get("query", "").lower()
    max_results = args.get("max_results", 10)

    if not query:
        return "Ошибка: не задан запрос для поиска."

    results = []
    files = sorted(LOGS_DIR.glob("*.json")) if LOGS_DIR.exists() else []

    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                continue
            for msg in data:
                text = (msg.get("user_input", "") + " " + msg.get("assistant_response", "")).lower()
                if query in text:
                    results.append({
                        "file": f.name,
                        "timestamp": msg.get("timestamp", "")[:19],
                        "snippet": msg.get("user_input", "")[:100]
                    })
                    if len(results) >= max_results:
                        break
        except (json.JSONDecodeError, OSError):
            continue
        if len(results) >= max_results:
            break

    if not results:
        return f"Совпадений для «{query}» не найдено."

    lines = [f"Найдено {len(results)} совпадений для «{query}»:", ""]
    for r in results:
        lines.append(f"  [{r['timestamp']}] {r['file']}")
        lines.append(f"    {r['snippet']}")
        lines.append("")
    return "\n".join(lines)


def tool_list_memory(args: dict) -> str:
    """Список файлов памяти."""
    files = sorted(MEMORY_DIR.glob("*.json")) if MEMORY_DIR.exists() else []
    if not files:
        return "Файлы памяти не найдены."

    lines = [f"Файлы памяти ({len(files)} шт.):", ""]
    for f in files:
        stat = f.stat()
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            key_count = len(data) if isinstance(data, dict) else f"[{len(data)} элементов]"
        except (json.JSONDecodeError, OSError):
            key_count = "?"
        lines.append(f"  {f.name}  ({round(stat.st_size/1024,1)} KB, ключей: {key_count})")
    return "\n".join(lines)


def tool_read_memory(args: dict) -> str:
    """Читает файл памяти."""
    filename = args.get("filename", "")
    safe_path = (MEMORY_DIR / Path(filename).name).resolve()
    if not str(safe_path).startswith(str(MEMORY_DIR.resolve())):
        return "Ошибка: недопустимый путь к файлу."
    if not safe_path.exists():
        return f"Файл не найден: {filename}"

    with open(safe_path, encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, ensure_ascii=False, indent=2)


def tool_get_usage_stats(args: dict) -> str:
    """Статистика использования."""
    stats_file = LOGS_DIR / "usage_stats.json"
    lines = []

    # Из usage_stats.json
    if stats_file.exists():
        with open(stats_file, encoding="utf-8") as f:
            stats = json.load(f)
        lines.append("=== Статистика из usage_stats.json ===")
        lines.append(json.dumps(stats, ensure_ascii=False, indent=2))
        lines.append("")

    # Агрегация по всем логам
    total_tokens = 0
    total_cost = 0.0
    total_messages = 0
    conv_count = 0

    for f in LOGS_DIR.glob("conversation_*.json"):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                conv_count += 1
                for msg in data:
                    usage = msg.get("usage", {})
                    total_tokens += usage.get("total_tokens", 0)
                    total_cost += usage.get("cost", 0.0)
                    total_messages += 1
        except (json.JSONDecodeError, OSError):
            continue

    lines.append("=== Агрегированная статистика по логам ===")
    lines.append(f"  Разговоров:   {conv_count}")
    lines.append(f"  Сообщений:    {total_messages}")
    lines.append(f"  Всего токенов:{total_tokens:,}")
    lines.append(f"  Общая стоимость: ${total_cost:.4f}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# JSON-RPC / MCP dispatch
# ══════════════════════════════════════════════════════════════════════════════

TOOL_HANDLERS = {
    "list_logs": tool_list_logs,
    "read_log": tool_read_log,
    "search_logs": tool_search_logs,
    "list_memory": tool_list_memory,
    "read_memory": tool_read_memory,
    "get_usage_stats": tool_get_usage_stats,
}


def send(obj: dict) -> None:
    """Отправляет JSON-RPC сообщение в stdout (одна строка)."""
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def make_error(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_request(req: dict) -> None:
    """Обрабатывает входящий JSON-RPC запрос по правилам MCP-протокола."""
    method = req.get("method", "")
    req_id = req.get("id")          # None для notifications
    params = req.get("params", {})

    # ── initialize ────────────────────────────────────────────────────────────
    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}           # сервер поддерживает инструменты
                },
                "serverInfo": {
                    "name": "tinyai-logs-server",
                    "version": "1.0.0"
                }
            }
        })

    # ── notifications/initialized ─────────────────────────────────────────────
    elif method == "notifications/initialized":
        pass  # notification — ответ не нужен

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
        if req_id is not None:   # notifications молча игнорируются
            send(make_error(req_id, -32601, f"Неизвестный метод: {method}"))


# ══════════════════════════════════════════════════════════════════════════════
# Главный цикл
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Читает JSON-RPC запросы из stdin (по одному на строку)
    и пишет ответы в stdout.
    stderr используется для отладочных сообщений.
    """
    print(f"[MCP server] TinyAI Logs Server запущен (pid={os.getpid()})", file=sys.stderr)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as e:
            send(make_error(None, -32700, f"Parse error: {e}"))
            continue

        print(f"[MCP server] ← {req.get('method', '?')} (id={req.get('id')})", file=sys.stderr)
        handle_request(req)

    print("[MCP server] stdin закрыт, завершаю работу.", file=sys.stderr)


if __name__ == "__main__":
    main()
