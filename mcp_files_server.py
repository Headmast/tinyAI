#!/usr/bin/env python3
"""
MCP-сервер для работы с файлами проекта.

Протокол: Model Context Protocol (MCP) v2024-11-05
Транспорт: JSON-RPC 2.0 по stdout/stdin (stdio transport)

Доступные инструменты (tools):
  - list_project_files  — список файлов/каталогов
  - read_file           — чтение содержимого файла
  - search_in_files     — поиск текста по файлам проекта
  - get_file_info       — метаданные файла
  - write_file          — создание/перезапись файла
  - apply_diff          — замена текста в файле (str_replace)

Запуск (через MCP-клиент):
    echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python mcp_files_server.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent

# Каталоги и файлы, которые пропускаем при обходе
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache"}
SKIP_EXTENSIONS = {".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".bin", ".pkl", ".db"}

# Файлы, которые нельзя модифицировать
PROTECTED_PATHS = {".git", ".env", ".env.local", ".env.production"}


# ══════════════════════════════════════════════════════════════════════════════
# Утилиты безопасности
# ══════════════════════════════════════════════════════════════════════════════

def _safe_resolve(path_str: str) -> Optional[Path]:
    """
    Резолвит путь и проверяет что он внутри BASE_DIR.
    Возвращает None при нарушении.
    """
    try:
        resolved = (BASE_DIR / path_str).resolve()
        base_resolved = BASE_DIR.resolve()
        if not str(resolved).startswith(str(base_resolved)):
            return None
        return resolved
    except (ValueError, OSError):
        return None


def _is_protected(path: Path) -> bool:
    """Проверяет, является ли путь защищённым от записи."""
    rel = path.relative_to(BASE_DIR.resolve())
    parts = rel.parts
    for part in parts:
        if part in PROTECTED_PATHS:
            return True
    return str(rel) in PROTECTED_PATHS


def _is_binary(path: Path) -> bool:
    """Эвристика: бинарный ли файл (по первым 8KB)."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except (OSError, IOError):
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Реестр инструментов
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "list_project_files",
        "description": (
            "Возвращает список файлов и каталогов по указанному пути проекта. "
            "Пропускает .git, __pycache__, node_modules. "
            "Показывает имя, тип (file/dir), размер."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Относительный путь от корня проекта (по умолчанию '.')"
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob-паттерн для фильтрации (например '*.py')"
                }
            },
            "required": []
        }
    },
    {
        "name": "read_file",
        "description": (
            "Читает содержимое файла проекта. "
            "Возвращает текст файла и количество строк. "
            "Можно ограничить количество строк через max_lines."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Относительный путь к файлу от корня проекта"
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Максимальное количество строк для чтения (по умолчанию все)"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "search_in_files",
        "description": (
            "Ищет текст по файлам проекта (регистро-независимо). "
            "Возвращает файл, номер строки и совпавшую строку. "
            "Пропускает бинарные файлы и скрытые каталоги."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Текст для поиска (без учёта регистра)"
                },
                "path": {
                    "type": "string",
                    "description": "Каталог для поиска (по умолчанию '.' — весь проект)"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob-паттерн файлов (например '*.py')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Максимум результатов (по умолчанию 30)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_file_info",
        "description": (
            "Возвращает метаданные файла: размер в байтах, количество строк, "
            "дату последнего изменения, является ли бинарным."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Относительный путь к файлу от корня проекта"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": (
            "Создаёт или перезаписывает файл. Создаёт родительские каталоги при необходимости. "
            "Запрещено писать в .git/, .env и бинарные файлы."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Относительный путь к файлу от корня проекта"
                },
                "content": {
                    "type": "string",
                    "description": "Содержимое для записи"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "apply_diff",
        "description": (
            "Заменяет текст в файле (str_replace). "
            "old_text должен встречаться в файле ровно 1 раз. "
            "Возвращает успех и превью изменённого участка."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Относительный путь к файлу от корня проекта"
                },
                "old_text": {
                    "type": "string",
                    "description": "Текст для замены (должен встречаться ровно 1 раз)"
                },
                "new_text": {
                    "type": "string",
                    "description": "Новый текст для вставки вместо old_text"
                }
            },
            "required": ["path", "old_text", "new_text"]
        }
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Реализация инструментов
# ══════════════════════════════════════════════════════════════════════════════

def tool_list_project_files(args: dict) -> str:
    """Список файлов и каталогов."""
    rel_path = args.get("path", ".")
    pattern = args.get("pattern", "")

    resolved = _safe_resolve(rel_path)
    if resolved is None:
        return "Ошибка: недопустимый путь (выход за пределы проекта)."

    if not resolved.exists():
        return f"Ошибка: путь не существует: {rel_path}"

    if not resolved.is_dir():
        return f"Ошибка: {rel_path} — не каталог."

    entries = []
    try:
        items = sorted(resolved.iterdir())
    except PermissionError:
        return f"Ошибка: нет доступа к {rel_path}"

    for item in items:
        name = item.name
        # Пропускаем скрытые каталоги и стоп-листы
        if name in SKIP_DIRS:
            continue
        if item.is_file() and item.suffix in SKIP_EXTENSIONS:
            continue

        if pattern and not item.match(pattern):
            continue

        if item.is_dir():
            entries.append({"name": name + "/", "type": "dir"})
        else:
            try:
                size = item.stat().st_size
            except OSError:
                size = 0
            entries.append({"name": name, "type": "file", "size": size})

    if not entries:
        return f"Каталог {rel_path} пуст (или все файлы отфильтрованы)."

    lines = [f"Содержимое {rel_path}/ ({len(entries)} элементов):"]
    for e in entries:
        if e["type"] == "dir":
            lines.append(f"  📁 {e['name']}")
        else:
            size_str = _format_size(e.get("size", 0))
            lines.append(f"  📄 {e['name']}  ({size_str})")
    return "\n".join(lines)


def tool_read_file(args: dict) -> str:
    """Чтение файла."""
    path_str = args.get("path", "")
    max_lines = args.get("max_lines")

    if not path_str:
        return "Ошибка: не указан путь к файлу."

    resolved = _safe_resolve(path_str)
    if resolved is None:
        return "Ошибка: недопустимый путь (выход за пределы проекта)."

    if not resolved.exists():
        return f"Ошибка: файл не найден: {path_str}"

    if not resolved.is_file():
        return f"Ошибка: {path_str} — не файл."

    if _is_binary(resolved):
        return f"Ошибка: {path_str} — бинарный файл, чтение невозможно."

    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = resolved.read_text(encoding="latin-1")
        except Exception:
            return f"Ошибка: не удалось прочитать {path_str} (проблема кодировки)."
    except OSError as e:
        return f"Ошибка чтения: {e}"

    lines = content.splitlines()
    total_lines = len(lines)

    if max_lines and max_lines > 0 and max_lines < total_lines:
        content = "\n".join(lines[:max_lines])
        return f"Файл: {path_str} (строки 1-{max_lines} из {total_lines}):\n\n{content}"

    return f"Файл: {path_str} ({total_lines} строк):\n\n{content}"


def tool_search_in_files(args: dict) -> str:
    """Поиск текста по файлам проекта."""
    query = args.get("query", "")
    if not query:
        return "Ошибка: не указан текст для поиска."

    search_path = args.get("path", ".")
    file_pattern = args.get("file_pattern", "")
    max_results = min(args.get("max_results", 30), 100)

    resolved = _safe_resolve(search_path)
    if resolved is None:
        return "Ошибка: недопустимый путь поиска."

    if not resolved.exists():
        return f"Ошибка: путь не существует: {search_path}"

    query_lower = query.lower()
    results = []

    # Рекурсивный обход
    for root, dirs, files in os.walk(resolved):
        # Пропускаем скрытые каталоги
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            if file_pattern and not Path(fname).match(file_pattern):
                continue

            fpath = Path(root) / fname
            if fpath.suffix in SKIP_EXTENSIONS:
                continue

            # Пропускаем бинарные
            if _is_binary(fpath):
                continue

            try:
                text = fpath.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            for line_no, line in enumerate(text.splitlines(), 1):
                if query_lower in line.lower():
                    rel = fpath.relative_to(BASE_DIR.resolve())
                    results.append({
                        "file": str(rel),
                        "line": line_no,
                        "text": line.strip()[:200],
                    })
                    if len(results) >= max_results:
                        break
            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break

    if not results:
        return f"Ничего не найдено по запросу: '{query}'"

    lines = [f"Найдено совпадений: {len(results)} (запрос: '{query}'):"]
    for r in results:
        lines.append(f"  {r['file']}:{r['line']}  {r['text']}")
    return "\n".join(lines)


def tool_get_file_info(args: dict) -> str:
    """Метаданные файла."""
    path_str = args.get("path", "")
    if not path_str:
        return "Ошибка: не указан путь к файлу."

    resolved = _safe_resolve(path_str)
    if resolved is None:
        return "Ошибка: недопустимый путь (выход за пределы проекта)."

    if not resolved.exists():
        return f"Ошибка: файл не найден: {path_str}"

    stat = resolved.stat()
    info = {
        "path": path_str,
        "size_bytes": stat.st_size,
        "size_human": _format_size(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "is_dir": resolved.is_dir(),
        "is_binary": _is_binary(resolved) if resolved.is_file() else False,
    }

    if resolved.is_file() and not info["is_binary"]:
        try:
            content = resolved.read_text(encoding="utf-8")
            info["lines"] = len(content.splitlines())
        except (UnicodeDecodeError, OSError):
            info["lines"] = None

    return json.dumps(info, ensure_ascii=False, indent=2)


def tool_write_file(args: dict) -> str:
    """Создание/перезапись файла."""
    path_str = args.get("path", "")
    content = args.get("content", "")

    if not path_str:
        return "Ошибка: не указан путь к файлу."

    resolved = _safe_resolve(path_str)
    if resolved is None:
        return "Ошибка: недопустимый путь (выход за пределы проекта)."

    if _is_protected(resolved):
        return f"Ошибка: запись в защищённый файл/каталог запрещена: {path_str}"

    if resolved.exists() and _is_binary(resolved):
        return f"Ошибка: перезапись бинарного файла запрещена: {path_str}"

    # Создаём родительские каталоги
    resolved.parent.mkdir(parents=True, exist_ok=True)

    try:
        resolved.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"Ошибка записи: {e}"

    size = resolved.stat().st_size
    lines_count = len(content.splitlines())
    return json.dumps({
        "success": True,
        "path": path_str,
        "bytes_written": size,
        "lines": lines_count,
    }, ensure_ascii=False)


def tool_apply_diff(args: dict) -> str:
    """Замена текста в файле (str_replace)."""
    path_str = args.get("path", "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")

    if not path_str:
        return "Ошибка: не указан путь к файлу."
    if not old_text:
        return "Ошибка: не указан old_text для замены."

    resolved = _safe_resolve(path_str)
    if resolved is None:
        return "Ошибка: недопустимый путь (выход за пределы проекта)."

    if not resolved.exists():
        return f"Ошибка: файл не найден: {path_str}"

    if not resolved.is_file():
        return f"Ошибка: {path_str} — не файл."

    if _is_protected(resolved):
        return f"Ошибка: модификация защищённого файла запрещена: {path_str}"

    if _is_binary(resolved):
        return f"Ошибка: модификация бинарного файла запрещена: {path_str}"

    try:
        content = resolved.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        return f"Ошибка чтения: {e}"

    count = content.count(old_text)
    if count == 0:
        return "Ошибка: old_text не найден в файле."
    if count > 1:
        return f"Ошибка: old_text найден {count} раз (должен быть ровно 1). Уточните фрагмент."

    new_content = content.replace(old_text, new_text, 1)

    try:
        resolved.write_text(new_content, encoding="utf-8")
    except OSError as e:
        return f"Ошибка записи: {e}"

    # Превью изменения (показываем контекст вокруг замены)
    pos = new_content.find(new_text)
    start = max(0, pos - 50)
    end = min(len(new_content), pos + len(new_text) + 50)
    preview = new_content[start:end]

    return json.dumps({
        "success": True,
        "path": path_str,
        "preview": f"...{preview}...",
    }, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# Вспомогательные
# ══════════════════════════════════════════════════════════════════════════════

def _format_size(size_bytes: int) -> str:
    """Человекочитаемый размер файла."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# ══════════════════════════════════════════════════════════════════════════════
# JSON-RPC / MCP dispatch
# ══════════════════════════════════════════════════════════════════════════════

TOOL_HANDLERS = {
    "list_project_files": tool_list_project_files,
    "read_file": tool_read_file,
    "search_in_files": tool_search_in_files,
    "get_file_info": tool_get_file_info,
    "write_file": tool_write_file,
    "apply_diff": tool_apply_diff,
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
                    "name": "tinyai-files-server",
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
    print(f"[MCP files server] TinyAI Files Server запущен (pid={os.getpid()})", file=sys.stderr)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as e:
            send(make_error(None, -32700, f"Parse error: {e}"))
            continue

        print(f"[MCP files server] ← {req.get('method', '?')} (id={req.get('id')})", file=sys.stderr)
        handle_request(req)


if __name__ == "__main__":
    main()
