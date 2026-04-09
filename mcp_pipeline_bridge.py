"""
MCPPipelineBridge — мост для AI-агентов к 4 MCP-серверам пайплайна (День 19).

Запускает 4 MCP-сервера как подпроцессы и предоставляет единый интерфейс
call_tool(name, arguments) для AI-агентов с function calling.

Серверы:
  search_server    → search
  summarize_server → summarize
  format_server    → format_content
  store_server     → save_to_file, save_to_db

Использование в агентах:
    bridge = MCPPipelineBridge()
    result = bridge.call_tool("search", {"query": "AI"})
    # ...
    bridge.close()
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

SERVERS_DIR = Path(__file__).parent / "pipeline" / "servers"

# Маппинг: tool → server file
TOOL_TO_SERVER = {
    "search": SERVERS_DIR / "search_server.py",
    "summarize": SERVERS_DIR / "summarize_server.py",
    "format_content": SERVERS_DIR / "format_server.py",
    "save_to_file": SERVERS_DIR / "store_server.py",
    "save_to_db": SERVERS_DIR / "store_server.py",
}

# Все доступные инструменты (для function calling definitions)
PIPELINE_TOOLS = {"search", "summarize", "format_content", "save_to_file", "save_to_db"}

# Определения инструментов в формате OpenAI function calling
PIPELINE_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Поиск по логам разговоров. Возвращает найденные фрагменты.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Текст для поиска"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Суммаризация текста через LLM. Стили: news, digest, brief.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Текст для суммаризации"},
                    "style": {
                        "type": "string",
                        "description": "Стиль: news, digest, brief",
                        "enum": ["news", "digest", "brief"],
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "format_content",
            "description": "Форматирование контента для платформы (telegram, website, rss, plain).",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Контент"},
                    "platform": {
                        "type": "string",
                        "description": "Платформа: telegram, website, rss, plain",
                        "enum": ["telegram", "website", "rss", "plain"],
                    },
                    "title": {"type": "string", "description": "Заголовок"},
                },
                "required": ["content", "platform"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_file",
            "description": "Сохранение контента в файл.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Контент"},
                    "filename": {"type": "string", "description": "Имя файла"},
                },
                "required": ["content", "filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_db",
            "description": "Сохранение контента в БД SQLite.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Заголовок"},
                    "content": {"type": "string", "description": "Текст"},
                    "platform": {
                        "type": "string",
                        "description": "Платформа",
                        "enum": ["telegram", "website", "rss", "plain"],
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Теги",
                    },
                    "summary": {"type": "string", "description": "Краткое описание"},
                },
                "required": ["title", "content", "platform"],
            },
        },
    },
]


class _MCPConn:
    """Minimalist MCP connection to a single server subprocess."""

    def __init__(self, server_path: Path):
        self.server_path = server_path
        self.name = ""
        self._proc: Optional[subprocess.Popen] = None
        self._msg_id = 0
        self._ready = False

    def ensure_ready(self):
        if self._ready:
            return
        self._proc = subprocess.Popen(
            [sys.executable, str(self.server_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        resp = self._send_recv("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "agent-bridge", "version": "1.0"},
        })
        self.name = resp.get("result", {}).get("serverInfo", {}).get("name", "?")
        self._msg_id += 1
        self._proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self._proc.stdin.flush()
        self._ready = True

    def call_tool(self, name: str, arguments: dict) -> str:
        self.ensure_ready()
        resp = self._send_recv("tools/call", {"name": name, "arguments": arguments})
        result = resp.get("result", {})
        if result.get("isError"):
            c = result.get("content", [{}])
            return c[0].get("text", "error") if c else "error"
        c = result.get("content", [{}])
        return c[0].get("text", "") if c else ""

    def close(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._proc = None
        self._ready = False

    def _send_recv(self, method: str, params: Optional[dict] = None) -> dict:
        self._msg_id += 1
        msg = {"jsonrpc": "2.0", "id": self._msg_id, "method": method}
        if params:
            msg["params"] = params
        self._proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        if not raw:
            raise RuntimeError(f"Server {self.server_path.name} closed")
        return json.loads(raw)


class MCPPipelineBridge:
    """
    Единый мост ко всем 4 MCP-серверам пайплайна.

    Ленивый старт: серверы запускаются при первом вызове инструмента.
    Агент вызывает call_tool() — бридж маршрутизирует к нужному серверу.
    """

    def __init__(self, verbose: bool = False):
        self._verbose = verbose
        self._conns: Dict[str, _MCPConn] = {}

    def call_tool(self, name: str, arguments: dict) -> str:
        """Вызывает MCP-инструмент, маршрутизируя к нужному серверу."""
        server_path = TOOL_TO_SERVER.get(name)
        if not server_path:
            return f"[MCP Pipeline] Unknown tool: {name}"

        key = str(server_path)
        if key not in self._conns:
            conn = _MCPConn(server_path)
            self._conns[key] = conn
            if self._verbose:
                conn.ensure_ready()
                print(f"  🔗 {conn.name}")

        try:
            return self._conns[key].call_tool(name, arguments)
        except Exception as e:
            return f"[MCP Pipeline] Error calling {name}: {e}"

    def close(self):
        """Закрывает все MCP-подключения."""
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()

    def __del__(self):
        self.close()
