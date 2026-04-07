"""
MCP-клиент для TinyAI — подключается к локальному MCP-серверу
и получает список доступных инструментов.

Как работает:
  1. Клиент запускает mcp_server.py как подпроцесс (stdio transport)
  2. Отправляет JSON-RPC запрос "initialize" → получает capabilities сервера
  3. Отправляет уведомление "notifications/initialized"
  4. Отправляет "tools/list" → получает список инструментов
  5. (опционально) вызывает инструменты через "tools/call"

Транспорт: stdin/stdout подпроцесса (стандарт MCP stdio)
Протокол:  JSON-RPC 2.0 — каждое сообщение на отдельной строке
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List

# Путь к серверу
SERVER_SCRIPT = Path(__file__).parent / "mcp_server.py"
PYTHON = sys.executable  # тот же Python, что запустил клиент


# ══════════════════════════════════════════════════════════════════════════════
# Низкоуровневый транспорт (stdio)
# ══════════════════════════════════════════════════════════════════════════════

class MCPConnection:
    """
    Управляет подпроцессом-сервером и обменом JSON-RPC сообщениями.

    Каждое сообщение — одна строка JSON.
    stdin → запросы клиента
    stdout ← ответы сервера
    stderr ← отладочные сообщения сервера (выводятся отдельно)
    """

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._msg_id = 0

    def connect(self) -> None:
        """Запускает сервер как подпроцесс."""
        if not SERVER_SCRIPT.exists():
            raise FileNotFoundError(f"Сервер не найден: {SERVER_SCRIPT}")

        self._proc = subprocess.Popen(
            [PYTHON, str(SERVER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # stderr сервера читаем отдельно
            text=True,
            encoding="utf-8",
            bufsize=1                 # line-buffered
        )
        print(f"[client] Сервер запущен (pid={self._proc.pid})")

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _send(self, obj: dict) -> None:
        """Пишет JSON-строку в stdin сервера."""
        line = json.dumps(obj, ensure_ascii=False)
        print(f"[client] → {obj.get('method', '?')} (id={obj.get('id')})")
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        """Читает одну строку JSON из stdout сервера."""
        raw = self._proc.stdout.readline()
        if not raw:
            raise ConnectionError("Сервер закрыл соединение (stdout EOF)")
        obj = json.loads(raw.strip())
        print(f"[client] ← получен ответ (id={obj.get('id')})")
        return obj

    def request(self, method: str, params: Optional[dict] = None) -> dict:
        """Отправляет запрос и возвращает ответ."""
        msg_id = self._next_id()
        msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params:
            msg["params"] = params
        self._send(msg)
        return self._recv()

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        """Отправляет уведомление (без id, ответа не ждём)."""
        msg = {"jsonrpc": "2.0", "method": method}
        if params:
            msg["params"] = params
        self._send(msg)

    def close(self) -> None:
        """Закрывает соединение и завершает подпроцесс."""
        if self._proc:
            self._proc.stdin.close()
            self._proc.wait(timeout=3)
            # Вывести stderr сервера
            stderr_output = self._proc.stderr.read()
            if stderr_output:
                print("\n[server stderr]")
                for line in stderr_output.strip().splitlines():
                    print(f"  {line}")
            print(f"[client] Сервер завершён (код={self._proc.returncode})")


# ══════════════════════════════════════════════════════════════════════════════
# Высокоуровневый MCP-клиент
# ══════════════════════════════════════════════════════════════════════════════

class MCPClient:
    """
    Высокоуровневый клиент MCP.

    Выполняет MCP-рукопожатие и предоставляет методы для работы
    с инструментами сервера.
    """

    def __init__(self):
        self._conn = MCPConnection()
        self.server_info: dict = {}
        self.server_capabilities: dict = {}

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    def connect(self) -> None:
        """Устанавливает соединение и выполняет MCP-рукопожатие."""
        print("\n" + "="*60)
        print("  Шаг 1: Запускаем MCP-сервер")
        print("="*60)
        self._conn.connect()

        print("\n" + "="*60)
        print("  Шаг 2: Отправляем initialize")
        print("="*60)
        resp = self._conn.request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "tinyai-mcp-client",
                "version": "1.0.0"
            }
        })
        if "error" in resp:
            raise RuntimeError(f"initialize failed: {resp['error']}")

        result = resp["result"]
        self.server_info = result.get("serverInfo", {})
        self.server_capabilities = result.get("capabilities", {})

        print(f"\n  Сервер: {self.server_info.get('name')} v{self.server_info.get('version')}")
        print(f"  Протокол: {result.get('protocolVersion')}")
        print(f"  Capabilities: {list(self.server_capabilities.keys())}")

        print("\n" + "="*60)
        print("  Шаг 3: Отправляем notifications/initialized")
        print("="*60)
        self._conn.notify("notifications/initialized")
        print("  Уведомление отправлено — рукопожатие завершено!\n")

    def list_tools(self) -> List[dict]:
        """Запрашивает список инструментов у сервера."""
        print("="*60)
        print("  Шаг 4: Запрашиваем tools/list")
        print("="*60)
        resp = self._conn.request("tools/list")
        if "error" in resp:
            raise RuntimeError(f"tools/list failed: {resp['error']}")
        return resp["result"].get("tools", [])

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> str:
        """Вызывает инструмент на сервере и возвращает текстовый результат."""
        resp = self._conn.request("tools/call", {
            "name": name,
            "arguments": arguments or {}
        })
        if "error" in resp:
            raise RuntimeError(f"tools/call '{name}' failed: {resp['error']}")
        content = resp["result"].get("content", [])
        return "\n".join(c.get("text", "") for c in content if c.get("type") == "text")

    def disconnect(self) -> None:
        self._conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# Демонстрация: подключение + список инструментов + вызов инструмента
# ══════════════════════════════════════════════════════════════════════════════

def print_tools(tools: List[dict]) -> None:
    """Красиво выводит список инструментов."""
    print(f"\n{'='*60}")
    print(f"  ДОСТУПНЫЕ ИНСТРУМЕНТЫ MCP ({len(tools)} шт.)")
    print(f"{'='*60}")
    for i, tool in enumerate(tools, 1):
        name = tool.get("name", "?")
        desc = tool.get("description", "")
        schema = tool.get("inputSchema", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])

        print(f"\n  {i}. {name}")
        print(f"     {desc}")
        if props:
            print(f"     Параметры:")
            for param, info in props.items():
                req_mark = " *" if param in required else ""
                ptype = info.get("type", "any")
                pdesc = info.get("description", "")
                print(f"       - {param}{req_mark} ({ptype}): {pdesc}")
    print()


def main() -> None:
    print("\n" + "█"*60)
    print("  TinyAI MCP Client  —  День 16")
    print("  Подключение к локальному MCP-серверу логов/памяти")
    print("█"*60)

    with MCPClient() as client:
        # ── Шаг 4: получаем список инструментов ──────────────────────────────
        tools = client.list_tools()
        print_tools(tools)

        # ── Демо: вызываем несколько инструментов ─────────────────────────────
        print("="*60)
        print("  ДЕМО: Вызов инструментов")
        print("="*60)

        print("\n→ Вызываем list_logs")
        print("-"*40)
        result = client.call_tool("list_logs")
        print(result)

        print("\n→ Вызываем list_memory")
        print("-"*40)
        result = client.call_tool("list_memory")
        print(result)

        print("\n→ Вызываем get_usage_stats")
        print("-"*40)
        result = client.call_tool("get_usage_stats")
        print(result)

        print("\n→ Вызываем search_logs с запросом 'нейрон'")
        print("-"*40)
        result = client.call_tool("search_logs", {"query": "нейрон", "max_results": 3})
        print(result)

    print("\n" + "█"*60)
    print("  Готово! MCP-соединение закрыто.")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()
