"""
MCPBridge — мост между AgentLoop и MCP-сервером логов/памяти.

Запускает mcp_server.py как подпроцесс, выполняет MCP-рукопожатие и
предоставляет единственный метод call_tool(name, arguments) → str.

Используется внутри ToolDispatcher: когда LLM вызывает MCP-инструмент
(list_logs, read_log, search_logs, list_memory, read_memory, get_usage_stats),
диспетчер делегирует вызов этому бриджу.

Жизненный цикл:
  - Сервер запускается лениво при первом вызове call_tool()
  - Закрывается через close() или при удалении объекта (__del__)
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

MCP_SERVER_PATH = Path(__file__).parent.parent / "mcp_server.py"
MCP_TOOLS = {
    "list_logs",
    "read_log",
    "search_logs",
    "list_memory",
    "read_memory",
    "get_usage_stats",
}


class MCPBridge:
    """
    Тонкая обёртка над MCP stdio-транспортом для использования внутри агента.

    Особенности:
      - Ленивый старт: сервер запускается только при первом call_tool()
      - Потокобезопасность не требуется (агент однопоточный)
      - Ошибки не бросают исключений наружу — возвращают строку с описанием
    """

    def __init__(self, verbose: bool = False):
        self._proc: Optional[subprocess.Popen] = None
        self._msg_id: int = 0
        self._ready: bool = False
        self._verbose = verbose

    # ── публичный интерфейс ────────────────────────────────────────────────

    def call_tool(self, name: str, arguments: dict) -> str:
        """
        Вызывает MCP-инструмент и возвращает текстовый результат.
        При первом вызове автоматически запускает сервер и делает handshake.
        """
        if not self._ready:
            try:
                self._start()
            except Exception as e:
                return f"[MCP] Ошибка запуска сервера: {e}"

        try:
            resp = self._request("tools/call", {
                "name": name,
                "arguments": arguments,
            })
        except Exception as e:
            return f"[MCP] Ошибка вызова инструмента '{name}': {e}"

        if "error" in resp:
            err = resp["error"]
            return f"[MCP] {err.get('message', str(err))}"

        content = resp.get("result", {}).get("content", [])
        return "\n".join(c.get("text", "") for c in content if c.get("type") == "text")

    def close(self) -> None:
        """Закрывает соединение с MCP-сервером."""
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._proc = None
        self._ready = False

    def __del__(self):
        self.close()

    # ── внутренние методы ─────────────────────────────────────────────────

    def _start(self) -> None:
        """Запускает mcp_server.py и выполняет MCP-рукопожатие."""
        if not MCP_SERVER_PATH.exists():
            raise FileNotFoundError(f"mcp_server.py не найден: {MCP_SERVER_PATH}")

        self._proc = subprocess.Popen(
            [sys.executable, str(MCP_SERVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,   # подавляем отладочный stderr сервера
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        # initialize
        resp = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "news-agent-mcp-bridge", "version": "1.0.0"},
        })
        if "error" in resp:
            raise RuntimeError(f"MCP initialize failed: {resp['error']}")

        # notifications/initialized (без id — уведомление)
        self._notify("notifications/initialized")

        self._ready = True
        if self._verbose:
            info = resp.get("result", {}).get("serverInfo", {})
            print(f"  [MCP Bridge] подключён к {info.get('name')} v{info.get('version')}")

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _send(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        raw = self._proc.stdout.readline()
        if not raw:
            raise ConnectionError("MCP сервер закрыл stdout")
        return json.loads(raw.strip())

    def _request(self, method: str, params: Optional[dict] = None) -> dict:
        msg = {"jsonrpc": "2.0", "id": self._next_id(), "method": method}
        if params:
            msg["params"] = params
        self._send(msg)
        return self._recv()

    def _notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})
