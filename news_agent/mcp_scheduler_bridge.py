"""
MCPSchedulerBridge — мост к серверу планировщика (День 18).

Запускает mcp_scheduler_server.py как подпроцесс, выполняет MCP-рукопожатие
и предоставляет метод call_tool(name, arguments) → str.

Аналог MCPBridge, но подключается к серверу планировщика.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

MCP_SCHEDULER_SERVER_PATH = Path(__file__).parent.parent / "mcp_scheduler_server.py"

# Инструменты планировщика
SCHEDULER_MCP_TOOLS = {
    "scheduler_add_task",
    "scheduler_list_tasks",
    "scheduler_remove_task",
    "scheduler_get_summary",
    "scheduler_list_snapshots",
    "scheduler_list_backups",
    "scheduler_trigger_task",
    "scheduler_get_task_runs",
}


class MCPSchedulerBridge:
    """
    Тонкая обёртка над MCP stdio-транспортом для сервера планировщика.

    Ленивый старт: сервер запускается только при первом call_tool().
    """

    def __init__(self, verbose: bool = False) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._msg_id: int = 0
        self._ready: bool = False
        self._verbose = verbose

    def call_tool(self, name: str, arguments: dict) -> str:
        """Вызывает инструмент планировщика, при необходимости запускает сервер."""
        if not self._ready:
            try:
                self._start()
            except Exception as exc:
                return f"[MCPScheduler] Ошибка запуска сервера: {exc}"

        try:
            resp = self._request("tools/call", {"name": name, "arguments": arguments})
        except Exception as exc:
            return f"[MCPScheduler] Ошибка вызова '{name}': {exc}"

        if "error" in resp:
            err = resp["error"]
            return f"[MCPScheduler] {err.get('message', str(err))}"

        content = resp.get("result", {}).get("content", [])
        return "\n".join(c.get("text", "") for c in content if c.get("type") == "text")

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._proc = None
        self._ready = False

    def __del__(self) -> None:
        self.close()

    # ── внутренние методы ─────────────────────────────────────────────────

    def _start(self) -> None:
        if not MCP_SCHEDULER_SERVER_PATH.exists():
            raise FileNotFoundError(
                f"mcp_scheduler_server.py не найден: {MCP_SCHEDULER_SERVER_PATH}"
            )

        self._proc = subprocess.Popen(
            [sys.executable, str(MCP_SCHEDULER_SERVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        resp = self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "tinyai-scheduler-bridge",
                    "version": "18.0.0",
                },
            },
        )
        if "error" in resp:
            raise RuntimeError(f"MCP initialize failed: {resp['error']}")

        self._notify("notifications/initialized")
        self._ready = True

        if self._verbose:
            info = resp.get("result", {}).get("serverInfo", {})
            print(
                f"  [SchedulerBridge] подключён к "
                f"{info.get('name')} v{info.get('version')}"
            )

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
            raise ConnectionError("MCPScheduler сервер закрыл stdout")
        return json.loads(raw.strip())

    def _request(self, method: str, params: Optional[dict] = None) -> dict:
        msg = {"jsonrpc": "2.0", "id": self._next_id(), "method": method}
        if params:
            msg["params"] = params
        self._send(msg)
        return self._recv()

    def _notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})
