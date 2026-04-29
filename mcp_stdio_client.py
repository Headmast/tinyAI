"""
Универсальный MCP-клиент для stdio-серверов (JSON-RPC 2.0).

Извлечён из dev_assistant._GitMCPClient — общий паттерн для взаимодействия
с любым MCP-сервером, запущенным как subprocess с stdin/stdout.

Использование:
    from mcp_stdio_client import MCPStdioClient

    client = MCPStdioClient(server_path="mcp_git_server.py")
    result = client.call_tool("git_current_branch", {})
    print(result)
    client.close()
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


class MCPStdioClient:
    """
    Минимальный MCP-клиент: запускает сервер как subprocess,
    общается по stdin/stdout через JSON-RPC 2.0.

    Lazy initialization — subprocess запускается при первом вызове call_tool().
    """

    def __init__(self, server_path: str | Path, client_name: str = "mcp-client"):
        self._server_path = Path(server_path)
        self._client_name = client_name
        self._proc: Optional[subprocess.Popen] = None
        self._msg_id = 0

    def _ensure_started(self):
        """Запускает subprocess при необходимости + делает MCP handshake."""
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            [sys.executable, str(self._server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Handshake — initialize
        self._send_recv("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": self._client_name, "version": "1.0"},
        })
        # Notification — initialized
        self._proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        self._proc.stdin.flush()

    def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """Вызывает MCP-инструмент и возвращает текстовый результат."""
        self._ensure_started()
        resp = self._send_recv("tools/call", {"name": name, "arguments": arguments or {}})
        result = resp.get("result", {})
        content = result.get("content", [{}])
        return content[0].get("text", "") if content else ""

    def list_tools(self) -> list[dict]:
        """Возвращает список инструментов сервера."""
        self._ensure_started()
        resp = self._send_recv("tools/list", {})
        result = resp.get("result", {})
        return result.get("tools", [])

    def close(self):
        """Завершает subprocess сервера."""
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._proc = None

    def _send_recv(self, method: str, params: dict | None = None) -> dict:
        """Отправляет JSON-RPC запрос и читает ответ."""
        self._msg_id += 1
        msg = {"jsonrpc": "2.0", "id": self._msg_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        if not raw:
            raise RuntimeError(f"MCP server ({self._server_path.name}) closed unexpectedly")
        return json.loads(raw)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
