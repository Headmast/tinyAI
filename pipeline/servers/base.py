"""
Общий каркас MCP JSON-RPC сервера (День 19).

Все 4 сервера пайплайна используют этот базовый класс для:
  - stdio JSON-RPC транспорта
  - MCP handshake (initialize, notifications/initialized)
  - tools/list, tools/call диспетчеризации
"""

import json
import sys
from typing import Any, Callable, Dict, List, Optional


class MCPServerBase:
    """Базовый MCP-сервер с JSON-RPC поверх stdio."""

    def __init__(self, name: str, version: str, tools: List[dict],
                 handlers: Dict[str, Callable]):
        self.name = name
        self.version = version
        self.tools = tools
        self.handlers = handlers

    def handle_request(self, req: dict) -> Optional[dict]:
        method = req.get("method", "")
        req_id = req.get("id")

        if method == "initialize":
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self.name, "version": self.version},
                },
            }

        if method == "notifications/initialized":
            return None

        if method == "tools/list":
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"tools": self.tools},
            }

        if method == "tools/call":
            params = req.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            handler = self.handlers.get(tool_name)
            if not handler:
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                        "isError": True,
                    },
                }
            try:
                result_text = handler(arguments)
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": result_text}]},
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {e}"}],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def run(self):
        """Главный цикл: читает JSON-RPC из stdin, пишет ответы в stdout."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
