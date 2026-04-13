"""
MCPRegistry — центральный реестр MCP-серверов с автоматическим обнаружением
инструментов (День 20: Orchestration MCP).

Позволяет:
  - Регистрировать MCP-серверы по имени и пути
  - Автоматически обнаруживать инструменты через tools/list
  - Получать единый список всех инструментов со всех серверов
  - Генерировать определения в формате OpenAI function calling

Использование:
    registry = MCPRegistry()
    registry.register("logs", "mcp_server.py")
    registry.register("scheduler", "mcp_scheduler_server.py")
    registry.discover_tools()  # вызывает tools/list у каждого сервера
    all_defs = registry.get_tool_definitions()
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class ServerInfo:
    """Информация о зарегистрированном MCP-сервере."""
    name: str
    server_path: Path
    tags: List[str] = field(default_factory=list)
    tools: List[dict] = field(default_factory=list)
    tool_names: Set[str] = field(default_factory=set)
    discovered: bool = False


class _MCPConn:
    """Соединение с MCP-сервером через stdio subprocess."""

    def __init__(self, server_path: Path):
        self.server_path = server_path
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
            "clientInfo": {"name": "orchestrator", "version": "1.0"},
        })
        self._msg_id += 1
        self._proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        self._proc.stdin.flush()
        self._ready = True

    def list_tools(self) -> List[dict]:
        """Вызывает tools/list и возвращает список инструментов сервера."""
        self.ensure_ready()
        resp = self._send_recv("tools/list", {})
        return resp.get("result", {}).get("tools", [])

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
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": self._msg_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        if not raw:
            raise RuntimeError(f"Server {self.server_path.name} closed")
        return json.loads(raw)


def _input_schema_to_parameters(schema: dict) -> dict:
    """Конвертирует MCP inputSchema в OpenAI function parameters."""
    return {
        "type": schema.get("type", "object"),
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }


class MCPRegistry:
    """
    Центральный реестр MCP-серверов.

    Регистрирует серверы, обнаруживает их инструменты через tools/list,
    предоставляет единый маппинг tool_name → server_name и определения
    инструментов в формате OpenAI function calling.
    """

    def __init__(self, verbose: bool = False):
        self._servers: Dict[str, ServerInfo] = {}
        self._tool_to_server: Dict[str, str] = {}
        self._connections: Dict[str, _MCPConn] = {}
        self._verbose = verbose

    def register(self, name: str, server_path: str, tags: Optional[List[str]] = None):
        """Регистрирует MCP-сервер в реестре."""
        path = Path(server_path)
        if not path.is_absolute():
            path = Path(__file__).parent / path
        if name in self._servers:
            raise ValueError(f"Server '{name}' already registered")
        self._servers[name] = ServerInfo(
            name=name, server_path=path, tags=tags or [],
        )

    def discover_tools(self):
        """Обнаруживает инструменты всех зарегистрированных серверов через tools/list."""
        for name, info in self._servers.items():
            if info.discovered:
                continue
            conn = self._get_connection(name)
            tools = conn.list_tools()
            info.tools = tools
            info.tool_names = {t["name"] for t in tools}
            info.discovered = True
            for t in tools:
                tool_name = t["name"]
                if tool_name in self._tool_to_server:
                    existing = self._tool_to_server[tool_name]
                    if self._verbose:
                        print(
                            f"  ⚠️  Tool '{tool_name}' already registered on "
                            f"'{existing}', skipping from '{name}'"
                        )
                    continue
                self._tool_to_server[tool_name] = name
            if self._verbose:
                print(f"  🔍 {name}: обнаружено {len(tools)} инструментов — "
                      f"{', '.join(info.tool_names)}")

    def list_servers(self) -> List[Dict[str, Any]]:
        """Список серверов со статусом и количеством инструментов."""
        result = []
        for name, info in self._servers.items():
            result.append({
                "name": name,
                "path": str(info.server_path),
                "tags": info.tags,
                "discovered": info.discovered,
                "tool_count": len(info.tools),
                "tools": sorted(info.tool_names),
            })
        return result

    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Возвращает имя сервера для данного инструмента."""
        return self._tool_to_server.get(tool_name)

    def get_all_tool_names(self) -> Set[str]:
        """Множество всех доступных инструментов."""
        return set(self._tool_to_server.keys())

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Определения всех инструментов в формате OpenAI function calling."""
        definitions = []
        for server_name, info in self._servers.items():
            for tool in info.tools:
                tool_name = tool["name"]
                if self._tool_to_server.get(tool_name) != server_name:
                    continue  # пропускаем дубликаты
                definitions.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool.get("description", ""),
                        "parameters": _input_schema_to_parameters(
                            tool.get("inputSchema", {})
                        ),
                    },
                })
        return definitions

    def get_connection(self, server_name: str) -> _MCPConn:
        """Возвращает соединение с сервером (для MCPRouter)."""
        if server_name not in self._servers:
            raise ValueError(f"Server '{server_name}' not registered")
        return self._get_connection(server_name)

    def health_check(self) -> List[Dict[str, Any]]:
        """
        Проверяет доступность каждого зарегистрированного сервера.

        Для каждого сервера пытается вызвать tools/list и замеряет latency.
        Возвращает список с результатами проверки.
        """
        results = []
        for name, info in self._servers.items():
            entry: Dict[str, Any] = {
                "name": name,
                "status": "unknown",
                "latency_ms": 0,
                "tool_count": 0,
                "error": None,
            }
            t0 = time.perf_counter()
            try:
                conn = self._get_connection(name)
                tools = conn.list_tools()
                elapsed = (time.perf_counter() - t0) * 1000
                entry["status"] = "healthy"
                entry["latency_ms"] = round(elapsed, 1)
                entry["tool_count"] = len(tools)
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                entry["status"] = "unhealthy"
                entry["latency_ms"] = round(elapsed, 1)
                entry["error"] = str(e)
            results.append(entry)
        return results

    def _get_connection(self, server_name: str) -> _MCPConn:
        if server_name not in self._connections:
            info = self._servers[server_name]
            self._connections[server_name] = _MCPConn(info.server_path)
        return self._connections[server_name]

    def close(self):
        """Закрывает все соединения."""
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
