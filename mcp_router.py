"""
MCPRouter — единый маршрутизатор MCP-вызовов (День 20: Orchestration MCP).

Принимает MCPRegistry и предоставляет единый интерфейс call_tool(name, args),
автоматически направляя запрос к нужному серверу.

Использование:
    registry = MCPRegistry()
    registry.register("logs", "mcp_server.py")
    registry.discover_tools()

    router = MCPRouter(registry)
    result = router.call_tool("list_logs", {"filter": "2026"})
    # → маршрутизирует к серверу "logs"
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mcp_registry import MCPRegistry


@dataclass
class ToolCallRecord:
    """Запись о вызове инструмента."""
    tool: str
    server: str
    elapsed_ms: float
    success: bool
    error: Optional[str] = None


class MCPRouter:
    """
    Единый прокси для вызова MCP-инструментов с любого зарегистрированного сервера.

    Автоматически определяет, на каком сервере находится инструмент,
    и маршрутизирует вызов через MCPRegistry.
    """

    def __init__(self, registry: MCPRegistry, verbose: bool = False):
        self._registry = registry
        self._verbose = verbose
        self._call_log: List[ToolCallRecord] = []

    def call_tool(self, name: str, arguments: dict) -> str:
        """
        Вызывает инструмент, маршрутизируя к нужному серверу.

        Возвращает текстовый результат или строку с описанием ошибки.
        """
        server_name = self._registry.get_server_for_tool(name)
        if server_name is None:
            error_msg = f"[MCPRouter] Инструмент '{name}' не найден ни на одном сервере"
            self._call_log.append(ToolCallRecord(
                tool=name, server="?", elapsed_ms=0,
                success=False, error=error_msg,
            ))
            return error_msg

        t0 = time.perf_counter()
        try:
            conn = self._registry.get_connection(server_name)
            result = conn.call_tool(name, arguments)
            elapsed = (time.perf_counter() - t0) * 1000

            self._call_log.append(ToolCallRecord(
                tool=name, server=server_name,
                elapsed_ms=round(elapsed, 1), success=True,
            ))

            if self._verbose:
                print(f"  🔧 {name} → {server_name} ({elapsed:.0f}ms)")

            return result

        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            error_msg = f"[MCPRouter] Ошибка вызова '{name}' на '{server_name}': {e}"
            self._call_log.append(ToolCallRecord(
                tool=name, server=server_name,
                elapsed_ms=round(elapsed, 1), success=False, error=error_msg,
            ))
            return error_msg

    @property
    def call_history(self) -> List[ToolCallRecord]:
        """История всех вызовов."""
        return list(self._call_log)

    @property
    def servers_used(self) -> List[str]:
        """Уникальные серверы, задействованные в вызовах."""
        return list(dict.fromkeys(r.server for r in self._call_log if r.success))

    def get_summary(self) -> Dict[str, Any]:
        """Сводка по вызовам: количество, серверы, ошибки, время."""
        total = len(self._call_log)
        ok = sum(1 for r in self._call_log if r.success)
        errors = total - ok
        total_ms = sum(r.elapsed_ms for r in self._call_log)
        servers = self.servers_used
        return {
            "total_calls": total,
            "successful": ok,
            "errors": errors,
            "total_elapsed_ms": round(total_ms, 1),
            "servers_used": servers,
            "calls": [
                {
                    "tool": r.tool,
                    "server": r.server,
                    "elapsed_ms": r.elapsed_ms,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self._call_log
            ],
        }
