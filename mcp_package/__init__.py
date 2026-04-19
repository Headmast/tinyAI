"""
mcp_package — MCP-компоненты TinyAI.

Пакет предоставляет единую точку импорта для всех MCP-модулей:

    from mcp_package import MCPAgent, MCPRegistry, MCPRouter
    from mcp_package import MCPOrchestratorAgent
    from mcp_package.pipeline import PipelineExecutor, Step
    from mcp_package.scheduler import MCPSchedulerAgent

Корневые модули (mcp_server.py, mcp_agent.py и т.д.) остаются
рабочими для обратной совместимости.
"""

from mcp_agent import MCPAgent
from mcp_registry import MCPRegistry
from mcp_router import MCPRouter
from mcp_orchestrator_agent import MCPOrchestratorAgent

__all__ = [
    "MCPAgent",
    "MCPRegistry",
    "MCPRouter",
    "MCPOrchestratorAgent",
]
