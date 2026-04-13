"""
Тесты для Orchestration MCP (День 20).

Покрываем:
  1. MCPRegistry — регистрация серверов, обнаружение инструментов, маппинг
  2. MCPRouter — маршрутизация вызовов, обработка ошибок, история
  3. MCPOrchestratorAgent — мульти-серверный function calling loop
  4. Интеграция — E2E с реальными MCP-серверами
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_tool_schema(name: str, description: str = "", props: dict = None):
    """Создаёт MCP tool schema (серверный формат)."""
    return {
        "name": name,
        "description": description or f"Tool {name}",
        "inputSchema": {
            "type": "object",
            "properties": props or {},
            "required": [],
        },
    }


def _mock_conn(tools: list):
    """Создаёт мок _MCPConn с заданным списком tools."""
    conn = MagicMock()
    conn.list_tools.return_value = tools
    conn.call_tool.return_value = "mock result"
    conn.close.return_value = None
    return conn


def _mock_llm_response(content: str = None, tool_calls: list = None):
    """Создаёт мок ответа LLM."""
    choice = MagicMock()
    choice.message.content = content or ""
    choice.message.tool_calls = tool_calls
    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )
    return response


def _mock_tool_call(tc_id: str, name: str, args: dict):
    """Создаёт мок tool_call от LLM."""
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


# ══════════════════════════════════════════════════════════════════════════════
# 1. MCPRegistry
# ══════════════════════════════════════════════════════════════════════════════

class TestMCPRegistry:
    """Тесты реестра MCP-серверов."""

    def test_register_server(self):
        from mcp_registry import MCPRegistry
        reg = MCPRegistry()
        reg.register("test", str(PROJECT_ROOT / "mcp_server.py"), tags=["logs"])
        servers = reg.list_servers()
        assert len(servers) == 1
        assert servers[0]["name"] == "test"
        assert servers[0]["tags"] == ["logs"]
        assert servers[0]["discovered"] is False

    def test_register_duplicate_raises(self):
        from mcp_registry import MCPRegistry
        reg = MCPRegistry()
        reg.register("s1", str(PROJECT_ROOT / "mcp_server.py"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register("s1", str(PROJECT_ROOT / "mcp_server.py"))

    def test_register_relative_path(self):
        from mcp_registry import MCPRegistry
        reg = MCPRegistry()
        reg.register("test", "mcp_server.py")
        servers = reg.list_servers()
        assert "mcp_server.py" in servers[0]["path"]

    def test_discover_tools_populates_mapping(self):
        from mcp_registry import MCPRegistry
        reg = MCPRegistry()
        reg.register("s1", str(PROJECT_ROOT / "mcp_server.py"))

        tools_s1 = [
            _make_tool_schema("tool_a", "Tool A"),
            _make_tool_schema("tool_b", "Tool B"),
        ]
        mock_conn = _mock_conn(tools_s1)

        with patch.object(reg, "_get_connection", return_value=mock_conn):
            reg.discover_tools()

        assert reg.get_server_for_tool("tool_a") == "s1"
        assert reg.get_server_for_tool("tool_b") == "s1"
        assert reg.get_all_tool_names() == {"tool_a", "tool_b"}

    def test_discover_tools_multiple_servers(self):
        from mcp_registry import MCPRegistry
        reg = MCPRegistry()
        reg.register("logs", str(PROJECT_ROOT / "mcp_server.py"))
        reg.register("scheduler", str(PROJECT_ROOT / "mcp_scheduler_server.py"))

        tools_logs = [_make_tool_schema("list_logs"), _make_tool_schema("read_log")]
        tools_sched = [_make_tool_schema("scheduler_add_task")]

        conn_logs = _mock_conn(tools_logs)
        conn_sched = _mock_conn(tools_sched)

        def get_conn(name):
            return conn_logs if name == "logs" else conn_sched

        with patch.object(reg, "_get_connection", side_effect=get_conn):
            reg.discover_tools()

        assert reg.get_server_for_tool("list_logs") == "logs"
        assert reg.get_server_for_tool("read_log") == "logs"
        assert reg.get_server_for_tool("scheduler_add_task") == "scheduler"
        assert len(reg.get_all_tool_names()) == 3

    def test_discover_tools_deduplication(self):
        """Если два сервера имеют инструмент с одинаковым именем,
        используется первый зарегистрированный."""
        from mcp_registry import MCPRegistry
        reg = MCPRegistry(verbose=False)
        reg.register("s1", str(PROJECT_ROOT / "mcp_server.py"))
        reg.register("s2", str(PROJECT_ROOT / "mcp_scheduler_server.py"))

        tools_s1 = [_make_tool_schema("shared_tool")]
        tools_s2 = [_make_tool_schema("shared_tool")]

        conn_s1 = _mock_conn(tools_s1)
        conn_s2 = _mock_conn(tools_s2)

        def get_conn(name):
            return conn_s1 if name == "s1" else conn_s2

        with patch.object(reg, "_get_connection", side_effect=get_conn):
            reg.discover_tools()

        assert reg.get_server_for_tool("shared_tool") == "s1"

    def test_get_tool_definitions_format(self):
        """Определения должны быть в формате OpenAI function calling."""
        from mcp_registry import MCPRegistry
        reg = MCPRegistry()
        reg.register("s1", str(PROJECT_ROOT / "mcp_server.py"))

        tools = [_make_tool_schema("my_tool", "My description", {"arg1": {"type": "string"}})]
        mock_conn = _mock_conn(tools)

        with patch.object(reg, "_get_connection", return_value=mock_conn):
            reg.discover_tools()

        defs = reg.get_tool_definitions()
        assert len(defs) == 1
        d = defs[0]
        assert d["type"] == "function"
        assert d["function"]["name"] == "my_tool"
        assert d["function"]["description"] == "My description"
        assert "arg1" in d["function"]["parameters"]["properties"]

    def test_get_server_for_unknown_tool(self):
        from mcp_registry import MCPRegistry
        reg = MCPRegistry()
        assert reg.get_server_for_tool("nonexistent") is None

    def test_list_servers_shows_tools(self):
        from mcp_registry import MCPRegistry
        reg = MCPRegistry()
        reg.register("s1", str(PROJECT_ROOT / "mcp_server.py"), tags=["test"])

        tools = [_make_tool_schema("t1"), _make_tool_schema("t2")]
        mock_conn = _mock_conn(tools)

        with patch.object(reg, "_get_connection", return_value=mock_conn):
            reg.discover_tools()

        servers = reg.list_servers()
        assert servers[0]["tool_count"] == 2
        assert sorted(servers[0]["tools"]) == ["t1", "t2"]
        assert servers[0]["discovered"] is True

    def test_close_clears_connections(self):
        from mcp_registry import MCPRegistry
        reg = MCPRegistry()
        reg.register("s1", str(PROJECT_ROOT / "mcp_server.py"))

        mock_conn = _mock_conn([_make_tool_schema("t1")])
        reg._connections["s1"] = mock_conn

        reg.close()
        mock_conn.close.assert_called_once()
        assert len(reg._connections) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. MCPRouter
# ══════════════════════════════════════════════════════════════════════════════

class TestMCPRouter:
    """Тесты маршрутизатора MCP-вызовов."""

    def _make_registry_with_tools(self, tool_server_map: dict):
        """Создаёт registry+router с моками для заданного маппинга tool→server."""
        from mcp_registry import MCPRegistry
        from mcp_router import MCPRouter

        reg = MCPRegistry()
        conns = {}

        for tool_name, server_name in tool_server_map.items():
            if server_name not in conns:
                reg.register(server_name, str(PROJECT_ROOT / "mcp_server.py"))
                conns[server_name] = _mock_conn([])

            reg._tool_to_server[tool_name] = server_name
            reg._connections[server_name] = conns[server_name]

        router = MCPRouter(reg)
        return router, reg, conns

    def test_route_to_correct_server(self):
        router, reg, conns = self._make_registry_with_tools({
            "list_logs": "logs",
            "scheduler_add_task": "scheduler",
        })

        conns["logs"].call_tool.return_value = "logs result"
        conns["scheduler"].call_tool.return_value = "scheduler result"

        r1 = router.call_tool("list_logs", {})
        assert r1 == "logs result"
        conns["logs"].call_tool.assert_called_once_with("list_logs", {})

        r2 = router.call_tool("scheduler_add_task", {"name": "test"})
        assert r2 == "scheduler result"
        conns["scheduler"].call_tool.assert_called_once_with(
            "scheduler_add_task", {"name": "test"}
        )

    def test_unknown_tool_returns_error(self):
        from mcp_registry import MCPRegistry
        from mcp_router import MCPRouter

        reg = MCPRegistry()
        router = MCPRouter(reg)
        result = router.call_tool("nonexistent_tool", {})
        assert "не найден" in result

    def test_call_history_tracked(self):
        router, reg, conns = self._make_registry_with_tools({
            "tool1": "server1",
            "tool2": "server2",
        })

        router.call_tool("tool1", {})
        router.call_tool("tool2", {"x": 1})
        router.call_tool("tool1", {})

        history = router.call_history
        assert len(history) == 3
        assert history[0].tool == "tool1"
        assert history[0].server == "server1"
        assert history[0].success is True
        assert history[1].tool == "tool2"
        assert history[1].server == "server2"

    def test_servers_used(self):
        router, reg, conns = self._make_registry_with_tools({
            "t1": "s1",
            "t2": "s2",
            "t3": "s1",
        })

        router.call_tool("t1", {})
        router.call_tool("t2", {})
        router.call_tool("t3", {})

        assert router.servers_used == ["s1", "s2"]

    def test_error_handling(self):
        router, reg, conns = self._make_registry_with_tools({
            "failing_tool": "broken_server",
        })
        conns["broken_server"].call_tool.side_effect = RuntimeError("connection lost")

        result = router.call_tool("failing_tool", {})
        assert "Ошибка" in result
        assert router.call_history[-1].success is False
        assert router.call_history[-1].error is not None

    def test_get_summary(self):
        router, reg, conns = self._make_registry_with_tools({
            "t1": "s1",
            "t2": "s2",
        })

        router.call_tool("t1", {})
        router.call_tool("t2", {})
        router.call_tool("nonexistent", {})  # will fail at router (unknown)

        # nonexistent goes through unknown tool path
        from mcp_registry import MCPRegistry
        from mcp_router import MCPRouter
        reg2 = MCPRegistry()
        router2 = MCPRouter(reg2)
        router2.call_tool("missing", {})

        summary = router2.get_summary()
        assert summary["total_calls"] == 1
        assert summary["errors"] == 1
        assert summary["successful"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. MCPOrchestratorAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestMCPOrchestratorAgent:
    """Тесты оркестрационного агента."""

    def _make_agent(self, tool_server_map: dict, llm_responses: list):
        """Создаёт агента с моками реестра, роутера и LLM."""
        from mcp_registry import MCPRegistry
        from mcp_orchestrator_agent import MCPOrchestratorAgent

        reg = MCPRegistry()
        conns = {}

        # Регистрируем серверы и tools
        for tool_name, server_name in tool_server_map.items():
            if server_name not in conns:
                reg.register(server_name, str(PROJECT_ROOT / "mcp_server.py"))
                conns[server_name] = _mock_conn([])
                reg._connections[server_name] = conns[server_name]

            reg._tool_to_server[tool_name] = server_name
            # Добавляем tool в серверную инфу для get_tool_definitions
            info = reg._servers[server_name]
            info.tools.append(_make_tool_schema(tool_name))
            info.tool_names.add(tool_name)
            info.discovered = True

        # Мок OpenAI клиента
        client = MagicMock()
        client.chat.completions.create.side_effect = llm_responses

        agent = MCPOrchestratorAgent(
            client=client, registry=reg,
            model="test-model", verbose=False,
        )

        return agent, conns, client

    def test_direct_answer_no_tools(self):
        """LLM отвечает напрямую без вызова инструментов."""
        resp = _mock_llm_response(content="Привет!")
        agent, conns, client = self._make_agent(
            {"list_logs": "logs"}, [resp]
        )

        answer = agent.chat("Привет")
        assert answer == "Привет!"
        assert agent.tool_calls_total == 0

    def test_single_tool_call(self):
        """LLM вызывает один инструмент, получает результат, отвечает."""
        tc = _mock_tool_call("tc1", "list_logs", {"filter": "2026"})
        resp1 = _mock_llm_response(tool_calls=[tc])
        resp2 = _mock_llm_response(content="Вот результат анализа логов")

        agent, conns, client = self._make_agent(
            {"list_logs": "logs"}, [resp1, resp2]
        )
        conns["logs"].call_tool.return_value = '["log1.json", "log2.json"]'

        answer = agent.chat("Покажи логи")
        assert "результат" in answer.lower() or "лог" in answer.lower()
        assert agent.tool_calls_total == 1
        conns["logs"].call_tool.assert_called_once_with("list_logs", {"filter": "2026"})

    def test_multi_server_flow(self):
        """LLM вызывает инструменты с разных серверов последовательно."""
        # Итерация 1: вызываем list_logs (logs сервер)
        tc1 = _mock_tool_call("tc1", "list_logs", {})
        resp1 = _mock_llm_response(tool_calls=[tc1])

        # Итерация 2: вызываем scheduler_list_tasks (scheduler сервер)
        tc2 = _mock_tool_call("tc2", "scheduler_list_tasks", {})
        resp2 = _mock_llm_response(tool_calls=[tc2])

        # Итерация 3: вызываем search (pipeline сервер)
        tc3 = _mock_tool_call("tc3", "search", {"query": "AI"})
        resp3 = _mock_llm_response(tool_calls=[tc3])

        # Итерация 4: финальный ответ
        resp4 = _mock_llm_response(content="Анализ завершён: 3 лога, 2 задачи, найдено 5 результатов")

        agent, conns, client = self._make_agent(
            {
                "list_logs": "logs",
                "scheduler_list_tasks": "scheduler",
                "search": "pipeline",
            },
            [resp1, resp2, resp3, resp4],
        )

        conns["logs"].call_tool.return_value = "3 файла найдено"
        conns["scheduler"].call_tool.return_value = "2 задачи активны"
        conns["pipeline"].call_tool.return_value = "5 результатов"

        answer = agent.chat("Сделай полный анализ системы")
        assert agent.tool_calls_total == 3

        # Проверяем что все 3 сервера были вызваны
        conns["logs"].call_tool.assert_called_once()
        conns["scheduler"].call_tool.assert_called_once()
        conns["pipeline"].call_tool.assert_called_once()

    def test_parallel_tool_calls_in_one_iteration(self):
        """LLM вызывает несколько инструментов за одну итерацию."""
        tc1 = _mock_tool_call("tc1", "list_logs", {})
        tc2 = _mock_tool_call("tc2", "scheduler_list_tasks", {})
        resp1 = _mock_llm_response(tool_calls=[tc1, tc2])
        resp2 = _mock_llm_response(content="Готово")

        agent, conns, client = self._make_agent(
            {"list_logs": "logs", "scheduler_list_tasks": "scheduler"},
            [resp1, resp2],
        )

        conns["logs"].call_tool.return_value = "logs data"
        conns["scheduler"].call_tool.return_value = "tasks data"

        answer = agent.chat("Покажи логи и задачи")
        assert answer == "Готово"
        assert agent.tool_calls_total == 2

    def test_token_usage_accumulated(self):
        """Токены суммируются по итерациям."""
        resp = _mock_llm_response(content="Ок")
        agent, _, _ = self._make_agent({"t1": "s1"}, [resp])

        agent.chat("Тест")
        usage = agent.token_usage
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    def test_max_iterations_limit(self):
        """При превышении лимита итераций возвращается сообщение об ошибке."""
        # Всегда возвращаем tool_calls — бесконечный цикл
        tc = _mock_tool_call("tc1", "t1", {})
        endless = _mock_llm_response(tool_calls=[tc])

        agent, conns, client = self._make_agent({"t1": "s1"}, [endless] * 20)

        answer = agent.chat("Бесконечная задача")
        assert "лимит" in answer.lower() or "Превышен" in answer

    def test_routing_summary(self):
        """routing_summary возвращает информацию о маршрутизации."""
        tc1 = _mock_tool_call("tc1", "list_logs", {})
        resp1 = _mock_llm_response(tool_calls=[tc1])
        resp2 = _mock_llm_response(content="Готово")

        agent, conns, client = self._make_agent(
            {"list_logs": "logs"}, [resp1, resp2]
        )

        agent.chat("Покажи логи")
        summary = agent.routing_summary
        assert summary["total_calls"] == 1
        assert summary["successful"] == 1
        assert "logs" in summary["servers_used"]

    def test_context_manager(self):
        """Поддержка with-блока."""
        resp = _mock_llm_response(content="Ок")
        agent, _, _ = self._make_agent({"t1": "s1"}, [resp])

        with agent:
            agent.chat("Тест")
        # close() вызван — не бросает исключений

    def test_malformed_tool_args(self):
        """LLM возвращает невалидный JSON в arguments — обрабатывается gracefully."""
        tc = MagicMock()
        tc.id = "tc1"
        tc.function.name = "list_logs"
        tc.function.arguments = "not valid json"

        resp1 = _mock_llm_response(tool_calls=[tc])
        resp2 = _mock_llm_response(content="Результат")

        agent, conns, client = self._make_agent(
            {"list_logs": "logs"}, [resp1, resp2]
        )
        conns["logs"].call_tool.return_value = "data"

        answer = agent.chat("Тест")
        assert answer == "Результат"
        # Вызов с пустыми args при невалидном JSON
        conns["logs"].call_tool.assert_called_once_with("list_logs", {})


# ══════════════════════════════════════════════════════════════════════════════
# 4. Интеграция — E2E с реальными MCP-серверами
# ══════════════════════════════════════════════════════════════════════════════

class _MCPTestHelper:
    """Утилиты для интеграционных тестов с реальными серверами."""

    @staticmethod
    def start_server(server_path: Path) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, str(server_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )

    @staticmethod
    def send_recv(proc: subprocess.Popen, method: str,
                  params: dict = None, msg_id: int = 1) -> dict:
        msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params:
            msg["params"] = params
        proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        raw = proc.stdout.readline()
        assert raw, f"Server {method} returned empty"
        return json.loads(raw)


@pytest.mark.integration
class TestRegistryIntegration:
    """E2E: регистрация реальных серверов и обнаружение инструментов."""

    def test_discover_logs_server(self):
        from mcp_registry import MCPRegistry

        with MCPRegistry() as reg:
            reg.register("logs", str(PROJECT_ROOT / "mcp_server.py"), tags=["logs", "memory"])
            reg.discover_tools()

            tool_names = reg.get_all_tool_names()
            assert "list_logs" in tool_names
            assert "search_logs" in tool_names
            assert "save_memory" in tool_names
            assert reg.get_server_for_tool("list_logs") == "logs"

    def test_discover_scheduler_server(self):
        from mcp_registry import MCPRegistry

        with MCPRegistry() as reg:
            reg.register("scheduler", str(PROJECT_ROOT / "mcp_scheduler_server.py"),
                         tags=["scheduler"])
            reg.discover_tools()

            tool_names = reg.get_all_tool_names()
            assert "scheduler_add_task" in tool_names
            assert "scheduler_list_tasks" in tool_names

    def test_discover_multiple_servers(self):
        from mcp_registry import MCPRegistry

        with MCPRegistry() as reg:
            reg.register("logs", str(PROJECT_ROOT / "mcp_server.py"), tags=["logs"])
            reg.register("scheduler", str(PROJECT_ROOT / "mcp_scheduler_server.py"),
                         tags=["scheduler"])
            reg.discover_tools()

            all_tools = reg.get_all_tool_names()
            # Должны быть инструменты с обоих серверов
            assert "list_logs" in all_tools
            assert "scheduler_add_task" in all_tools

            # Маппинг корректный
            assert reg.get_server_for_tool("list_logs") == "logs"
            assert reg.get_server_for_tool("scheduler_add_task") == "scheduler"

            # Определения в формате function calling
            defs = reg.get_tool_definitions()
            assert len(defs) >= 2
            names = {d["function"]["name"] for d in defs}
            assert "list_logs" in names
            assert "scheduler_add_task" in names


@pytest.mark.integration
class TestRouterIntegration:
    """E2E: маршрутизация вызовов через реальные серверы."""

    def test_route_call_to_logs_server(self):
        from mcp_registry import MCPRegistry
        from mcp_router import MCPRouter

        with MCPRegistry() as reg:
            reg.register("logs", str(PROJECT_ROOT / "mcp_server.py"))
            reg.discover_tools()

            router = MCPRouter(reg)
            result = router.call_tool("list_logs", {})
            # Должен вернуть строку (может быть пустой если нет логов)
            assert isinstance(result, str)
            assert router.servers_used == ["logs"]

    def test_route_calls_to_multiple_servers(self):
        from mcp_registry import MCPRegistry
        from mcp_router import MCPRouter

        with MCPRegistry() as reg:
            reg.register("logs", str(PROJECT_ROOT / "mcp_server.py"))
            reg.register("scheduler", str(PROJECT_ROOT / "mcp_scheduler_server.py"))
            reg.discover_tools()

            router = MCPRouter(reg)

            r1 = router.call_tool("list_logs", {})
            assert isinstance(r1, str)

            r2 = router.call_tool("scheduler_list_tasks", {})
            assert isinstance(r2, str)

            # Оба сервера были задействованы
            assert set(router.servers_used) == {"logs", "scheduler"}

            summary = router.get_summary()
            assert summary["total_calls"] == 2
            assert summary["successful"] == 2
            assert summary["errors"] == 0

    def test_route_unknown_tool(self):
        from mcp_registry import MCPRegistry
        from mcp_router import MCPRouter

        with MCPRegistry() as reg:
            reg.register("logs", str(PROJECT_ROOT / "mcp_server.py"))
            reg.discover_tools()

            router = MCPRouter(reg)
            result = router.call_tool("totally_fake_tool", {})
            assert "не найден" in result


@pytest.mark.integration
class TestOrchestratorIntegration:
    """E2E тест с тремя серверами (logs + scheduler + pipeline search)."""

    def test_three_server_discovery(self):
        """Реестр обнаруживает инструменты с 3 серверов."""
        from mcp_registry import MCPRegistry

        search_server = PROJECT_ROOT / "pipeline" / "servers" / "search_server.py"
        if not search_server.exists():
            pytest.skip("search_server.py not found")

        with MCPRegistry() as reg:
            reg.register("logs", str(PROJECT_ROOT / "mcp_server.py"), tags=["logs"])
            reg.register("scheduler", str(PROJECT_ROOT / "mcp_scheduler_server.py"),
                         tags=["scheduler"])
            reg.register("search", str(search_server), tags=["content"])
            reg.discover_tools()

            all_tools = reg.get_all_tool_names()
            servers = reg.list_servers()
            assert len(servers) == 3

            # Инструменты с каждого сервера обнаружены
            assert "list_logs" in all_tools
            assert "scheduler_list_tasks" in all_tools
            assert "search" in all_tools

    def test_cross_server_routing(self):
        """Роутер корректно направляет вызовы к 3 разным серверам."""
        from mcp_registry import MCPRegistry
        from mcp_router import MCPRouter

        search_server = PROJECT_ROOT / "pipeline" / "servers" / "search_server.py"
        if not search_server.exists():
            pytest.skip("search_server.py not found")

        with MCPRegistry() as reg:
            reg.register("logs", str(PROJECT_ROOT / "mcp_server.py"))
            reg.register("scheduler", str(PROJECT_ROOT / "mcp_scheduler_server.py"))
            reg.register("search", str(search_server))
            reg.discover_tools()

            router = MCPRouter(reg)

            # Вызываем инструменты с 3 серверов
            r1 = router.call_tool("list_logs", {})
            assert isinstance(r1, str)

            r2 = router.call_tool("scheduler_list_tasks", {})
            assert isinstance(r2, str)

            r3 = router.call_tool("search", {"query": "test"})
            assert isinstance(r3, str)

            # Все 3 сервера задействованы
            assert set(router.servers_used) == {"logs", "scheduler", "search"}

            summary = router.get_summary()
            assert summary["total_calls"] == 3
            assert summary["successful"] == 3
