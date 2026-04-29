"""
Тесты для Support Assistant: MCP-сервер, ассистент, индексация.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════════════════════
# Тесты MCP Support Server (unit — без subprocess)
# ══════════════════════════════════════════════════════════════════════════════

class TestMCPSupportServerTools:
    """Unit-тесты инструментов MCP support server."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import mcp_support_server as server
        self.server = server

    @pytest.mark.unit
    def test_get_user_found(self):
        result = self.server.tool_get_user({"user_id": "user_1"})
        data = json.loads(result)
        assert data["user_id"] == "user_1"
        assert data["name"] == "Алексей Петров"
        assert data["plan"] == "pro"

    @pytest.mark.unit
    def test_get_user_not_found(self):
        result = self.server.tool_get_user({"user_id": "nonexistent"})
        assert "не найден" in result

    @pytest.mark.unit
    def test_get_user_missing_id(self):
        result = self.server.tool_get_user({})
        assert "Ошибка" in result

    @pytest.mark.unit
    def test_get_ticket_found(self):
        result = self.server.tool_get_ticket({"ticket_id": "ticket_1"})
        data = json.loads(result)
        assert data["ticket_id"] == "ticket_1"
        assert data["user_id"] == "user_1"
        assert len(data["messages"]) == 3

    @pytest.mark.unit
    def test_get_ticket_not_found(self):
        result = self.server.tool_get_ticket({"ticket_id": "nonexistent"})
        assert "не найден" in result

    @pytest.mark.unit
    def test_get_user_tickets(self):
        result = self.server.tool_get_user_tickets({"user_id": "user_1"})
        data = json.loads(result)
        assert len(data) == 2  # user_1 has ticket_1 and ticket_5
        assert all(t["ticket_id"].startswith("ticket_") for t in data)

    @pytest.mark.unit
    def test_get_user_tickets_none(self):
        result = self.server.tool_get_user_tickets({"user_id": "user_999"})
        assert "нет тикетов" in result

    @pytest.mark.unit
    def test_search_tickets_by_subject(self):
        result = self.server.tool_search_tickets({"query": "авторизация"})
        data = json.loads(result)
        assert len(data) >= 1
        assert any("авторизация" in t["subject"].lower() or "авторизация" in t.get("tags", []) for t in data)

    @pytest.mark.unit
    def test_search_tickets_with_status_filter(self):
        result = self.server.tool_search_tickets({"query": "оплата", "status": "resolved"})
        data = json.loads(result)
        assert all(t["status"] == "resolved" for t in data)

    @pytest.mark.unit
    def test_search_tickets_no_results(self):
        result = self.server.tool_search_tickets({"query": "несуществующий_запрос_xyz"})
        assert "не найдены" in result

    @pytest.mark.unit
    def test_get_active_tickets(self):
        result = self.server.tool_get_active_tickets({})
        data = json.loads(result)
        assert all(t["status"] in ("open", "in_progress") for t in data)
        assert len(data) >= 4  # at least 4 active tickets in test data

    @pytest.mark.unit
    def test_handle_request_initialize(self, capsys):
        self.server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        })
        output = capsys.readouterr().out.strip()
        resp = json.loads(output)
        assert resp["id"] == 1
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert resp["result"]["serverInfo"]["name"] == "tinyai-support-server"

    @pytest.mark.unit
    def test_handle_request_tools_list(self, capsys):
        self.server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        })
        output = capsys.readouterr().out.strip()
        resp = json.loads(output)
        tools = resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        assert "get_user" in tool_names
        assert "get_ticket" in tool_names
        assert "get_user_tickets" in tool_names
        assert "search_tickets" in tool_names
        assert "get_active_tickets" in tool_names
        assert len(tools) == 5

    @pytest.mark.unit
    def test_handle_request_tools_call(self, capsys):
        self.server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_user", "arguments": {"user_id": "user_2"}}
        })
        output = capsys.readouterr().out.strip()
        resp = json.loads(output)
        text = resp["result"]["content"][0]["text"]
        data = json.loads(text)
        assert data["name"] == "Мария Иванова"

    @pytest.mark.unit
    def test_handle_request_unknown_tool(self, capsys):
        self.server.handle_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}}
        })
        output = capsys.readouterr().out.strip()
        resp = json.loads(output)
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    @pytest.mark.unit
    def test_handle_request_unknown_method(self, capsys):
        self.server.handle_request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "unknown/method",
            "params": {}
        })
        output = capsys.readouterr().out.strip()
        resp = json.loads(output)
        assert "error" in resp


# ══════════════════════════════════════════════════════════════════════════════
# Тесты SupportAssistant (unit — мокнутый LLM)
# ══════════════════════════════════════════════════════════════════════════════

class TestSupportAssistant:
    """Unit-тесты SupportAssistant с мокнутым LLM и MCP."""

    @pytest.fixture
    def mock_llm_response(self):
        """Мок ответа LLM."""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Тестовый ответ поддержки."
        mock_resp.choices[0].message.reasoning = None
        return mock_resp

    @pytest.fixture
    def assistant(self, tmp_path):
        """SupportAssistant с тестовым индексом."""
        with patch("support_assistant.OpenAIEmbedder"), \
             patch("support_assistant.get_llm_client"):
            from support_assistant import SupportAssistant
            return SupportAssistant(index_dir=tmp_path, verbose=False)

    @pytest.mark.unit
    def test_search_faq_no_index(self, assistant):
        """Без индекса _search_faq возвращает пустой список."""
        results = assistant._search_faq("тест")
        assert results == []

    @pytest.mark.unit
    def test_answer_without_context(self, assistant, mock_llm_response):
        """Ответ без user/ticket контекста."""
        assistant._llm_client = MagicMock()
        assistant._llm_client.chat.completions.create.return_value = mock_llm_response
        with patch("support_assistant.rag_search", return_value=[]):
            answer = assistant.answer("Как сбросить пароль?")
        assert "Тестовый ответ поддержки" in answer

    @pytest.mark.unit
    def test_answer_with_user_context(self, assistant, mock_llm_response):
        """Ответ с контекстом пользователя — MCP вызывается."""
        mock_mcp = MagicMock()
        mock_mcp.call_tool.return_value = '{"user_id": "user_1", "plan": "pro"}'
        assistant._llm_client = MagicMock()
        assistant._llm_client.chat.completions.create.return_value = mock_llm_response

        with patch("support_assistant.rag_search", return_value=[]):
            assistant._mcp_client = mock_mcp
            answer = assistant.answer("Проблема", user_id="user_1")

        # MCP was called with get_user and get_user_tickets
        calls = mock_mcp.call_tool.call_args_list
        assert any(c[0][0] == "get_user" for c in calls)
        assert any(c[0][0] == "get_user_tickets" for c in calls)
        assert "Тестовый ответ поддержки" in answer

    @pytest.mark.unit
    def test_answer_with_ticket_context(self, assistant, mock_llm_response):
        """Ответ с контекстом тикета — MCP вызывается."""
        mock_mcp = MagicMock()
        mock_mcp.call_tool.return_value = '{"ticket_id": "ticket_1", "subject": "Test"}'
        assistant._llm_client = MagicMock()
        assistant._llm_client.chat.completions.create.return_value = mock_llm_response

        with patch("support_assistant.rag_search", return_value=[]):
            assistant._mcp_client = mock_mcp
            answer = assistant.answer("Помогите", ticket_id="ticket_1")

        calls = mock_mcp.call_tool.call_args_list
        assert any(c[0][0] == "get_ticket" for c in calls)
        assert "Тестовый ответ поддержки" in answer

    @pytest.mark.unit
    def test_answer_llm_error(self, assistant):
        """При ошибке LLM возвращается сообщение об ошибке."""
        assistant._llm_client = MagicMock()
        assistant._llm_client.chat.completions.create.side_effect = Exception("API error")
        with patch("support_assistant.rag_search", return_value=[]):
            answer = assistant.answer("тест")
        assert "Ошибка LLM" in answer

    @pytest.mark.unit
    def test_close(self, assistant):
        """close() завершает MCP-клиент."""
        mock_mcp = MagicMock()
        assistant._mcp_client = mock_mcp
        assistant.close()
        mock_mcp.close.assert_called_once()
        assert assistant._mcp_client is None


# ══════════════════════════════════════════════════════════════════════════════
# Тесты MCPStdioClient (unit)
# ══════════════════════════════════════════════════════════════════════════════

class TestMCPStdioClient:
    """Unit-тесты MCPStdioClient."""

    @pytest.mark.unit
    def test_context_manager(self):
        """Клиент работает как context manager."""
        from mcp_stdio_client import MCPStdioClient
        client = MCPStdioClient(server_path="fake_server.py")
        # close() без запущенного процесса — не падает
        client.close()
        assert client._proc is None

    @pytest.mark.unit
    def test_close_without_start(self):
        """close() без запуска — безопасен."""
        from mcp_stdio_client import MCPStdioClient
        client = MCPStdioClient(server_path="test.py")
        client.close()  # Should not raise


# ══════════════════════════════════════════════════════════════════════════════
# Тесты SupportIndex (unit — мокнутый embedder)
# ══════════════════════════════════════════════════════════════════════════════

class TestSupportIndex:
    """Unit-тесты индексации FAQ."""

    @pytest.mark.unit
    def test_load_support_docs(self):
        """Загрузка всех .md из support_data/."""
        from support_index import load_support_docs
        docs = load_support_docs()
        assert len(docs) >= 3  # faq.md, product_guide.md, troubleshooting.md, getting_started.md
        sources = [d.metadata["source"] for d in docs]
        assert any("faq.md" in s for s in sources)
        assert any("product_guide.md" in s for s in sources)
        assert any("troubleshooting.md" in s for s in sources)

    @pytest.mark.unit
    def test_load_support_docs_missing_dir(self, tmp_path, monkeypatch):
        """Если директория отсутствует — пустой список."""
        import support_index
        monkeypatch.setattr(support_index, "SUPPORT_DATA_DIR", tmp_path / "nonexistent")
        docs = support_index.load_support_docs()
        assert docs == []


# ══════════════════════════════════════════════════════════════════════════════
# Тесты DevAssistant refactoring (проверяем, что не сломали)
# ══════════════════════════════════════════════════════════════════════════════

class TestDevAssistantRefactored:
    """Проверяем, что DevAssistant всё ещё работает после рефакторинга."""

    @pytest.mark.unit
    def test_import(self):
        """DevAssistant импортируется без ошибок."""
        from dev_assistant import DevAssistant
        assert DevAssistant is not None

    @pytest.mark.unit
    def test_uses_mcp_stdio_client(self):
        """DevAssistant использует MCPStdioClient."""
        from dev_assistant import DevAssistant
        from mcp_stdio_client import MCPStdioClient
        assistant = DevAssistant()
        client = assistant._get_git_client()
        assert isinstance(client, MCPStdioClient)
        assistant.close()

    @pytest.mark.unit
    def test_search_docs_no_index(self, tmp_path):
        """Без индекса _search_docs возвращает пустой список."""
        from dev_assistant import DevAssistant
        assistant = DevAssistant(index_dir=tmp_path)
        results = assistant._search_docs("тест")
        assert results == []
