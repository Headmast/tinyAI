"""
Интеграционные тесты MCP Support Server (запуск как subprocess).

Тестирует реальный JSON-RPC 2.0 обмен через stdin/stdout.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

MCP_SERVER = Path(__file__).parent.parent / "mcp_support_server.py"


def _send_requests(requests: list[dict]) -> list[dict]:
    """Отправляет JSON-RPC запросы в MCP-сервер и возвращает ответы."""
    input_lines = "\n".join(json.dumps(r) for r in requests) + "\n"
    proc = subprocess.run(
        [sys.executable, str(MCP_SERVER)],
        input=input_lines,
        capture_output=True,
        text=True,
        timeout=10,
    )
    responses = []
    for line in proc.stdout.strip().splitlines():
        if line.strip():
            responses.append(json.loads(line))
    return responses


@pytest.mark.integration
class TestMCPSupportServerIntegration:
    """Integration-тесты: запуск MCP support server как subprocess."""

    def test_initialize(self):
        """Сервер отвечает на initialize."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        ])
        assert len(responses) == 1
        resp = responses[0]
        assert resp["id"] == 1
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert resp["result"]["serverInfo"]["name"] == "tinyai-support-server"

    def test_tools_list(self):
        """Сервер возвращает список из 5 инструментов."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ])
        tools_resp = responses[1]
        tools = tools_resp["result"]["tools"]
        assert len(tools) == 5
        names = {t["name"] for t in tools}
        assert names == {"get_user", "get_user_tickets", "get_ticket", "search_tickets", "get_active_tickets"}

    def test_get_user(self):
        """tools/call get_user возвращает профиль пользователя."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "get_user", "arguments": {"user_id": "user_3"}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        data = json.loads(text)
        assert data["name"] == "Дмитрий Сидоров"
        assert data["plan"] == "enterprise"

    def test_get_ticket(self):
        """tools/call get_ticket возвращает тикет с сообщениями."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "get_ticket", "arguments": {"ticket_id": "ticket_3"}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        data = json.loads(text)
        assert data["ticket_id"] == "ticket_3"
        assert data["status"] == "in_progress"
        assert len(data["messages"]) == 3

    def test_get_user_tickets(self):
        """tools/call get_user_tickets возвращает тикеты пользователя."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "get_user_tickets", "arguments": {"user_id": "user_3"}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        data = json.loads(text)
        assert len(data) == 2  # ticket_3 and ticket_7
        assert all(isinstance(t, dict) for t in data)

    def test_search_tickets(self):
        """tools/call search_tickets находит тикеты по ключевому слову."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "search_tickets", "arguments": {"query": "авторизация"}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        data = json.loads(text)
        assert len(data) >= 1

    def test_get_active_tickets(self):
        """tools/call get_active_tickets возвращает открытые тикеты."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "get_active_tickets", "arguments": {}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        data = json.loads(text)
        assert all(t["status"] in ("open", "in_progress") for t in data)

    def test_unknown_tool_error(self):
        """Вызов несуществующего инструмента — ошибка."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "nonexistent", "arguments": {}
            }},
        ])
        assert "error" in responses[1]
        assert responses[1]["error"]["code"] == -32601
