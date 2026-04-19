"""
Тесты для MCPSchedulerAgent: init, chat (function calling), tool routing, token tracking.
Все тесты работают offline (mock).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_scheduler_agent import (
    MCPSchedulerAgent,
    ALL_TOOL_DEFINITIONS,
    SYSTEM_PROMPT,
    MAX_TOOL_ITERATIONS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_response(content="ok", tool_calls=None, usage=None):
    """Создаёт фейковый OpenAI response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage or MagicMock(
        prompt_tokens=10, completion_tokens=20, total_tokens=30
    )
    return resp


def _make_tool_call(name, arguments=None, call_id="call_001"):
    """Создаёт фейковый tool_call."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments or {})
    return tc


def _make_agent():
    """Создаёт MCPSchedulerAgent с мок-клиентом."""
    mock_client = MagicMock()
    with patch("mcp_scheduler_agent.MCPBridge"), \
         patch("mcp_scheduler_agent.MCPSchedulerBridge"):
        agent = MCPSchedulerAgent(
            client=mock_client,
            model="test-model",
            verbose=False,
        )
    return agent, mock_client


# ── TestInit ──────────────────────────────────────────────────────────────────

class TestInit:
    def test_agent_creates_with_defaults(self):
        agent, _ = _make_agent()
        assert agent.model == "test-model"
        assert agent.verbose is False

    def test_system_prompt_in_messages(self):
        agent, _ = _make_agent()
        assert len(agent._messages) == 1
        assert agent._messages[0]["role"] == "system"
        assert agent._messages[0]["content"] == SYSTEM_PROMPT

    def test_initial_token_usage_zero(self):
        agent, _ = _make_agent()
        usage = agent.token_usage
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0

    def test_initial_tool_calls_zero(self):
        agent, _ = _make_agent()
        assert agent.tool_calls_total == 0


# ── TestToolDefinitions ───────────────────────────────────────────────────────

class TestToolDefinitions:
    def test_all_tools_have_function_type(self):
        for tool in ALL_TOOL_DEFINITIONS:
            assert tool["type"] == "function"

    def test_all_tools_have_name(self):
        for tool in ALL_TOOL_DEFINITIONS:
            assert "name" in tool["function"]
            assert len(tool["function"]["name"]) > 0

    def test_scheduler_tools_present(self):
        names = {t["function"]["name"] for t in ALL_TOOL_DEFINITIONS}
        expected = {
            "scheduler_add_task",
            "scheduler_list_tasks",
            "scheduler_remove_task",
            "scheduler_get_summary",
        }
        assert expected.issubset(names)


# ── TestChat ──────────────────────────────────────────────────────────────────

class TestChat:
    def test_simple_response_no_tools(self):
        agent, mock_client = _make_agent()
        mock_client.chat.completions.create.return_value = _make_response("Hello!")

        result = agent.chat("hi")
        assert result == "Hello!"

    def test_user_message_appended(self):
        agent, mock_client = _make_agent()
        mock_client.chat.completions.create.return_value = _make_response("ok")

        agent.chat("test input")

        # messages: system, user, assistant
        assert agent._messages[1]["role"] == "user"
        assert agent._messages[1]["content"] == "test input"

    def test_assistant_message_appended(self):
        agent, mock_client = _make_agent()
        mock_client.chat.completions.create.return_value = _make_response("response text")

        agent.chat("q")

        assert agent._messages[-1]["role"] == "assistant"
        assert agent._messages[-1]["content"] == "response text"

    def test_token_usage_accumulated(self):
        agent, mock_client = _make_agent()
        usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        mock_client.chat.completions.create.return_value = _make_response("ok", usage=usage)

        agent.chat("q")

        assert agent.token_usage["total_tokens"] == 150

    def test_multiple_chats_accumulate_tokens(self):
        agent, mock_client = _make_agent()
        usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        mock_client.chat.completions.create.return_value = _make_response("ok", usage=usage)

        agent.chat("q1")
        agent.chat("q2")

        assert agent.token_usage["total_tokens"] == 60


# ── TestToolCalling ───────────────────────────────────────────────────────────

class TestToolCalling:
    def test_scheduler_tool_routed_to_sched_bridge(self):
        agent, mock_client = _make_agent()

        tc = _make_tool_call("scheduler_list_tasks")
        resp_with_tools = _make_response(content="", tool_calls=[tc])
        resp_final = _make_response("Done!")

        mock_client.chat.completions.create.side_effect = [resp_with_tools, resp_final]
        agent._sched_bridge.call_tool.return_value = '{"tasks": []}'

        result = agent.chat("list tasks")

        agent._sched_bridge.call_tool.assert_called_once_with("scheduler_list_tasks", {})
        assert result == "Done!"

    def test_log_tool_routed_to_logs_bridge(self):
        agent, mock_client = _make_agent()

        tc = _make_tool_call("list_logs")
        resp_with_tools = _make_response(content="", tool_calls=[tc])
        resp_final = _make_response("Logs listed")

        mock_client.chat.completions.create.side_effect = [resp_with_tools, resp_final]
        agent._logs_bridge.call_tool.return_value = '["log1.json"]'

        result = agent.chat("show logs")

        agent._logs_bridge.call_tool.assert_called_once_with("list_logs", {})
        assert result == "Logs listed"

    def test_tool_call_count_incremented(self):
        agent, mock_client = _make_agent()

        tc1 = _make_tool_call("scheduler_list_tasks", call_id="c1")
        tc2 = _make_tool_call("scheduler_get_summary", call_id="c2")
        resp_with_tools = _make_response(content="", tool_calls=[tc1, tc2])
        resp_final = _make_response("Done")

        mock_client.chat.completions.create.side_effect = [resp_with_tools, resp_final]
        agent._sched_bridge.call_tool.return_value = "{}"

        agent.chat("q")

        assert agent.tool_calls_total == 2

    def test_tool_with_arguments(self):
        agent, mock_client = _make_agent()

        tc = _make_tool_call("scheduler_add_task", {"name": "test", "interval": 300})
        resp_with_tools = _make_response(content="", tool_calls=[tc])
        resp_final = _make_response("Task added")

        mock_client.chat.completions.create.side_effect = [resp_with_tools, resp_final]
        agent._sched_bridge.call_tool.return_value = '{"ok": true}'

        result = agent.chat("add task")

        agent._sched_bridge.call_tool.assert_called_once_with(
            "scheduler_add_task", {"name": "test", "interval": 300}
        )
        assert result == "Task added"

    def test_max_iterations_exceeded(self):
        agent, mock_client = _make_agent()

        tc = _make_tool_call("scheduler_list_tasks")
        resp_with_tools = _make_response(content="", tool_calls=[tc])
        agent._sched_bridge.call_tool.return_value = "{}"

        # Always return tool calls — should stop after MAX_TOOL_ITERATIONS
        mock_client.chat.completions.create.return_value = resp_with_tools

        result = agent.chat("q")

        assert "лимит" in result.lower() or "итераций" in result.lower()
        assert mock_client.chat.completions.create.call_count == MAX_TOOL_ITERATIONS


# ── TestClose ─────────────────────────────────────────────────────────────────

class TestClose:
    def test_close_calls_both_bridges(self):
        agent, _ = _make_agent()
        agent.close()
        agent._logs_bridge.close.assert_called_once()
        agent._sched_bridge.close.assert_called_once()
