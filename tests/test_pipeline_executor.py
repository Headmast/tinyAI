"""
Тесты для 4-серверного MCP Pipeline (День 19).

Покрываем:
  1. pipeline/db.py — CRUD контента и pipeline_runs
  2. pipeline/servers/ — 4 отдельных MCP-сервера (юнит + протокол)
  3. mcp_pipeline.py — исполнитель пайплайнов (подстановки, маршрутизация)
  4. mcp_pipeline_bridge.py — мост для AI-агентов
  5. Интеграция — полный цикл через 4 реальных MCP-сервера
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SERVERS_DIR = PROJECT_ROOT / "pipeline" / "servers"

sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# 1. pipeline/db.py
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineDB:
    """Тесты БД контента."""

    @pytest.fixture(autouse=True)
    def _setup_db(self, tmp_path):
        import pipeline.db as db_mod
        self._original_path = db_mod.DB_PATH
        db_mod.DB_PATH = tmp_path / "test_content.db"
        db_mod.init_db()
        yield
        db_mod.DB_PATH = self._original_path

    def test_init_creates_tables(self):
        from pipeline.db import get_connection
        with get_connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            names = {r["name"] for r in tables}
            assert "content_items" in names
            assert "pipeline_runs" in names

    def test_save_and_get_content(self):
        from pipeline.db import save_content, get_content
        cid = save_content(
            title="Тест", content="Контент", platform="telegram",
            tags=["ai", "test"], summary="Кратко",
        )
        assert cid >= 1
        item = get_content(cid)
        assert item is not None
        assert item["title"] == "Тест"
        assert item["platform"] == "telegram"
        assert item["tags"] == ["ai", "test"]
        assert item["status"] == "draft"

    def test_list_content_filter(self):
        from pipeline.db import save_content, list_content
        save_content(title="A", content="aaa", platform="telegram")
        save_content(title="B", content="bbb", platform="website")
        save_content(title="C", content="ccc", platform="telegram")
        assert len(list_content()) == 3
        tg = list_content(platform="telegram")
        assert len(tg) == 2

    def test_update_status(self):
        from pipeline.db import save_content, get_content, update_status
        cid = save_content(title="X", content="x", platform="plain")
        assert update_status(cid, "published")
        item = get_content(cid)
        assert item["status"] == "published"
        assert item["published_at"] is not None

    def test_get_content_not_found(self):
        from pipeline.db import get_content
        assert get_content(99999) is None

    def test_pipeline_run_lifecycle(self):
        from pipeline.db import (
            create_pipeline_run, finish_pipeline_run,
            get_pipeline_run, list_pipeline_runs,
        )
        run_id = "test1234"
        steps = [{"tool": "search", "args": {"query": "test"}}]
        create_pipeline_run(run_id, steps)
        assert get_pipeline_run(run_id)["status"] == "running"

        finish_pipeline_run(run_id, [{"tool": "search"}], status="completed", total_ms=500)
        run = get_pipeline_run(run_id)
        assert run["status"] == "completed"
        assert run["total_ms"] == 500
        assert len(list_pipeline_runs()) >= 1

    def test_pipeline_run_not_found(self):
        from pipeline.db import get_pipeline_run
        assert get_pipeline_run("nonexistent") is None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Серверы — юнит-тесты обработчиков (без MCP-протокола)
# ══════════════════════════════════════════════════════════════════════════════

class TestSearchServer:
    """Тесты search_server."""

    def test_empty_query(self):
        from pipeline.servers.search_server import handle_search
        result = json.loads(handle_search({"query": ""}))
        assert "error" in result

    def test_no_matches(self):
        from pipeline.servers.search_server import handle_search
        result = json.loads(handle_search({"query": "zzznonexistent42"}))
        assert result["found"] == 0


class TestFormatServer:
    """Тесты format_server."""

    def test_telegram(self):
        from pipeline.servers.format_server import handle_format_content
        result = json.loads(handle_format_content({
            "content": "Текст", "platform": "telegram", "title": "Заголовок",
        }))
        assert "<b>Заголовок</b>" in result["formatted"]
        assert result["platform"] == "telegram"

    def test_website(self):
        from pipeline.servers.format_server import handle_format_content
        result = json.loads(handle_format_content({
            "content": "P1\n\nP2", "platform": "website", "title": "T",
        }))
        assert "<html" in result["formatted"]
        assert "<h1>T</h1>" in result["formatted"]

    def test_rss(self):
        from pipeline.servers.format_server import handle_format_content
        result = json.loads(handle_format_content({
            "content": "RSS", "platform": "rss", "title": "RT",
        }))
        assert "<item>" in result["formatted"]
        assert "RT" in result["formatted"]

    def test_plain(self):
        from pipeline.servers.format_server import handle_format_content
        result = json.loads(handle_format_content({
            "content": "plain text", "platform": "plain",
        }))
        assert "plain text" in result["formatted"]

    def test_empty(self):
        from pipeline.servers.format_server import handle_format_content
        result = json.loads(handle_format_content({"content": "", "platform": "tg"}))
        assert "error" in result


class TestStoreServer:
    """Тесты store_server."""

    def test_save_to_file(self, tmp_path):
        from pipeline.servers.store_server import handle_save_to_file
        import pipeline.servers.store_server as srv
        orig = srv.POSTS_DIR
        srv.POSTS_DIR = tmp_path
        try:
            result = json.loads(handle_save_to_file({
                "content": "hello", "filename": "test.md",
            }))
            assert result["saved"] is True
            assert (tmp_path / "test.md").read_text() == "hello"
        finally:
            srv.POSTS_DIR = orig

    def test_save_to_file_path_traversal(self, tmp_path):
        from pipeline.servers.store_server import handle_save_to_file
        import pipeline.servers.store_server as srv
        orig = srv.POSTS_DIR
        srv.POSTS_DIR = tmp_path
        try:
            result = json.loads(handle_save_to_file({
                "content": "x", "filename": "../../etc/passwd",
            }))
            assert result["saved"] is True
            # Path.name strips traversal
        finally:
            srv.POSTS_DIR = orig

    def test_save_to_file_empty(self):
        from pipeline.servers.store_server import handle_save_to_file
        result = json.loads(handle_save_to_file({"content": "x", "filename": ""}))
        assert "error" in result

    def test_save_to_db(self, tmp_path):
        import pipeline.db as db_mod
        orig = db_mod.DB_PATH
        db_mod.DB_PATH = tmp_path / "test.db"
        db_mod.init_db()
        try:
            from pipeline.servers.store_server import handle_save_to_db
            result = json.loads(handle_save_to_db({
                "title": "T", "content": "C", "platform": "telegram", "tags": ["a"],
            }))
            assert result["saved"] is True
            assert result["content_id"] >= 1
        finally:
            db_mod.DB_PATH = orig

    def test_save_to_db_empty(self):
        from pipeline.servers.store_server import handle_save_to_db
        result = json.loads(handle_save_to_db({
            "title": "", "content": "", "platform": "plain",
        }))
        assert "error" in result


class TestSummarizeServer:
    """Тесты summarize_server."""

    def test_empty_text(self):
        from pipeline.servers.summarize_server import handle_summarize
        result = json.loads(handle_summarize({"text": ""}))
        assert "error" in result


# ══════════════════════════════════════════════════════════════════════════════
# 3. MCP JSON-RPC протокол (базовый каркас)
# ══════════════════════════════════════════════════════════════════════════════

class TestMCPServerBase:
    """Тесты базового MCP-каркаса через один из серверов."""

    def _make_server(self):
        from pipeline.servers.base import MCPServerBase
        return MCPServerBase(
            name="test-server", version="0.1",
            tools=[{"name": "echo", "description": "echo", "inputSchema": {}}],
            handlers={"echo": lambda args: json.dumps(args)},
        )

    def test_initialize(self):
        srv = self._make_server()
        resp = srv.handle_request({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "1"}},
        })
        assert resp["result"]["serverInfo"]["name"] == "test-server"

    def test_tools_list(self):
        srv = self._make_server()
        resp = srv.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert len(resp["result"]["tools"]) == 1

    def test_tools_call(self):
        srv = self._make_server()
        resp = srv.handle_request({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "echo", "arguments": {"msg": "hi"}},
        })
        text = resp["result"]["content"][0]["text"]
        assert json.loads(text) == {"msg": "hi"}

    def test_unknown_tool(self):
        srv = self._make_server()
        resp = srv.handle_request({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        })
        assert resp["result"]["isError"] is True

    def test_notification(self):
        srv = self._make_server()
        resp = srv.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert resp is None

    def test_unknown_method(self):
        srv = self._make_server()
        resp = srv.handle_request({"jsonrpc": "2.0", "id": 5, "method": "x/y"})
        assert resp["error"]["code"] == -32601


# ══════════════════════════════════════════════════════════════════════════════
# 4. mcp_pipeline.py — подстановки и маршрутизация
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineExecutor:
    """Тесты подстановок и логики пайплайна."""

    def test_resolve_prev_result(self):
        from mcp_pipeline import PipelineExecutor, StepResult
        ex = PipelineExecutor(verbose=False, save_to_db=False)
        prev = StepResult(step_index=0, tool="search", server="s",
                          raw='{"summary": "hello"}')
        prev.parsed = json.loads(prev.raw)
        result = ex._resolve_args({"text": "$prev.result"}, [prev])
        assert result["text"] == "hello"

    def test_resolve_steps_n_result(self):
        from mcp_pipeline import PipelineExecutor, StepResult
        ex = PipelineExecutor(verbose=False, save_to_db=False)
        s0 = StepResult(step_index=0, tool="search", server="s",
                        raw='{"results": "d0"}')
        s0.parsed = json.loads(s0.raw)
        s1 = StepResult(step_index=1, tool="summarize", server="s",
                        raw='{"summary": "s1"}')
        s1.parsed = json.loads(s1.raw)
        result = ex._resolve_args(
            {"a": "$steps.0.result", "b": "$steps.1.result"}, [s0, s1],
        )
        assert result["a"] == "d0"
        assert result["b"] == "s1"

    def test_resolve_json_field(self):
        from mcp_pipeline import PipelineExecutor, StepResult
        ex = PipelineExecutor(verbose=False, save_to_db=False)
        s0 = StepResult(step_index=0, tool="t", server="s",
                        raw='{"summary": "краткое", "style": "news"}')
        s0.parsed = json.loads(s0.raw)
        result = ex._resolve_args(
            {"x": "$steps.0.json.summary", "y": "$steps.0.json.style"}, [s0],
        )
        assert result["x"] == "краткое"
        assert result["y"] == "news"

    def test_resolve_no_substitution(self):
        from mcp_pipeline import PipelineExecutor
        ex = PipelineExecutor(verbose=False, save_to_db=False)
        result = ex._resolve_args({"q": "plain", "n": 42}, [])
        assert result == {"q": "plain", "n": 42}

    def test_resolve_list_values(self):
        from mcp_pipeline import PipelineExecutor, StepResult
        ex = PipelineExecutor(verbose=False, save_to_db=False)
        s0 = StepResult(step_index=0, tool="t", server="s",
                        raw='{"summary": "val"}')
        s0.parsed = json.loads(s0.raw)
        result = ex._resolve_args({"tags": ["fixed", "$prev.result"]}, [s0])
        assert result["tags"] == ["fixed", "val"]

    def test_extract_snippets(self):
        from mcp_pipeline import PipelineExecutor, StepResult
        ex = PipelineExecutor(verbose=False, save_to_db=False)
        sr = StepResult(step_index=0, tool="search", server="s", raw="")
        sr.parsed = {"results": [{"snippet": "A"}, {"snippet": "B"}], "found": 2}
        text = ex._extract_main_text(sr)
        assert "A" in text
        assert "B" in text

    def test_extract_priority(self):
        from mcp_pipeline import PipelineExecutor, StepResult
        ex = PipelineExecutor(verbose=False, save_to_db=False)
        sr = StepResult(step_index=0, tool="t", server="s", raw="")
        sr.parsed = {"summary": "S", "formatted": "F"}
        assert ex._extract_main_text(sr) == "S"  # summary first

    def test_extract_fallback_raw(self):
        from mcp_pipeline import PipelineExecutor, StepResult
        ex = PipelineExecutor(verbose=False, save_to_db=False)
        sr = StepResult(step_index=0, tool="t", server="s", raw="raw text")
        sr.parsed = None
        assert ex._extract_main_text(sr) == "raw text"

    def test_tool_to_server_mapping(self):
        from mcp_pipeline import TOOL_TO_SERVER
        assert "search" in TOOL_TO_SERVER
        assert "summarize" in TOOL_TO_SERVER
        assert "format_content" in TOOL_TO_SERVER
        assert "save_to_file" in TOOL_TO_SERVER
        assert "save_to_db" in TOOL_TO_SERVER
        # search и summarize -> разные серверы
        assert TOOL_TO_SERVER["search"] != TOOL_TO_SERVER["summarize"]
        # save_to_file и save_to_db -> один сервер
        assert TOOL_TO_SERVER["save_to_file"] == TOOL_TO_SERVER["save_to_db"]


# ══════════════════════════════════════════════════════════════════════════════
# 5. mcp_pipeline_bridge.py — мост для AI-агентов
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineBridge:
    """Тесты моста к 4 серверам."""

    def test_tool_definitions(self):
        from mcp_pipeline_bridge import PIPELINE_TOOL_DEFINITIONS, PIPELINE_TOOLS
        names = {t["function"]["name"] for t in PIPELINE_TOOL_DEFINITIONS}
        assert names == PIPELINE_TOOLS
        assert len(PIPELINE_TOOL_DEFINITIONS) == 5

    def test_unknown_tool(self):
        from mcp_pipeline_bridge import MCPPipelineBridge
        bridge = MCPPipelineBridge()
        result = bridge.call_tool("nonexistent", {})
        assert "Unknown tool" in result
        bridge.close()

    def test_tool_to_server_mapping(self):
        from mcp_pipeline_bridge import TOOL_TO_SERVER
        # 4 уникальных сервера
        unique_servers = set(str(v) for v in TOOL_TO_SERVER.values())
        assert len(unique_servers) == 4


# ══════════════════════════════════════════════════════════════════════════════
# 6. Интеграция: каждый MCP-сервер через subprocess
# ══════════════════════════════════════════════════════════════════════════════

class _MCPTestHelper:
    """Утилиты для тестирования MCP-серверов через subprocess."""

    def __init__(self):
        self._msg_counter = 0

    def send_recv(self, proc, method, params=None):
        self._msg_counter += 1
        msg = {"jsonrpc": "2.0", "id": self._msg_counter, "method": method}
        if params:
            msg["params"] = params
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        raw = proc.stdout.readline()
        return json.loads(raw)

    def start_server(self, server_file):
        return subprocess.Popen(
            [sys.executable, str(server_file)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )


@pytest.mark.integration
class TestSearchServerIntegration:
    """E2E: search_server через subprocess."""

    def test_handshake_and_search(self):
        h = _MCPTestHelper()
        proc = h.start_server(SERVERS_DIR / "search_server.py")
        try:
            resp = h.send_recv(proc, "initialize", {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            })
            assert resp["result"]["serverInfo"]["name"] == "tinyai-search-server"

            resp = h.send_recv(proc, "tools/list")
            names = [t["name"] for t in resp["result"]["tools"]]
            assert names == ["search"]

            resp = h.send_recv(proc, "tools/call", {
                "name": "search", "arguments": {"query": "xyznonexist"},
            })
            data = json.loads(resp["result"]["content"][0]["text"])
            assert data["found"] == 0
        finally:
            proc.stdin.close()
            proc.wait(timeout=5)


@pytest.mark.integration
class TestFormatServerIntegration:
    """E2E: format_server через subprocess."""

    def test_format_telegram(self):
        h = _MCPTestHelper()
        proc = h.start_server(SERVERS_DIR / "format_server.py")
        try:
            h.send_recv(proc, "initialize", {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            })
            resp = h.send_recv(proc, "tools/call", {
                "name": "format_content",
                "arguments": {"content": "News", "platform": "telegram", "title": "T"},
            })
            data = json.loads(resp["result"]["content"][0]["text"])
            assert "<b>T</b>" in data["formatted"]
        finally:
            proc.stdin.close()
            proc.wait(timeout=5)


@pytest.mark.integration
class TestStoreServerIntegration:
    """E2E: store_server через subprocess."""

    def test_save_to_file(self):
        h = _MCPTestHelper()
        proc = h.start_server(SERVERS_DIR / "store_server.py")
        try:
            h.send_recv(proc, "initialize", {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            })

            resp = h.send_recv(proc, "tools/list")
            names = [t["name"] for t in resp["result"]["tools"]]
            assert "save_to_file" in names
            assert "save_to_db" in names

            resp = h.send_recv(proc, "tools/call", {
                "name": "save_to_file",
                "arguments": {"content": "test", "filename": "integration_test.txt"},
            })
            data = json.loads(resp["result"]["content"][0]["text"])
            assert data["saved"] is True
        finally:
            proc.stdin.close()
            proc.wait(timeout=5)


@pytest.mark.integration
class TestSummarizeServerIntegration:
    """E2E: summarize_server — только handshake (LLM не вызываем)."""

    def test_handshake_and_tools_list(self):
        h = _MCPTestHelper()
        proc = h.start_server(SERVERS_DIR / "summarize_server.py")
        try:
            resp = h.send_recv(proc, "initialize", {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            })
            assert resp["result"]["serverInfo"]["name"] == "tinyai-summarize-server"

            resp = h.send_recv(proc, "tools/list")
            names = [t["name"] for t in resp["result"]["tools"]]
            assert names == ["summarize"]
        finally:
            proc.stdin.close()
            proc.wait(timeout=5)


@pytest.mark.integration
class TestPipelineBridgeIntegration:
    """E2E: MCPPipelineBridge вызывает реальные серверы."""

    def test_bridge_format_and_search(self):
        from mcp_pipeline_bridge import MCPPipelineBridge
        bridge = MCPPipelineBridge(verbose=False)
        try:
            # format через format_server
            result = bridge.call_tool("format_content", {
                "content": "Test", "platform": "telegram", "title": "T",
            })
            data = json.loads(result)
            assert "<b>T</b>" in data["formatted"]

            # search через search_server
            result = bridge.call_tool("search", {"query": "zzz_no_match"})
            data = json.loads(result)
            assert data["found"] == 0
        finally:
            bridge.close()
