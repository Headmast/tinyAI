"""
Интеграционные тесты MCP Files Server (запуск как subprocess).

Тестирует реальный JSON-RPC 2.0 обмен через stdin/stdout.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

MCP_SERVER = Path(__file__).parent.parent / "mcp_files_server.py"
PROJECT_ROOT = Path(__file__).parent.parent


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
class TestMCPFilesServerIntegration:
    """Integration-тесты: запуск MCP files server как subprocess."""

    def test_initialize(self):
        """Сервер отвечает на initialize."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        ])
        assert len(responses) == 1
        resp = responses[0]
        assert resp["id"] == 1
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert resp["result"]["serverInfo"]["name"] == "tinyai-files-server"

    def test_tools_list(self):
        """Сервер возвращает список из 6 инструментов."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ])
        tools_resp = responses[1]
        tools = tools_resp["result"]["tools"]
        assert len(tools) == 6
        names = {t["name"] for t in tools}
        assert names == {
            "list_project_files", "read_file", "search_in_files",
            "get_file_info", "write_file", "apply_diff",
        }

    def test_list_project_files(self):
        """list_project_files возвращает файлы корня проекта."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "list_project_files", "arguments": {"path": "."}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        # Должны быть известные файлы проекта
        assert "pyproject.toml" in text
        assert "README.md" in text
        # Не должно быть __pycache__
        assert "__pycache__" not in text

    def test_list_project_files_subdir(self):
        """list_project_files для подкаталога."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "list_project_files", "arguments": {"path": "core"}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        assert "config.py" in text

    def test_read_file(self):
        """read_file читает известный файл."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "read_file", "arguments": {"path": "pyproject.toml"}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        assert "pyproject.toml" in text
        assert "[project]" in text or "[tool" in text

    def test_read_file_max_lines(self):
        """read_file с ограничением строк."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "read_file", "arguments": {"path": "pyproject.toml", "max_lines": 3}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        assert "строки 1-3" in text

    def test_read_file_path_traversal(self):
        """read_file отклоняет path traversal."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "read_file", "arguments": {"path": "../../etc/passwd"}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        assert "Ошибка" in text
        assert "недопустимый" in text or "не найден" in text

    def test_read_file_not_found(self):
        """read_file для несуществующего файла."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "read_file", "arguments": {"path": "nonexistent_file_xyz.txt"}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        assert "Ошибка" in text
        assert "не найден" in text

    def test_search_in_files(self):
        """search_in_files находит известную строку."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "search_in_files", "arguments": {"query": "MCPStdioClient"}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        assert "Найдено совпадений" in text
        assert "mcp_stdio_client.py" in text

    def test_search_in_files_with_pattern(self):
        """search_in_files с file_pattern."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "search_in_files",
                "arguments": {"query": "def main", "file_pattern": "*.py"}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        assert "Найдено совпадений" in text

    def test_search_in_files_no_results(self):
        """search_in_files для несуществующего текста."""
        # Используем строку, которая гарантированно не встречается нигде в проекте
        unique = "q" * 50 + "_never_exists"
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "search_in_files",
                "arguments": {"query": unique}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        assert "Ничего не найдено" in text

    def test_get_file_info(self):
        """get_file_info возвращает метаданные."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "get_file_info", "arguments": {"path": "pyproject.toml"}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        info = json.loads(text)
        assert info["path"] == "pyproject.toml"
        assert info["size_bytes"] > 0
        assert info["is_binary"] is False
        assert "lines" in info

    def test_write_file(self):
        """write_file создаёт файл и возвращает успех."""
        test_path = "_test_tmp_write_file.txt"
        full_path = PROJECT_ROOT / test_path
        try:
            responses = _send_requests([
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                    "name": "write_file",
                    "arguments": {"path": test_path, "content": "Hello, test!\nLine 2"}
                }},
            ])
            text = responses[1]["result"]["content"][0]["text"]
            result = json.loads(text)
            assert result["success"] is True
            assert result["lines"] == 2

            # Проверяем что файл реально создан
            assert full_path.exists()
            assert full_path.read_text(encoding="utf-8") == "Hello, test!\nLine 2"
        finally:
            if full_path.exists():
                full_path.unlink()

    def test_write_file_protected(self):
        """write_file отклоняет запись в .env."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "write_file",
                "arguments": {"path": ".env", "content": "SECRET=hacked"}
            }},
        ])
        text = responses[1]["result"]["content"][0]["text"]
        assert "Ошибка" in text
        assert "защищённый" in text or "запрещена" in text

    def test_apply_diff(self):
        """apply_diff заменяет текст в файле."""
        test_path = "_test_tmp_apply_diff.txt"
        full_path = PROJECT_ROOT / test_path
        try:
            # Создаём файл
            full_path.write_text("aaa\nbbb\nccc\n", encoding="utf-8")

            responses = _send_requests([
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                    "name": "apply_diff",
                    "arguments": {
                        "path": test_path,
                        "old_text": "bbb",
                        "new_text": "BBB_REPLACED"
                    }
                }},
            ])
            text = responses[1]["result"]["content"][0]["text"]
            result = json.loads(text)
            assert result["success"] is True

            # Проверяем содержимое
            new_content = full_path.read_text(encoding="utf-8")
            assert "BBB_REPLACED" in new_content
            assert "bbb" not in new_content
        finally:
            if full_path.exists():
                full_path.unlink()

    def test_apply_diff_not_found(self):
        """apply_diff с несуществующим old_text."""
        test_path = "_test_tmp_apply_diff2.txt"
        full_path = PROJECT_ROOT / test_path
        try:
            full_path.write_text("hello world\n", encoding="utf-8")

            responses = _send_requests([
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                    "name": "apply_diff",
                    "arguments": {
                        "path": test_path,
                        "old_text": "nonexistent text",
                        "new_text": "replacement"
                    }
                }},
            ])
            text = responses[1]["result"]["content"][0]["text"]
            assert "Ошибка" in text
            assert "не найден" in text
        finally:
            if full_path.exists():
                full_path.unlink()

    def test_apply_diff_multiple_matches(self):
        """apply_diff отклоняет если old_text встречается более 1 раза."""
        test_path = "_test_tmp_apply_diff3.txt"
        full_path = PROJECT_ROOT / test_path
        try:
            full_path.write_text("aaa\naaa\nbbb\n", encoding="utf-8")

            responses = _send_requests([
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                    "name": "apply_diff",
                    "arguments": {
                        "path": test_path,
                        "old_text": "aaa",
                        "new_text": "AAA"
                    }
                }},
            ])
            text = responses[1]["result"]["content"][0]["text"]
            assert "Ошибка" in text
            assert "2 раз" in text or "раз" in text
        finally:
            if full_path.exists():
                full_path.unlink()

    def test_unknown_tool(self):
        """Вызов несуществующего инструмента возвращает ошибку."""
        responses = _send_requests([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "nonexistent_tool", "arguments": {}
            }},
        ])
        resp = responses[1]
        assert "error" in resp
        assert resp["error"]["code"] == -32601
