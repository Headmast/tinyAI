"""
Тесты для CodeReviewer — AI-ревью, поиск, анализ, исправления, рефакторинг.

Проверяет:
  - review_diff() со структурированным выводом
  - review_pr() с мокнутым MCP Git
  - search_code() — поиск + AI-анализ
  - analyze_file() — анализ файла
  - fix_code() — предложение исправлений
  - refactor() — предложение рефакторинга
  - Обрезку diff
  - Извлечение имён файлов
"""

import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_reviewer import CodeReviewer, MAX_DIFF_CHARS


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_DIFF = """\
diff --git a/core/config.py b/core/config.py
index abc1234..def5678 100644
--- a/core/config.py
+++ b/core/config.py
@@ -10,6 +10,7 @@
 DEFAULT_MODEL = "zai-org/GLM-4.7"
 DEFAULT_FAST_MODEL = "zai-org/GLM-4.7-Flash"
 DEFAULT_TEMPERATURE = 0.3
+NEW_CONSTANT = "test"

diff --git a/new_file.py b/new_file.py
new file mode 100644
--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,5 @@
+def hello():
+    print("hello")
+    # TODO: add error handling
+    x = eval(input())  # potential security issue
+    return x
"""

SAMPLE_REVIEW = """\
## 🐛 Потенциальные баги
1. `new_file.py`: использование `eval(input())` — критическая уязвимость

## 🏗️ Архитектурные проблемы
Не обнаружено

## 💡 Рекомендации
1. Заменить `eval()` на безопасный парсинг

## 📊 Общая оценка
Требует доработки: критическая уязвимость безопасности.
"""


@pytest.fixture
def mock_llm_client():
    """Мок OpenAI клиента, возвращающего ревью."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=SAMPLE_REVIEW))]
    )
    return client


SAMPLE_FILE_CODE = """\
import os
import subprocess

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True)
    return result.stdout.decode()

def get_key():
    key = os.getenv("SECRET_KEY", "hardcoded-secret")
    return key
"""

SAMPLE_GREP_RESULT = """\
Найдено совпадений: 3:
new_file.py:4:    x = eval(input())  # potential security issue
mcp_server.py:10:    result = eval(data)
core/config.py:5:# eval is not used here
"""

SAMPLE_ANALYSIS = """\
## 🐛 Баги и ошибки
1. shell=True в subprocess.run — уязвимость command injection

## ⚠️ Проблемы безопасности
1. Хардкод секрета: hardcoded-secret

## 🏗️ Архитектурные замечания
Не обнаружено

## 💡 Предложения по улучшению
1. Использовать shell=False

## 📊 Оценка качества
Требует доработки
"""


@pytest.fixture
def mock_llm_analysis():
    """Мок LLM клиента для анализа/fix/refactor."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=SAMPLE_ANALYSIS))]
    )
    return client


@pytest.fixture
def mock_git_client():
    """Мок _GitMCPClient с поддержкой всех инструментов."""
    git_client = MagicMock()

    def _call_tool(name, args=None):
        return {
            "git_diff": SAMPLE_DIFF,
            "git_status": "M core/config.py\nA new_file.py",
            "git_current_branch": "Текущая ветка: feature/test",
            "git_grep": SAMPLE_GREP_RESULT,
            "git_show_file": SAMPLE_FILE_CODE,
        }.get(name, "")

    git_client.call_tool.side_effect = _call_tool
    return git_client


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCodeReviewer:
    """Тесты CodeReviewer."""

    def test_extract_changed_filenames(self):
        """Извлечение имён файлов из diff."""
        reviewer = CodeReviewer()
        files = reviewer._extract_changed_filenames(SAMPLE_DIFF)
        assert "core/config.py" in files
        assert "new_file.py" in files
        assert len(files) == 2
        reviewer.close()

    def test_truncate_diff_short(self):
        """Короткий diff не обрезается."""
        result = CodeReviewer._truncate_diff("short diff", max_chars=1000)
        assert result == "short diff"

    def test_truncate_diff_long(self):
        """Длинный diff обрезается с пометкой."""
        long_diff = "x" * 10000
        result = CodeReviewer._truncate_diff(long_diff, max_chars=100)
        assert len(result) < len(long_diff)
        assert "обрезано" in result

    @patch("code_reviewer.get_llm_client")
    def test_review_diff(self, mock_get_client, mock_llm_client):
        """review_diff() возвращает структурированное ревью."""
        mock_get_client.return_value = mock_llm_client

        reviewer = CodeReviewer()
        result = reviewer.review_diff(SAMPLE_DIFF, "M core/config.py\nA new_file.py")

        assert "Потенциальные баги" in result
        assert "Архитектурные проблемы" in result
        assert "Рекомендации" in result
        assert "Общая оценка" in result

        # Проверяем, что LLM был вызван
        mock_llm_client.chat.completions.create.assert_called_once()
        call_args = mock_llm_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-5-nano"
        reviewer.close()

    @patch("code_reviewer.get_llm_client")
    def test_review_diff_empty(self, mock_get_client):
        """Пустой diff — сообщение 'нет изменений'."""
        reviewer = CodeReviewer()
        result = reviewer.review_diff("Нет изменений.")
        assert "Нет изменений" in result
        reviewer.close()

    @patch("code_reviewer.get_llm_client")
    def test_review_pr(self, mock_get_client, mock_llm_client, mock_git_client):
        """review_pr() получает diff через MCP и возвращает ревью."""
        mock_get_client.return_value = mock_llm_client

        reviewer = CodeReviewer()
        reviewer._git_client = mock_git_client

        result = reviewer.review_pr(base_branch="main")

        assert "AI Code Review" in result
        assert "main" in result
        assert "Потенциальные баги" in result

        # Проверяем вызовы MCP
        mock_git_client.call_tool.assert_any_call("git_diff", {"base_branch": "main"})
        mock_git_client.call_tool.assert_any_call("git_status")
        mock_git_client.call_tool.assert_any_call("git_current_branch")

        reviewer.close()

    @patch("code_reviewer.get_llm_client")
    def test_review_staged(self, mock_get_client, mock_llm_client, mock_git_client):
        """review_staged() ревьюит staged изменения."""
        mock_get_client.return_value = mock_llm_client

        reviewer = CodeReviewer()
        reviewer._git_client = mock_git_client

        result = reviewer.review_staged()

        assert "staged" in result.lower()
        mock_git_client.call_tool.assert_any_call("git_diff", {"staged": True})

        reviewer.close()

    @patch("code_reviewer.get_llm_client")
    def test_review_working(self, mock_get_client, mock_llm_client, mock_git_client):
        """review_working() ревьюит рабочую директорию."""
        mock_get_client.return_value = mock_llm_client

        reviewer = CodeReviewer()
        reviewer._git_client = mock_git_client

        result = reviewer.review_working()

        assert "AI Code Review" in result
        mock_git_client.call_tool.assert_any_call("git_diff", {})

        reviewer.close()

    def test_model_default(self):
        """Модель по умолчанию — gpt-5-nano."""
        reviewer = CodeReviewer()
        assert reviewer.model == "gpt-5-nano"
        reviewer.close()

    @patch("code_reviewer.get_llm_client")
    def test_llm_error_handling(self, mock_get_client):
        """Ошибка LLM возвращается как текст, не бросает исключение."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API timeout")
        mock_get_client.return_value = mock_client

        reviewer = CodeReviewer()
        result = reviewer.review_diff(SAMPLE_DIFF)

        assert "Ошибка LLM" in result
        reviewer.close()


class TestCodeSearch:
    """Тесты search_code()."""

    @patch("code_reviewer.get_llm_client")
    def test_search_code(self, mock_get_client, mock_llm_analysis, mock_git_client):
        """search_code() ищет паттерн и возвращает AI-анализ."""
        mock_get_client.return_value = mock_llm_analysis

        reviewer = CodeReviewer()
        reviewer._git_client = mock_git_client

        result = reviewer.search_code("eval(", file_pattern="*.py")

        assert "Поиск по коду" in result
        assert "eval(" in result
        mock_git_client.call_tool.assert_any_call(
            "git_grep", {"pattern": "eval(", "ignore_case": True, "file_pattern": "*.py"}
        )
        reviewer.close()

    @patch("code_reviewer.get_llm_client")
    def test_search_code_not_found(self, mock_get_client):
        """Поиск без результатов."""
        git_client = MagicMock()
        git_client.call_tool.return_value = "Ничего не найдено по запросу: xyz"

        reviewer = CodeReviewer()
        reviewer._git_client = git_client

        result = reviewer.search_code("xyz")
        assert "Ничего не найдено" in result
        reviewer.close()


class TestAnalyzeFile:
    """Тесты analyze_file()."""

    @patch("code_reviewer.get_llm_client")
    def test_analyze_file(self, mock_get_client, mock_llm_analysis, mock_git_client):
        """analyze_file() анализирует файл и возвращает отчёт."""
        mock_get_client.return_value = mock_llm_analysis

        reviewer = CodeReviewer()
        reviewer._git_client = mock_git_client

        result = reviewer.analyze_file("core/config.py")

        assert "Анализ файла" in result
        assert "core/config.py" in result
        mock_git_client.call_tool.assert_any_call("git_show_file", {"path": "core/config.py"})
        reviewer.close()

    @patch("code_reviewer.get_llm_client")
    def test_analyze_file_not_found(self, mock_get_client):
        """Ошибка при несуществующем файле."""
        git_client = MagicMock()
        git_client.call_tool.return_value = "Ошибка git: fatal: path 'xxx' does not exist"

        reviewer = CodeReviewer()
        reviewer._git_client = git_client

        result = reviewer.analyze_file("xxx")
        assert "Ошибка" in result
        reviewer.close()


class TestFixCode:
    """Тесты fix_code()."""

    @patch("code_reviewer.get_llm_client")
    def test_fix_code(self, mock_get_client, mock_llm_analysis, mock_git_client):
        """fix_code() предлагает исправления."""
        mock_get_client.return_value = mock_llm_analysis

        reviewer = CodeReviewer()
        reviewer._git_client = mock_git_client

        result = reviewer.fix_code("mcp_server.py")

        assert "Исправления" in result
        assert "mcp_server.py" in result
        reviewer.close()

    @patch("code_reviewer.get_llm_client")
    def test_fix_code_with_issue(self, mock_get_client, mock_llm_analysis, mock_git_client):
        """fix_code() с описанием конкретной проблемы."""
        mock_get_client.return_value = mock_llm_analysis

        reviewer = CodeReviewer()
        reviewer._git_client = mock_git_client

        result = reviewer.fix_code("mcp_server.py", issue="утечка ресурсов")

        assert "Исправления" in result
        assert "утечка ресурсов" in result
        reviewer.close()


class TestRefactor:
    """Тесты refactor()."""

    @patch("code_reviewer.get_llm_client")
    def test_refactor(self, mock_get_client, mock_llm_analysis, mock_git_client):
        """refactor() предлагает рефакторинг."""
        mock_get_client.return_value = mock_llm_analysis

        reviewer = CodeReviewer()
        reviewer._git_client = mock_git_client

        result = reviewer.refactor("mcp_server.py")

        assert "Рефакторинг" in result
        assert "mcp_server.py" in result
        reviewer.close()

    @patch("code_reviewer.get_llm_client")
    def test_refactor_with_goal(self, mock_get_client, mock_llm_analysis, mock_git_client):
        """refactor() с указанной целью."""
        mock_get_client.return_value = mock_llm_analysis

        reviewer = CodeReviewer()
        reviewer._git_client = mock_git_client

        result = reviewer.refactor("mcp_server.py", goal="разделить на модули")

        assert "Рефакторинг" in result
        assert "разделить на модули" in result
        reviewer.close()
