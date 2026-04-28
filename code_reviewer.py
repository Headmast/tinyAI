"""
Code Reviewer — автоматическое AI-ревью кода.

Анализирует diff, закоммиченный код, предлагает исправления и рефакторинг.
Использует RAG + MCP Git + LLM (GPT-5-nano).

Использование:
    from code_reviewer import CodeReviewer

    reviewer = CodeReviewer(verbose=True)

    # Ревью diff
    report = reviewer.review_pr(base_branch="main")

    # Поиск проблем в закоммиченном коде
    report = reviewer.search_code("eval(", file_pattern="*.py")

    # Анализ файла с предложениями исправлений
    report = reviewer.analyze_file("core/config.py")

    # Рефакторинг файла
    report = reviewer.refactor("mcp_server.py", goal="разделить на модули")

    reviewer.close()
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from core.config import get_config, get_llm_client
from rag import SearchResult
from rag.search import search as rag_search

PROJECT_ROOT = Path(__file__).parent
DEFAULT_DEV_INDEX = PROJECT_ROOT / "rag_data_dev"
MCP_GIT_SERVER = PROJECT_ROOT / "mcp_git_server.py"

MAX_DIFF_CHARS = 8000  # лимит для gpt-5-nano


# ══════════════════════════════════════════════════════════════════════════════
# MCP Git-клиент (переиспользуем паттерн из dev_assistant.py)
# ══════════════════════════════════════════════════════════════════════════════

class _GitMCPClient:
    """Минимальный MCP-клиент для git-сервера."""

    def __init__(self, server_path: Path = MCP_GIT_SERVER):
        self._server_path = server_path
        self._proc: Optional[subprocess.Popen] = None
        self._msg_id = 0

    def _ensure_started(self):
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            [sys.executable, str(self._server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Handshake
        self._send_recv("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "code-reviewer", "version": "1.0"},
        })
        # Notification
        self._proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        self._proc.stdin.flush()

    def call_tool(self, name: str, arguments: dict | None = None) -> str:
        self._ensure_started()
        resp = self._send_recv("tools/call", {"name": name, "arguments": arguments or {}})
        result = resp.get("result", {})
        content = result.get("content", [{}])
        return content[0].get("text", "") if content else ""

    def close(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._proc = None

    def _send_recv(self, method: str, params: dict | None = None) -> dict:
        self._msg_id += 1
        msg = {"jsonrpc": "2.0", "id": self._msg_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        if not raw:
            raise RuntimeError("Git MCP server closed unexpectedly")
        return json.loads(raw)


# ══════════════════════════════════════════════════════════════════════════════
# CodeReviewer
# ══════════════════════════════════════════════════════════════════════════════

REVIEW_SYSTEM_PROMPT = """\
Ты — опытный код-ревьюер проекта TinyAI.
Твоя задача — проанализировать diff (изменения кода) и дать структурированное ревью.

Правила:
- Анализируй ТОЛЬКО показанный diff, не додумывай код
- Будь конкретным: указывай файлы и строки
- Если проблем не найдено — так и скажи, не выдумывай

Формат ответа (строго):

## 🐛 Потенциальные баги
<список или "Не обнаружено">

## 🏗️ Архитектурные проблемы
<список или "Не обнаружено">

## 💡 Рекомендации
<список улучшений или "Нет рекомендаций">

## 📊 Общая оценка
<краткий итог: всё ок / есть замечания / требует доработки>
"""

ANALYZE_SYSTEM_PROMPT = """\
Ты — опытный код-ревьюер проекта TinyAI.
Твоя задача — проанализировать файл (закоммиченный код) и найти проблемы.

Правила:
- Анализируй предоставленный код целиком
- Указывай конкретные строки и фрагменты
- Если код чистый — скажи об этом

Формат ответа (строго):

## 🐛 Баги и ошибки
<список с указанием строк или "Не обнаружено">

## ⚠️ Проблемы безопасности
<список или "Не обнаружено">

## 🏗️ Архитектурные замечания
<список или "Не обнаружено">

## 💡 Предложения по улучшению
<конкретные предложения с примерами кода>

## 📊 Оценка качества
<краткий итог>
"""

FIX_SYSTEM_PROMPT = """\
Ты — опытный разработчик проекта TinyAI.
Твоя задача — найти и исправить проблемы в коде.

Правила:
- Покажи конкретные исправления
- Для каждого исправления покажи ДО и ПОСЛЕ
- Объясни, почему нужно исправление

Формат ответа (строго):

## 🔍 Найденные проблемы
<список проблем>

## 🔧 Исправления
Для каждого исправления:
### Проблема N: <описание>
**Файл:** `<путь>`
**Было:**
```python
<старый код>
```
**Стало:**
```python
<исправленный код>
```
**Причина:** <объяснение>

## 📊 Итог
<сколько исправлений, критичность>
"""

REFACTOR_SYSTEM_PROMPT = """\
Ты — опытный разработчик проекта TinyAI.
Твоя задача — предложить рефакторинг кода.

Правила:
- Учитывай архитектуру проекта (из документации)
- Предлагай конкретные изменения с кодом
- Сохраняй обратную совместимость
- Не ломай существующие тесты

Формат ответа (строго):

## 🎯 Цель рефакторинга
<что улучшаем и зачем>

## 📋 План изменений
<пронумерованный список шагов>

## 🔧 Реализация
Для каждого шага:
### Шаг N: <описание>
**Файл:** `<путь>`
```python
<код после рефакторинга>
```

## ⚠️ Что может сломаться
<на что обратить внимание>

## 📊 Ожидаемый результат
<что улучшится>
"""


class CodeReviewer:
    """
    AI код-ревьюер: diff + RAG-контекст документации + LLM.

    Анализирует изменения кода и выдаёт структурированный отчёт.
    """

    def __init__(
        self,
        index_dir: Path = DEFAULT_DEV_INDEX,
        model: str = "gpt-5-nano",
        verbose: bool = False,
    ):
        self.index_dir = index_dir
        self.model = model
        self.verbose = verbose
        self._git_client: Optional[_GitMCPClient] = None

    def _get_git_client(self) -> _GitMCPClient:
        if self._git_client is None:
            self._git_client = _GitMCPClient()
        return self._git_client

    def _get_diff(self, base_branch: str = "", staged: bool = False) -> str:
        """Получает diff через MCP Git Server."""
        try:
            client = self._get_git_client()
            args = {}
            if base_branch:
                args["base_branch"] = base_branch
            elif staged:
                args["staged"] = True
            return client.call_tool("git_diff", args)
        except Exception as e:
            return f"Ошибка получения diff: {e}"

    def _get_changed_files(self) -> str:
        """Список изменённых файлов через MCP."""
        try:
            client = self._get_git_client()
            return client.call_tool("git_status")
        except Exception as e:
            return f"Ошибка получения статуса: {e}"

    def _get_branch(self) -> str:
        """Текущая ветка через MCP."""
        try:
            client = self._get_git_client()
            return client.call_tool("git_current_branch")
        except Exception as e:
            return "unknown"

    def _search_docs(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Поиск релевантной документации через RAG."""
        if not (self.index_dir / "chunks.db").exists():
            return []
        return rag_search(
            query=query,
            top_k=top_k,
            strategy="structure",
            index_dir=self.index_dir,
            similarity_threshold=0.2,
        )

    @staticmethod
    def _truncate_diff(diff: str, max_chars: int = MAX_DIFF_CHARS) -> str:
        """Обрезает diff если превышает лимит."""
        if len(diff) <= max_chars:
            return diff
        return diff[:max_chars] + f"\n\n... (обрезано, всего {len(diff)} символов)"

    def _extract_changed_filenames(self, diff: str) -> List[str]:
        """Извлекает имена файлов из diff."""
        files = []
        for line in diff.splitlines():
            if line.startswith("diff --git"):
                parts = line.split()
                if len(parts) >= 4:
                    # "diff --git a/file.py b/file.py" → "file.py"
                    files.append(parts[3].lstrip("b/"))
        return files

    def review_diff(self, diff_text: str, changed_files: str = "") -> str:
        """
        Ревью произвольного diff.

        Args:
            diff_text: Текст diff (unified format)
            changed_files: Список изменённых файлов (для контекста)

        Returns:
            Структурированный текст ревью
        """
        if not diff_text or diff_text.strip() in ("Нет изменений.", ""):
            return "✅ Нет изменений для ревью."

        # Извлекаем имена файлов для RAG-запроса
        filenames = self._extract_changed_filenames(diff_text)
        rag_query = " ".join(filenames[:5]) if filenames else "architecture code review"

        # RAG: ищем релевантную документацию
        results = self._search_docs(rag_query)
        doc_context_parts = []
        sources = []
        for r in results:
            src = r.chunk.metadata.get("source", "unknown")
            section = r.chunk.metadata.get("section", "")
            label = f"{src}"
            if section:
                label += f" / {section}"
            sources.append(label)
            doc_context_parts.append(
                f"--- Источник: {label} ---\n{r.chunk.text}"
            )
        doc_context = "\n\n".join(doc_context_parts) if doc_context_parts else ""

        # Обрезаем diff
        truncated_diff = self._truncate_diff(diff_text)

        # Формируем промпт
        user_parts = []
        if changed_files:
            user_parts.append(f"## Изменённые файлы\n{changed_files}")
        if doc_context:
            user_parts.append(f"## Контекст проекта (из документации)\n{doc_context}")
        user_parts.append(f"## Diff для ревью\n```diff\n{truncated_diff}\n```")

        user_prompt = "\n\n".join(user_parts)

        if self.verbose:
            print(f"  📂 Файлов в diff: {len(filenames)}")
            print(f"  📚 RAG-источников: {len(results)}")
            for s in sources:
                print(f"     {s}")
            print(f"  📝 Diff: {len(diff_text)} символов", end="")
            if len(diff_text) > MAX_DIFF_CHARS:
                print(f" (обрезано до {MAX_DIFF_CHARS})")
            else:
                print()

        # LLM вызов (GPT-5-nano через OpenAI)
        review = self._llm_call(REVIEW_SYSTEM_PROMPT, user_prompt)

        return review

    def review_pr(self, base_branch: str = "main") -> str:
        """
        Ревью PR: сравнивает текущую ветку с base_branch.

        Args:
            base_branch: Базовая ветка (по умолчанию "main")

        Returns:
            Текст ревью
        """
        branch = self._get_branch()
        diff = self._get_diff(base_branch=base_branch)
        changed_files = self._get_changed_files()

        if self.verbose:
            print(f"  🌿 {branch}")
            print(f"  🔀 Base: {base_branch}")

        header = f"# 🔍 AI Code Review\n\n**Ветка:** {branch}\n**Base:** {base_branch}\n\n"
        review = self.review_diff(diff, changed_files)
        return header + review

    def review_staged(self) -> str:
        """
        Ревью staged изменений (для локального использования перед коммитом).

        Returns:
            Текст ревью
        """
        branch = self._get_branch()
        diff = self._get_diff(staged=True)
        changed_files = self._get_changed_files()

        if self.verbose:
            print(f"  🌿 {branch}")
            print(f"  📋 Staged changes")

        header = f"# 🔍 AI Code Review (staged)\n\n**Ветка:** {branch}\n\n"
        review = self.review_diff(diff, changed_files)
        return header + review

    def review_working(self) -> str:
        """
        Ревью всех незакоммиченных изменений (staged + unstaged).

        Returns:
            Текст ревью
        """
        branch = self._get_branch()
        diff = self._get_diff()
        changed_files = self._get_changed_files()

        if self.verbose:
            print(f"  🌿 {branch}")
            print(f"  📋 Working directory changes")

        header = f"# 🔍 AI Code Review\n\n**Ветка:** {branch}\n\n"
        review = self.review_diff(diff, changed_files)
        return header + review

    def close(self):
        """Закрывает MCP-соединение."""
        if self._git_client:
            self._git_client.close()
            self._git_client = None

    # ── Поиск по коду ────────────────────────────────────────────────────────

    def _grep_code(self, pattern: str, file_pattern: str = "", ignore_case: bool = True) -> str:
        """Поиск паттерна в закоммиченном коде через MCP git grep."""
        try:
            client = self._get_git_client()
            args = {"pattern": pattern, "ignore_case": ignore_case}
            if file_pattern:
                args["file_pattern"] = file_pattern
            return client.call_tool("git_grep", args)
        except Exception as e:
            return f"Ошибка поиска: {e}"

    def _get_file(self, path: str) -> str:
        """Получает содержимое файла из HEAD через MCP."""
        try:
            client = self._get_git_client()
            return client.call_tool("git_show_file", {"path": path})
        except Exception as e:
            return f"Ошибка чтения файла: {e}"

    def _llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """Общий вызов LLM с fallback для reasoning-моделей."""
        client = get_llm_client("openai")
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=16384,
            )
            result = response.choices[0].message.content or ""
            if not result and hasattr(response.choices[0].message, "reasoning"):
                result = response.choices[0].message.reasoning or ""
            return result
        except Exception as e:
            return f"Ошибка LLM: {e}"

    def search_code(self, pattern: str, file_pattern: str = "*.py", ignore_case: bool = True) -> str:
        """
        Поиск паттерна в закоммиченном коде + AI-анализ найденного.

        Args:
            pattern: Строка или регулярное выражение для поиска
            file_pattern: Фильтр по файлам (по умолчанию "*.py")
            ignore_case: Игнорировать регистр

        Returns:
            Результаты поиска + AI-анализ
        """
        grep_result = self._grep_code(pattern, file_pattern, ignore_case)

        if self.verbose:
            print(f"  🔎 Поиск: '{pattern}' в {file_pattern}")
            lines = grep_result.strip().splitlines()
            print(f"  📋 Результат: {len(lines)} строк")

        if "Ничего не найдено" in grep_result:
            return f"🔎 Поиск: `{pattern}` в `{file_pattern}`\n\n{grep_result}"

        # RAG-контекст
        results = self._search_docs(pattern)
        doc_context = ""
        if results:
            parts = []
            for r in results:
                src = r.chunk.metadata.get("source", "unknown")
                parts.append(f"--- {src} ---\n{r.chunk.text}")
            doc_context = "\n\n".join(parts)

        user_prompt = f"## Результаты поиска `{pattern}`\n```\n{self._truncate_diff(grep_result)}\n```"
        if doc_context:
            user_prompt += f"\n\n## Контекст проекта\n{doc_context}"
        user_prompt += "\n\nПроанализируй найденные совпадения. Есть ли среди них проблемы, баги или места для улучшения?"

        analysis = self._llm_call(ANALYZE_SYSTEM_PROMPT, user_prompt)

        header = f"# 🔎 Поиск по коду: `{pattern}`\n\n**Файлы:** `{file_pattern}`\n\n"
        return header + analysis

    def analyze_file(self, file_path: str) -> str:
        """
        Анализ закоммиченного файла: баги, безопасность, качество.

        Args:
            file_path: Путь к файлу относительно корня проекта

        Returns:
            Структурированный отчёт об анализе
        """
        code = self._get_file(file_path)

        if code.startswith("Ошибка"):
            return f"❌ {code}"

        if self.verbose:
            lines = code.strip().splitlines()
            print(f"  📄 Файл: {file_path} ({len(lines)} строк)")

        # RAG-контекст
        results = self._search_docs(file_path)
        doc_context = ""
        if results:
            parts = []
            for r in results:
                src = r.chunk.metadata.get("source", "unknown")
                parts.append(f"--- {src} ---\n{r.chunk.text}")
            doc_context = "\n\n".join(parts)

        user_parts = [f"## Файл: `{file_path}`\n```python\n{self._truncate_diff(code, 10000)}\n```"]
        if doc_context:
            user_parts.append(f"## Документация проекта\n{doc_context}")
        user_parts.append("Проанализируй этот файл. Найди баги, проблемы безопасности и предложи улучшения.")

        analysis = self._llm_call(ANALYZE_SYSTEM_PROMPT, "\n\n".join(user_parts))

        header = f"# 📄 Анализ файла: `{file_path}`\n\n"
        return header + analysis

    def fix_code(self, file_path: str, issue: str = "") -> str:
        """
        Находит и предлагает конкретные исправления для файла.

        Args:
            file_path: Путь к файлу
            issue: Описание проблемы (опционально — если не указано, ищет все)

        Returns:
            Конкретные исправления в формате ДО/ПОСЛЕ
        """
        code = self._get_file(file_path)

        if code.startswith("Ошибка"):
            return f"❌ {code}"

        if self.verbose:
            lines = code.strip().splitlines()
            print(f"  📄 Файл: {file_path} ({len(lines)} строк)")
            if issue:
                print(f"  🎯 Проблема: {issue}")

        # RAG-контекст
        results = self._search_docs(file_path)
        doc_context = ""
        if results:
            parts = []
            for r in results:
                src = r.chunk.metadata.get("source", "unknown")
                parts.append(f"--- {src} ---\n{r.chunk.text}")
            doc_context = "\n\n".join(parts)

        user_parts = [f"## Файл: `{file_path}`\n```python\n{self._truncate_diff(code, 10000)}\n```"]
        if doc_context:
            user_parts.append(f"## Документация проекта\n{doc_context}")
        if issue:
            user_parts.append(f"## Конкретная проблема\n{issue}")
        user_parts.append("Найди проблемы и предложи конкретные исправления в формате ДО/ПОСЛЕ.")

        fixes = self._llm_call(FIX_SYSTEM_PROMPT, "\n\n".join(user_parts))

        header = f"# 🔧 Исправления: `{file_path}`\n\n"
        if issue:
            header += f"**Проблема:** {issue}\n\n"
        return header + fixes

    def refactor(self, file_path: str, goal: str = "") -> str:
        """
        Предлагает рефакторинг файла с конкретным кодом.

        Args:
            file_path: Путь к файлу
            goal: Цель рефакторинга (например "разделить на модули", "улучшить читаемость")

        Returns:
            План рефакторинга с конкретным кодом
        """
        code = self._get_file(file_path)

        if code.startswith("Ошибка"):
            return f"❌ {code}"

        if self.verbose:
            lines = code.strip().splitlines()
            print(f"  📄 Файл: {file_path} ({len(lines)} строк)")
            if goal:
                print(f"  🎯 Цель: {goal}")

        # RAG-контекст
        results = self._search_docs(file_path + " architecture")
        doc_context = ""
        if results:
            parts = []
            for r in results:
                src = r.chunk.metadata.get("source", "unknown")
                parts.append(f"--- {src} ---\n{r.chunk.text}")
            doc_context = "\n\n".join(parts)

        user_parts = [f"## Файл: `{file_path}`\n```python\n{self._truncate_diff(code, 10000)}\n```"]
        if doc_context:
            user_parts.append(f"## Архитектура проекта (из документации)\n{doc_context}")
        if goal:
            user_parts.append(f"## Цель рефакторинга\n{goal}")
        else:
            user_parts.append("Предложи рефакторинг для улучшения качества и читаемости кода.")

        refactoring = self._llm_call(REFACTOR_SYSTEM_PROMPT, "\n\n".join(user_parts))

        header = f"# ♻️ Рефакторинг: `{file_path}`\n\n"
        if goal:
            header += f"**Цель:** {goal}\n\n"
        return header + refactoring
