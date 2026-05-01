"""
FileAssistant — AI-ассистент для автономной работы с файлами проекта.

Использует MCP Files Server (mcp_files_server.py) для чтения, поиска,
анализа и модификации файлов. Агент самостоятельно решает, какие файлы
открыть и какие действия выполнить для достижения цели.

Использование:
    from file_assistant import FileAssistant
    from core.config import get_llm_client

    client = get_llm_client()
    assistant = FileAssistant(client)
    result = assistant.run("Найди все использования MCPRegistry в проекте")
    print(result)
    assistant.close()
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from core.config import get_config, get_llm_client
from mcp_stdio_client import MCPStdioClient

PROJECT_ROOT = Path(__file__).parent
MCP_FILES_SERVER = PROJECT_ROOT / "mcp_files_server.py"

# ── Определения инструментов для function calling ────────────────────────────

FILE_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_project_files",
            "description": (
                "Возвращает список файлов и каталогов по указанному пути проекта. "
                "Пропускает .git, __pycache__, node_modules. "
                "Показывает имя, тип (file/dir), размер."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Относительный путь от корня проекта (по умолчанию '.')",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob-паттерн для фильтрации (например '*.py')",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Читает содержимое файла проекта. "
                "Возвращает текст файла и количество строк."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Относительный путь к файлу от корня проекта",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Максимальное количество строк для чтения",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": (
                "Ищет текст по файлам проекта (регистро-независимо). "
                "Возвращает файл, номер строки и совпавшую строку."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Текст для поиска (без учёта регистра)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Каталог для поиска (по умолчанию '.' — весь проект)",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Glob-паттерн файлов (например '*.py')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Максимум результатов (по умолчанию 30)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_info",
            "description": (
                "Возвращает метаданные файла: размер, количество строк, "
                "дату изменения, является ли бинарным."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Относительный путь к файлу от корня проекта",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Создаёт или перезаписывает файл. "
                "Создаёт родительские каталоги при необходимости. "
                "Запрещено для .git/, .env."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Относительный путь к файлу от корня проекта",
                    },
                    "content": {
                        "type": "string",
                        "description": "Содержимое для записи",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_diff",
            "description": (
                "Заменяет текст в файле (str_replace). "
                "old_text должен встречаться в файле ровно 1 раз."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Относительный путь к файлу",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Текст для замены (должен встречаться ровно 1 раз)",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Новый текст для вставки вместо old_text",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]

# ── Системный промпт ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ты — AI-ассистент для автономной работы с файлами проекта TinyAI.

У тебя есть инструменты для:
1. ПРОСМОТРА — list_project_files, read_file, get_file_info
2. ПОИСКА — search_in_files (grep-like поиск по всем файлам)
3. СОЗДАНИЯ — write_file (новые файлы)
4. МОДИФИКАЦИИ — apply_diff (точечная замена текста в существующих файлах)

Правила работы:
- ПЛАНИРУЙ перед действием: сначала определи какие файлы нужны, потом читай их
- Используй search_in_files для поиска паттернов по кодовой базе
- При создании файлов используй write_file — он сохраняет файл на диск
- При модификации используй apply_diff — он заменяет текст в файле
- Не придумывай содержимое файлов — всегда читай реальные данные через инструменты
- После записи/изменения файла можешь прочитать его для проверки
- Отвечай кратко и по делу, ссылаясь на реальные файлы и данные
- Если задача требует анализа нескольких файлов — обрабатывай их последовательно

Формат ответа:
- Перечисли какие файлы были прочитаны/изменены/созданы
- Дай краткий итог выполненной работы
- При анализе — конкретные выводы с цитатами из кода"""

MAX_TOOL_ITERATIONS = 15


class FileAssistant:
    """
    AI-ассистент для автономной работы с файлами проекта.

    Использует агентный цикл: LLM решает какие инструменты вызвать,
    вызывает их через MCP Files Server, получает результаты и решает
    что делать дальше. До 15 итераций.
    """

    def __init__(
        self,
        client=None,
        model: str | None = None,
        verbose: bool = True,
        max_completion_tokens: int = 4096,
    ):
        config = get_config()
        self.client = client or get_llm_client()
        self.model = model or config.default_model
        self.verbose = verbose
        self.max_completion_tokens = max_completion_tokens
        self._mcp: Optional[MCPStdioClient] = None
        self._total_tokens: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._tool_calls_count = 0

    def _get_mcp(self) -> MCPStdioClient:
        """Lazy-инициализация MCP-клиента."""
        if self._mcp is None:
            self._mcp = MCPStdioClient(
                server_path=MCP_FILES_SERVER,
                client_name="file-assistant",
            )
        return self._mcp

    def run(self, task: str) -> str:
        """
        Выполняет задачу по работе с файлами.

        Агентный цикл: LLM решает какие инструменты вызвать → вызывает
        через MCP → получает результат → повторяет до завершения задачи.

        Возвращает финальный текстовый ответ.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📋 Задача: {task}")
            print(f"{'='*60}")

        for iteration in range(MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=FILE_TOOL_DEFINITIONS,
                max_completion_tokens=self.max_completion_tokens,
                temperature=0.3,
            )

            # Обновляем счётчики токенов
            if response.usage:
                self._total_tokens["prompt_tokens"] += response.usage.prompt_tokens
                self._total_tokens["completion_tokens"] += response.usage.completion_tokens
                self._total_tokens["total_tokens"] += response.usage.total_tokens

            choice = response.choices[0]
            assistant_msg = choice.message

            # Если нет tool_calls — финальный ответ
            tool_calls = getattr(assistant_msg, "tool_calls", None)
            if not tool_calls:
                text = assistant_msg.content or ""
                # Fallback для reasoning-моделей (GLM quirk)
                if not text and hasattr(assistant_msg, "reasoning"):
                    text = assistant_msg.reasoning or ""
                if self.verbose:
                    print(f"\n✅ Готово (итераций: {iteration + 1}, "
                          f"вызовов инструментов: {self._tool_calls_count})")
                return text

            # Есть tool_calls — вызываем инструменты
            messages.append({
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                self._tool_calls_count += 1
                if self.verbose:
                    args_preview = json.dumps(args, ensure_ascii=False)[:100]
                    print(f"  🔧 [{iteration+1}] {tool_name}({args_preview})")

                # Вызов через MCP
                mcp = self._get_mcp()
                result = mcp.call_tool(tool_name, args)

                if self.verbose:
                    preview = result[:150].replace("\n", " ")
                    print(f"     → {preview}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return "Превышен лимит итераций инструментов (15)."

    def close(self):
        """Закрывает MCP-соединение."""
        if self._mcp:
            self._mcp.close()
            self._mcp = None

    @property
    def token_usage(self) -> Dict[str, int]:
        return self._total_tokens.copy()

    @property
    def tool_calls_total(self) -> int:
        return self._tool_calls_count

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
# CLI для интерактивного использования
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Интерактивный режим: задаёте задачу — ассистент выполняет."""
    import argparse

    parser = argparse.ArgumentParser(description="File Assistant — AI для работы с файлами проекта")
    parser.add_argument("task", nargs="?", help="Задача для ассистента (или интерактивный режим)")
    parser.add_argument("--model", default=None, help="Модель LLM")
    parser.add_argument("--quiet", action="store_true", help="Минимальный вывод")
    args = parser.parse_args()

    with FileAssistant(model=args.model, verbose=not args.quiet) as assistant:
        if args.task:
            result = assistant.run(args.task)
            print(f"\n{'─'*60}")
            print(result)
        else:
            print("🗂️  File Assistant — AI для работы с файлами проекта")
            print("   Введите задачу или 'exit' для выхода.\n")
            while True:
                try:
                    task = input("📝 Задача: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not task or task.lower() in ("exit", "quit", "q"):
                    break
                result = assistant.run(task)
                print(f"\n{'─'*60}")
                print(result)
                print()


if __name__ == "__main__":
    main()
