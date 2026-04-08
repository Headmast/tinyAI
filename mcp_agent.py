"""
MCPAgent — диалоговый агент с доступом к MCP-инструментам.

День 17: Первый инструмент MCP.

Агент использует function calling для вызова MCP-инструментов:
  - list_logs, read_log, search_logs — работа с историей разговоров
  - list_memory, read_memory, save_memory, delete_memory_key — управление памятью
  - get_usage_stats — статистика использования
  - get_conversation_summary — краткая сводка разговора

Сценарий:
  1. Агент получает задачу от пользователя
  2. Решает, какие MCP-инструменты вызвать
  3. Отправляет tool_call через function calling
  4. Получает результат и формирует ответ пользователю
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from news_agent.mcp_bridge import MCPBridge, MCP_TOOLS

load_dotenv()

# ── Определения MCP-инструментов для function calling ────────────────────────

MCP_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_logs",
            "description": (
                "Возвращает список всех файлов логов разговоров с метаданными "
                "(имя файла, размер, дата изменения)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Необязательный фильтр по имени файла (подстрока)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_log",
            "description": (
                "Читает содержимое конкретного файла лога разговора. "
                "Возвращает сообщения с временными метками и статистикой токенов."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Имя файла лога (например conversation_20260317_192513.json)",
                    },
                    "last_n": {
                        "type": "integer",
                        "description": "Вернуть только последние N сообщений",
                    },
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": (
                "Ищет заданный текст во всех логах разговоров. "
                "Возвращает совпадения с контекстом."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Текст для поиска (без учёта регистра)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Максимальное число результатов (по умолчанию 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_memory",
            "description": "Возвращает список файлов долгосрочной памяти агента с кратким содержимым.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "Читает конкретный файл долгосрочной памяти агента.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Имя файла памяти (например long_term.json)",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_usage_stats",
            "description": (
                "Возвращает накопленную статистику использования: "
                "токены, стоимость, число разговоров."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Записывает данные в файл долгосрочной памяти агента. "
                "Создаёт файл если не существует, обновляет ключи если существует."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Имя файла памяти (например agent_notes.json)",
                    },
                    "key": {
                        "type": "string",
                        "description": "Ключ для записи",
                    },
                    "value": {
                        "type": "string",
                        "description": "Значение для записи (строка или JSON-строка)",
                    },
                },
                "required": ["filename", "key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_memory_key",
            "description": "Удаляет конкретный ключ из файла долгосрочной памяти.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Имя файла памяти",
                    },
                    "key": {
                        "type": "string",
                        "description": "Ключ для удаления",
                    },
                },
                "required": ["filename", "key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_conversation_summary",
            "description": (
                "Возвращает краткую сводку разговора: число сообщений, период, "
                "общее число токенов, первые и последние темы."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Имя файла лога разговора",
                    }
                },
                "required": ["filename"],
            },
        },
    },
]

SYSTEM_PROMPT = """Ты — интеллектуальный ассистент с доступом к системе управления историей и памятью.

У тебя есть MCP-инструменты для работы с:
1. ЛОГАМИ РАЗГОВОРОВ — list_logs, read_log, search_logs, get_conversation_summary
2. ПАМЯТЬЮ АГЕНТА — list_memory, read_memory, save_memory, delete_memory_key
3. СТАТИСТИКОЙ — get_usage_stats

Правила:
- Используй инструменты для получения реальных данных, не придумывай
- При вопросах об истории — сначала ищи в логах и памяти
- Если пользователь просит запомнить что-то — используй save_memory
- Отвечай кратко и по существу, ссылаясь на найденные данные
- При анализе нескольких файлов — используй инструменты последовательно"""


MAX_TOOL_ITERATIONS = 8


class MCPAgent:
    """
    Диалоговый агент с MCP-инструментами через function calling.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "zai-org/GLM-4.7",
        verbose: bool = True,
        max_completion_tokens: int = 4000,
    ):
        self.client = client
        self.model = model
        self.verbose = verbose
        self.max_completion_tokens = max_completion_tokens
        self._mcp = MCPBridge(verbose=verbose)
        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self._total_tokens: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._tool_calls_count = 0

    def chat(self, user_input: str) -> str:
        """
        Обрабатывает одно сообщение пользователя.
        Автоматически вызывает MCP-инструменты при необходимости.
        Возвращает финальный текстовый ответ.
        """
        self._messages.append({"role": "user", "content": user_input})

        for iteration in range(MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages,
                tools=MCP_TOOL_DEFINITIONS,
                max_completion_tokens=self.max_completion_tokens,
                temperature=0.7,
            )

            # Обновляем счётчики
            if response.usage:
                self._total_tokens["prompt_tokens"] += response.usage.prompt_tokens
                self._total_tokens["completion_tokens"] += response.usage.completion_tokens
                self._total_tokens["total_tokens"] += response.usage.total_tokens

            choice = response.choices[0]
            assistant_msg = choice.message

            # Если нет tool_calls — это финальный ответ
            tool_calls = getattr(assistant_msg, "tool_calls", None)
            if not tool_calls:
                text = assistant_msg.content or ""
                self._messages.append({"role": "assistant", "content": text})
                return text

            # Есть tool_calls — вызываем MCP-инструменты
            # Добавляем сообщение ассистента с tool_calls
            self._messages.append({
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
                    print(f"  🔧 MCP-вызов #{self._tool_calls_count}: {tool_name}({json.dumps(args, ensure_ascii=False)[:80]})")

                # Вызываем через MCP bridge
                result = self._mcp.call_tool(tool_name, args)

                if self.verbose:
                    preview = result[:120].replace("\n", " ")
                    print(f"  📊 Результат: {preview}...")

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # Лимит итераций
        return "Превышен лимит итераций инструментов."

    def close(self):
        """Закрывает MCP-соединение."""
        self._mcp.close()

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
