"""
MCPSchedulerAgent — агент с инструментами планировщика (День 18).

Объединяет два MCP-моста:
  1. MCPBridge         → mcp_server.py         (логи, память, статистика)
  2. MCPSchedulerBridge → mcp_scheduler_server.py (планировщик, сводки, резервные копии)

Агент умеет:
  - Просматривать и настраивать задачи по расписанию
  - Запускать задачи немедленно
  - Получать сводки активности чатов
  - Читать историю снимков и резервных копий
  - Работать с логами и памятью (через MCPBridge)
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from news_agent.mcp_bridge import MCPBridge, MCP_TOOLS
from news_agent.mcp_scheduler_bridge import MCPSchedulerBridge, SCHEDULER_MCP_TOOLS

load_dotenv()

# ── Определения инструментов для function calling ────────────────────────────

SCHEDULER_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "scheduler_add_task",
            "description": (
                "Добавляет периодическую задачу в планировщик. "
                "Типы: chat_collector (сбор истории чатов), "
                "chat_backup (резервное копирование), "
                "summary_generator (генерация сводок), "
                "reminder (напоминание с текстом)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Уникальное имя задачи",
                    },
                    "task_type": {
                        "type": "string",
                        "enum": [
                            "chat_collector",
                            "chat_backup",
                            "summary_generator",
                            "reminder",
                        ],
                    },
                    "interval_seconds": {
                        "type": "integer",
                        "description": "Интервал в секундах (минимум 60)",
                    },
                    "payload": {
                        "type": "string",
                        "description": "JSON-строка с параметрами (для reminder: '{\"message\": \"текст\"}')",
                    },
                },
                "required": ["name", "task_type", "interval_seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scheduler_list_tasks",
            "description": "Список всех задач планировщика с расписанием и статусом.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scheduler_remove_task",
            "description": "Деактивирует задачу планировщика по имени.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Имя задачи"}
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scheduler_get_summary",
            "description": "Возвращает последнюю агрегированную сводку активности чатов.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scheduler_list_snapshots",
            "description": "Снимки истории чата, собранные планировщиком.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное число записей (по умолчанию 20)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scheduler_list_backups",
            "description": "Список всех резервных копий чат-логов.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scheduler_trigger_task",
            "description": "Немедленно запускает задачу планировщика по имени.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Имя задачи"}
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scheduler_get_task_runs",
            "description": "История последних выполнений задачи.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Имя задачи"},
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное число записей",
                    },
                },
                "required": ["name"],
            },
        },
    },
    # ── Инструменты логов/памяти (из Дня 17) ─────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "list_logs",
            "description": "Список всех файлов логов разговоров с метаданными.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Фильтр по имени файла (подстрока)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_usage_stats",
            "description": "Статистика использования: токены, стоимость, число разговоров.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_conversation_summary",
            "description": "Краткая сводка конкретного файла лога разговора.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Имя файла лога",
                    }
                },
                "required": ["filename"],
            },
        },
    },
]

ALL_TOOL_DEFINITIONS = SCHEDULER_TOOL_DEFINITIONS


SYSTEM_PROMPT = """Ты — интеллектуальный ассистент-планировщик TinyAI (День 18).

У тебя есть MCP-инструменты для:
1. ПЛАНИРОВЩИКА ЗАДАЧ:
   - scheduler_add_task       — добавить задачу (chat_collector, chat_backup, summary_generator, reminder)
   - scheduler_list_tasks     — просмотр всех задач
   - scheduler_remove_task    — удалить задачу
   - scheduler_trigger_task   — запустить задачу немедленно
   - scheduler_get_task_runs  — история выполнений
2. ДАННЫХ И СВОДОК:
   - scheduler_get_summary    — последняя сводка активности чатов
   - scheduler_list_snapshots — снимки истории чатов
   - scheduler_list_backups   — список резервных копий
3. ЛОГОВ И СТАТИСТИКИ:
   - list_logs, get_usage_stats, get_conversation_summary

Правила:
- Используй реальные данные из инструментов, не придумывай
- Перед тем как добавить задачу — проверь, нет ли её уже (scheduler_list_tasks)
- Давай конкретные рекомендации по интервалам (300с=5мин, 3600с=1ч, 86400с=1день)
- Отвечай кратко и структурированно"""

MAX_TOOL_ITERATIONS = 10


class MCPSchedulerAgent:
    """
    Агент с доступом к планировщику задач через function calling.
    Поддерживает два MCP-моста: логи/память и планировщик.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "zai-org/GLM-4.7",
        verbose: bool = True,
        max_completion_tokens: int = 4000,
    ) -> None:
        self.client = client
        self.model = model
        self.verbose = verbose
        self.max_completion_tokens = max_completion_tokens

        self._logs_bridge = MCPBridge(verbose=False)
        self._sched_bridge = MCPSchedulerBridge(verbose=False)
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
        """Обрабатывает запрос, выполняет MCP-вызовы, возвращает ответ."""
        self._messages.append({"role": "user", "content": user_input})

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages,
                tools=ALL_TOOL_DEFINITIONS,
                max_completion_tokens=self.max_completion_tokens,
                temperature=0.7,
            )

            if response.usage:
                self._total_tokens["prompt_tokens"] += response.usage.prompt_tokens
                self._total_tokens["completion_tokens"] += response.usage.completion_tokens
                self._total_tokens["total_tokens"] += response.usage.total_tokens

            choice = response.choices[0]
            msg = choice.message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                text = msg.content or ""
                self._messages.append({"role": "assistant", "content": text})
                return text

            # Добавляем сообщение ассистента с tool_calls
            self._messages.append({
                "role": "assistant",
                "content": msg.content or "",
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

            # Выполняем каждый tool_call
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                self._tool_calls_count += 1
                if self.verbose:
                    args_preview = json.dumps(args, ensure_ascii=False)[:80]
                    print(
                        f"  🔧 MCP #{self._tool_calls_count}: "
                        f"{tool_name}({args_preview})"
                    )

                # Маршрутизируем к нужному бриджу
                if tool_name in SCHEDULER_MCP_TOOLS:
                    result = self._sched_bridge.call_tool(tool_name, args)
                else:
                    result = self._logs_bridge.call_tool(tool_name, args)

                if self.verbose:
                    print(f"  📊 → {result[:140].replace(chr(10), ' ')}")

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return "Превышен лимит итераций инструментов."

    def close(self) -> None:
        self._logs_bridge.close()
        self._sched_bridge.close()

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
