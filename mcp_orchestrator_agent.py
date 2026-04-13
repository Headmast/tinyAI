"""
MCPOrchestratorAgent — оркестрационный агент с динамическим доступом
ко всем MCP-серверам (День 20: Orchestration MCP).

В отличие от MCPAgent (1 сервер) и MCPSchedulerAgent (2 сервера),
этот агент:
  - Подключается к произвольному числу MCP-серверов через MCPRegistry
  - Автоматически обнаруживает все инструменты (tools/list)
  - Маршрутизирует tool_calls через MCPRouter
  - Поддерживает длинные мульти-серверные флоу (до 15 итераций)
  - Логирует каждый вызов: инструмент, сервер, время

Использование:
    registry = MCPRegistry()
    registry.register("logs", "mcp_server.py")
    registry.register("scheduler", "mcp_scheduler_server.py")
    registry.register("search", "pipeline/servers/search_server.py")
    # ...
    registry.discover_tools()

    agent = MCPOrchestratorAgent(client, registry)
    answer = agent.chat("Найди последние логи и создай задачу в планировщике")
    agent.close()
"""

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from mcp_registry import MCPRegistry
from mcp_router import MCPRouter

MAX_TOOL_ITERATIONS = 15

SYSTEM_PROMPT_TEMPLATE = """Ты — оркестрационный агент TinyAI с доступом к нескольким MCP-серверам.

У тебя есть инструменты с {server_count} серверов:

{tools_description}

Правила:
- Используй инструменты для получения реальных данных, не придумывай
- Выбирай наиболее подходящий инструмент для каждого шага задачи
- Для сложных задач — разбивай на последовательность вызовов инструментов
- Результат одного инструмента можно использовать как вход для другого
- Отвечай кратко и структурированно, ссылаясь на полученные данные
- При ошибке инструмента — объясни пользователю и предложи альтернативу"""


def _build_tools_description(registry: MCPRegistry) -> str:
    """Формирует текстовое описание инструментов для system prompt."""
    lines = []
    for server in registry.list_servers():
        name = server["name"]
        tags = ", ".join(server["tags"]) if server["tags"] else "—"
        tools_list = server["tools"]
        lines.append(f"Сервер «{name}» (теги: {tags}):")
        for tool_name in tools_list:
            lines.append(f"  - {tool_name}")
    return "\n".join(lines)


class MCPOrchestratorAgent:
    """
    Оркестрационный агент с динамическим обнаружением и маршрутизацией
    инструментов через несколько MCP-серверов.
    """

    def __init__(
        self,
        client: OpenAI,
        registry: MCPRegistry,
        model: str = "zai-org/GLM-4.7",
        verbose: bool = True,
        max_completion_tokens: int = 4000,
    ):
        self.client = client
        self.model = model
        self.verbose = verbose
        self.max_completion_tokens = max_completion_tokens

        self._registry = registry
        self._router = MCPRouter(registry, verbose=verbose)

        # Получаем определения инструментов из реестра
        self._tool_definitions = registry.get_tool_definitions()

        # Формируем system prompt
        tools_desc = _build_tools_description(registry)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            server_count=len(registry.list_servers()),
            tools_description=tools_desc,
        )

        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        self._total_tokens: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._tool_calls_count = 0

    def chat(self, user_input: str) -> str:
        """
        Обрабатывает запрос пользователя.
        Автоматически вызывает инструменты с нескольких серверов.
        Возвращает финальный текстовый ответ.
        """
        self._messages.append({"role": "user", "content": user_input})

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages,
                tools=self._tool_definitions,
                max_tokens=self.max_completion_tokens,
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

            # Выполняем каждый tool_call через router
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                self._tool_calls_count += 1
                server_name = self._registry.get_server_for_tool(tool_name) or "?"

                if self.verbose:
                    args_preview = json.dumps(args, ensure_ascii=False)[:80]
                    print(
                        f"  🔧 MCP #{self._tool_calls_count}: "
                        f"{tool_name}({args_preview}) → [{server_name}]"
                    )

                result = self._router.call_tool(tool_name, args)

                if self.verbose:
                    print(f"  📊 → {result[:140].replace(chr(10), ' ')}")

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return "Превышен лимит итераций инструментов."

    def close(self):
        """Закрывает все соединения через реестр."""
        self._registry.close()

    @property
    def token_usage(self) -> Dict[str, int]:
        return self._total_tokens.copy()

    @property
    def tool_calls_total(self) -> int:
        return self._tool_calls_count

    @property
    def routing_summary(self) -> Dict[str, Any]:
        """Сводка маршрутизации: какие серверы и инструменты были использованы."""
        return self._router.get_summary()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
