"""
AgentLoop — ReAct-агент для автономной генерации новостных постов.

Паттерн ReAct: Reason → Act → Observe → Repeat until done.
Агент сам выбирает инструменты, управляет порядком шагов
и завершает работу при наличии финального поста.
"""

import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from news_agent.roles import get_role
from news_agent.tools import TOOL_DEFINITIONS, ToolDispatcher


MAX_ITERATIONS = 12
FINAL_ANSWER_MARKER = "FINAL_POST:"


class AgentLoop:
    """
    Автономный ReAct-агент.
    Запускается через метод run(task) и самостоятельно оркестрирует
    вызовы инструментов до получения готового поста.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "zai-org/GLM-4.7-Flash",
        storage=None,
        verbose: bool = True,
        max_iterations: int = MAX_ITERATIONS,
    ):
        self.client = client
        self.model = model
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.dispatcher = ToolDispatcher(storage=storage)
        self._token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def run(self, task: str) -> Dict[str, Any]:
        """
        Запускает агентский цикл.
        Возвращает dict с финальным постом и метаданными.
        """
        role = get_role("autonomous")
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": role.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Задача: создай новостной пост на тему:\n\n{task}\n\n"
                    f"Используй доступные инструменты. "
                    f"Когда пост готов — сохрани его через save_post "
                    f"и в финальном ответе начни с '{FINAL_ANSWER_MARKER}'"
                ),
            },
        ]

        self._print_agent_header(task)

        iteration = 0
        final_post = None
        saved_post_id = None

        while iteration < self.max_iterations:
            iteration += 1
            self._print_iteration(iteration)

            response = self._call_llm(messages)
            assistant_message = response.choices[0].message

            self._update_token_usage(response)

            messages.append({"role": "assistant", "content": assistant_message.content or ""})

            if assistant_message.content and FINAL_ANSWER_MARKER in assistant_message.content:
                final_post = assistant_message.content.split(FINAL_ANSWER_MARKER, 1)[1].strip()
                if self.verbose:
                    print(f"\n✅ Агент завершил работу на итерации {iteration}")
                break

            tool_calls = getattr(assistant_message, "tool_calls", None)
            if not tool_calls:
                if self.verbose:
                    if assistant_message.content:
                        print(f"\n💬 Агент: {assistant_message.content[:200]}...")
                    print("⚠️  Нет вызовов инструментов — продолжаем...")
                messages.append({
                    "role": "user",
                    "content": (
                        "Продолжай. Используй инструменты для создания поста. "
                        f"Когда пост будет готов и сохранён — начни ответ с '{FINAL_ANSWER_MARKER}'"
                    ),
                })
                continue

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if self.verbose:
                    self._print_tool_call(tool_name, args)

                observation = self.dispatcher.dispatch(tool_name, args)

                if tool_name == "save_post":
                    try:
                        obs_data = json.loads(observation)
                        saved_post_id = obs_data.get("post_id")
                    except Exception:
                        pass

                if self.verbose:
                    obs_preview = observation[:150].replace("\n", " ")
                    print(f"  📊 Результат: {obs_preview}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": observation,
                })

        else:
            if self.verbose:
                print(f"\n⚠️  Достигнут лимит итераций ({self.max_iterations})")

        if final_post is None and saved_post_id:
            final_post = f"Пост сохранён с ID: {saved_post_id}"

        return {
            "final_post": final_post or "Агент не создал финальный пост",
            "saved_post_id": saved_post_id,
            "iterations": iteration,
            "token_usage": self._token_usage.copy(),
        }

    def _call_llm(self, messages: List[Dict[str, Any]]):
        """Вызывает LLM с инструментами (function calling)."""
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "tool_choice": "auto",
            "max_completion_tokens": 4000,
            "temperature": 0.5,
        }

        if "gpt-5-nano" in self.model:
            del params["temperature"]

        for attempt in range(3):
            try:
                return self.client.chat.completions.create(**params)
            except Exception as e:
                if attempt < 2:
                    if self.verbose:
                        print(f"  ⚠️  API ошибка: {e}. Повтор через 2с...")
                    time.sleep(2)
                else:
                    raise

    def _update_token_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            self._token_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            self._token_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0)
            self._token_usage["total_tokens"] += getattr(usage, "total_tokens", 0)
        else:
            content = response.choices[0].message.content or ""
            self._token_usage["total_tokens"] += len(content) // 4

    def _print_agent_header(self, task: str) -> None:
        if not self.verbose:
            return
        print("\n" + "═" * 65)
        print("  AUTONOMOUS NEWS AGENT  (ReAct)")
        print("═" * 65)
        print(f"  Задача:  {task[:80]}")
        print(f"  Модель:  {self.model}")
        print(f"  Лимит:   {self.max_iterations} итераций")
        print("═" * 65)

    def _print_iteration(self, n: int) -> None:
        if self.verbose:
            print(f"\n--- Итерация {n} ---", flush=True)

    def _print_tool_call(self, name: str, args: Dict[str, Any]) -> None:
        if not self.verbose:
            return
        args_preview = json.dumps(args, ensure_ascii=False)[:120]
        print(f"  🔧 Вызов инструмента: {name}({args_preview}...)")
