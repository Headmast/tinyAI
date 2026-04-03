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
from news_agent.token_counter import TokenCounter, TokenBudget, MODELS_NO_TEMPERATURE
from news_agent.usage_tracker import UsageTracker


MAX_ITERATIONS = 12
FINAL_ANSWER_MARKER = "FINAL_POST:"
DEFAULT_MAX_COMPLETION_TOKENS = 4_000
DEFAULT_CONTEXT_LIMIT = 128_000


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
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        context_limit: int = DEFAULT_CONTEXT_LIMIT,
        tracker: Optional[UsageTracker] = None,
    ):
        self.client = client
        self.model = model
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.max_completion_tokens = max_completion_tokens
        self.tracker = tracker
        self.dispatcher = ToolDispatcher(storage=storage)
        self._counter = TokenCounter(model=model)
        self._budget = TokenBudget(
            context_limit=context_limit,
            max_completion=max_completion_tokens,
        )
        self._token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._token_history: List[Dict[str, Any]] = []
        self._start_time: float = 0.0

    def run(self, task: str) -> Dict[str, Any]:
        """
        Запускает агентский цикл.
        Возвращает dict с финальным постом и метаданными включая детальную
        статистику токенов по каждой итерации.
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
        self._start_time = time.monotonic()

        while iteration < self.max_iterations:
            iteration += 1
            self._print_iteration(iteration)

            prompt_info = self._counter.count_messages(messages)
            prompt_tokens_before = prompt_info["total"]
            is_overflow, overflow_msg = self._budget.check_overflow(prompt_tokens_before)
            if overflow_msg and self.verbose:
                print(f"  {'🔴' if is_overflow else '⚠️ '} Токены: {overflow_msg}")
            if is_overflow:
                if self.verbose:
                    print(f"\n🔴 OVERFLOW: контекстное окно переполнено, агент остановлен")
                break

            effective_max = self._budget.effective_max_completion(prompt_tokens_before)
            response = self._call_llm(messages, max_completion_tokens=effective_max)
            assistant_message = response.choices[0].message

            self._update_token_usage(response)

            resp_text = assistant_message.content or ""
            completion_tokens = self._counter.count_response(resp_text)

            if self.verbose:
                self._print_token_breakdown(
                    iteration=iteration,
                    prompt_tokens=prompt_tokens_before,
                    completion_tokens=completion_tokens,
                    max_completion=effective_max,
                )

            self._token_history.append({
                "iteration": iteration,
                "prompt_tokens": prompt_tokens_before,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens_before + completion_tokens,
                "messages_count": len(messages),
            })

            messages.append({"role": "assistant", "content": resp_text})

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

        if self.verbose:
            self._print_token_summary()

        elapsed_ms = round((time.monotonic() - self._start_time) * 1000, 1)
        result: Dict[str, Any] = {
            "final_post": final_post or "Агент не создал финальный пост",
            "saved_post_id": saved_post_id,
            "iterations": iteration,
            "token_usage": self._token_usage.copy(),
            "token_history": list(self._token_history),
            "token_counter_method": self._counter.method_label,
            "elapsed_ms": elapsed_ms,
        }

        self._record_to_tracker(result)
        return result

    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        max_completion_tokens: Optional[int] = None,
    ):
        """Вызывает LLM с инструментами (function calling)."""
        effective = max_completion_tokens or self.max_completion_tokens
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "tool_choice": "auto",
            "max_completion_tokens": effective,
            "temperature": 0.5,
        }

        if self.model in MODELS_NO_TEMPERATURE:
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
            estimated = self._counter.count_response(content)
            self._token_usage["completion_tokens"] += estimated
            self._token_usage["total_tokens"] += estimated

    def _print_agent_header(self, task: str) -> None:
        if not self.verbose:
            return
        print("\n" + "═" * 65)
        print("  AUTONOMOUS NEWS AGENT  (ReAct)")
        print("═" * 65)
        print(f"  Задача:  {task[:80]}")
        print(f"  Модель:  {self.model}")
        print(f"  Лимит:   {self.max_iterations} итераций")
        print(f"  Max completion: {self.max_completion_tokens} токенов")
        print(f"  Счётчик токенов: {self._counter.method_label}")
        print("═" * 65)

    def _print_iteration(self, n: int) -> None:
        if self.verbose:
            print(f"\n--- Итерация {n} ---", flush=True)

    def _print_tool_call(self, name: str, args: Dict[str, Any]) -> None:
        if not self.verbose:
            return
        args_preview = json.dumps(args, ensure_ascii=False)[:120]
        print(f"  🔧 Вызов инструмента: {name}({args_preview}...)")

    def _print_token_breakdown(
        self,
        iteration: int,
        prompt_tokens: int,
        completion_tokens: int,
        max_completion: int,
    ) -> None:
        """Выводит краткую разбивку токенов для текущей итерации."""
        if not self.verbose:
            return
        context_pct = prompt_tokens / self._budget.context_limit * 100
        print(
            f"  📊 Токены: prompt={prompt_tokens:,} ({context_pct:.1f}%) | "
            f"completion={completion_tokens:,} | max_completion={max_completion:,}"
        )

    def _record_to_tracker(self, result: Dict[str, Any]) -> None:
        """Записывает итоги запуска агента в UsageTracker."""
        if self.tracker is None:
            return
        tu = result["token_usage"]
        self.tracker.record(
            command="agent",
            model=self.model,
            prompt_tokens=tu.get("prompt_tokens", 0),
            completion_tokens=tu.get("completion_tokens", 0),
            cost_usd=0.0,
            response_time_ms=result.get("elapsed_ms", 0.0),
            success=result["final_post"] != "Агент не создал финальный пост",
            tokens_estimated=(result["token_counter_method"] != "tiktoken"),
            iterations=result["iterations"],
        )

    def _print_token_summary(self) -> None:
        """Выводит итоговую таблицу токенов по всем итерациям."""
        if not self.verbose or not self._token_history:
            return
        print(f"\n{'─' * 65}")
        print(f"  ИТОГ ТОКЕНОВ ПО ИТЕРАЦИЯМ  ({self._counter.method_label})")
        print(f"  {'Iter':>4}  {'Prompt':>8}  {'Completion':>10}  {'Total':>8}  {'Msgs':>4}")
        print(f"  {'─' * 55}")
        for h in self._token_history:
            print(
                f"  {h['iteration']:>4}  {h['prompt_tokens']:>8,}  "
                f"{h['completion_tokens']:>10,}  {h['total_tokens']:>8,}  "
                f"{h['messages_count']:>4}"
            )
        total_p = self._token_usage["prompt_tokens"]
        total_c = self._token_usage["completion_tokens"]
        total_t = self._token_usage["total_tokens"]
        print(f"  {'─' * 55}")
        print(f"  {'API∑':>4}  {total_p:>8,}  {total_c:>10,}  {total_t:>8,}")
        print(f"{'─' * 65}\n")
