"""
JournalistAgent — диалоговый ассистент-журналист с системой инвариантов.

Инварианты — жёсткие редакционные ограничения, которые ассистент
не имеет права нарушать ни при каких обстоятельствах.

Двухуровневая защита:
  Layer 1 (pre_check): быстрое сканирование по ключевым словам до вызова LLM.
                       Нарушение выявляется мгновенно, без расхода API-токенов.
  Layer 2 (llm):       инварианты инжектируются в system prompt как абсолютные
                       ограничения. LLM явно учитывает их при рассуждении
                       и отказывает, если запрос нарушает инвариант.

Формат ответа при нарушении:
  ОТКАЗ:[INV-XXX] <объяснение>

Инварианты хранятся в journalist_agent/invariants.json — отдельно от диалога.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from journalist_agent.invariants import InvariantStore, ViolationResult


REFUSAL_PATTERN = re.compile(
    r"^ОТКАЗ:\[(?P<inv_id>[A-Z0-9\-]+)\]\s*(?P<explanation>.+)",
    re.DOTALL | re.IGNORECASE,
)

BASE_JOURNALIST_PROMPT = """Ты — опытный журналист-редактор информационного агентства.

Твои задачи:
- Писать новостные заметки, репортажи, аналитические статьи, интервью.
- Соблюдать стандарты журналистики: точность, факты, нейтральный тон.
- Предлагать темы, структуру материалов, headline и lede.
- Помогать с редактурой и рерайтом текстов.

Ты работаешь в рамках редакционных инвариантов (см. секцию выше).
При каждом запросе ты СНАЧАЛА проверяешь, не нарушает ли он инварианты,
и только потом приступаешь к выполнению задачи."""


@dataclass
class AgentResponse:
    """
    Структурированный ответ журналистского ассистента.

    Fields:
        text:      видимый текст ответа (или объяснение отказа)
        allowed:   True если запрос прошёл проверку инвариантов
        violation: результат проверки (ViolationResult с деталями нарушения)
        tokens:    статистика использованных токенов
    """

    text: str
    allowed: bool
    violation: ViolationResult = field(default_factory=ViolationResult.ok)
    tokens: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })


class JournalistAgent:
    """
    Ассистент-журналист с системой редакционных инвариантов.

    Инварианты хранятся отдельно от диалога (invariants.json).
    При каждом запросе проводится двухуровневая проверка.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "zai-org/GLM-4.7",
        invariant_store: Optional[InvariantStore] = None,
        verbose: bool = True,
        max_completion_tokens: int = 3000,
        temperature: float = 0.6,
        max_history: int = 20,
    ) -> None:
        self.client = client
        self.model = model
        self.store = invariant_store or InvariantStore()
        self.verbose = verbose
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.max_history = max_history
        self._history: List[Dict[str, str]] = []
        self._total_tokens: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._turn = 0

    def chat(self, user_input: str) -> AgentResponse:
        """
        Обрабатывает сообщение пользователя.

        Последовательность:
          1. Pre-check по ключевым словам инвариантов (Layer 1).
          2. Если нарушение — немедленный отказ без вызова LLM.
          3. Иначе — вызов LLM с инвариантами в system prompt (Layer 2).
          4. Парсинг ответа: если LLM вернул ОТКАЗ:[...] — фиксируем нарушение.
        """
        self._turn += 1

        if self.verbose:
            print(f"\n{'─' * 60}")
            print(f"  Ход #{self._turn} | Пользователь: {user_input[:80]}")
            print(f"{'─' * 60}")

        violation = self.store.pre_check(user_input)
        if violation.violated:
            if self.verbose:
                self._print_violation(violation, layer="pre_check")
            return AgentResponse(
                text=self._format_refusal_text(violation),
                allowed=False,
                violation=violation,
            )

        system_prompt = self._build_system_prompt()
        messages = self._build_messages(system_prompt, user_input)

        raw_response = self._call_llm(messages)
        raw_text = raw_response.choices[0].message.content or ""
        self._update_tokens(raw_response)

        llm_violation, visible_text = self._parse_llm_response(raw_text)

        self._history.append({"role": "user", "content": user_input})
        self._history.append({"role": "assistant", "content": visible_text})
        self._trim_history()

        if llm_violation.violated:
            if self.verbose:
                self._print_violation(llm_violation, layer="llm")
            return AgentResponse(
                text=visible_text,
                allowed=False,
                violation=llm_violation,
                tokens=self._snapshot_tokens(raw_response),
            )

        if self.verbose:
            self._print_success(visible_text)

        return AgentResponse(
            text=visible_text,
            allowed=True,
            violation=ViolationResult.ok(),
            tokens=self._snapshot_tokens(raw_response),
        )

    def _trim_history(self) -> None:
        """Обрезает историю до max_history пар (user + assistant)."""
        max_msgs = self.max_history * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def reset_dialog(self) -> None:
        """Сбрасывает историю диалога (инварианты сохраняются)."""
        self._history.clear()
        self._turn = 0
        if self.verbose:
            print("  История диалога сброшена. Инварианты сохранены.")

    def get_invariants_summary(self) -> str:
        return self.store.summary()

    def get_total_tokens(self) -> Dict[str, int]:
        return dict(self._total_tokens)

    def _build_system_prompt(self) -> str:
        """Собирает финальный system prompt: инварианты + базовая роль."""
        invariants_section = self.store.build_prompt_section()
        return f"{invariants_section}\n\n{BASE_JOURNALIST_PROMPT}"

    def _build_messages(
        self,
        system_prompt: str,
        user_input: str,
    ) -> List[Dict[str, str]]:
        """Формирует messages[] для API: system + история + новый вопрос."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_input})
        return messages

    def _parse_llm_response(self, raw_text: str) -> tuple[ViolationResult, str]:
        """
        Разбирает ответ LLM.
        Возвращает (ViolationResult, видимый_текст).
        """
        stripped = raw_text.strip()
        match = REFUSAL_PATTERN.match(stripped)
        if match:
            inv_id = match.group("inv_id")
            explanation = match.group("explanation").strip()
            violation = ViolationResult.from_llm(
                invariant_id=inv_id,
                llm_explanation=explanation,
                invariants=self.store.active,
            )
            return violation, stripped

        return ViolationResult.ok(), stripped

    def _format_refusal_text(self, violation: ViolationResult) -> str:
        """Формирует читаемый текст отказа для пользователя."""
        inv = violation.invariant
        if inv is None:
            return f"Отказ: запрос нарушает редакционную политику."
        return (
            f"Я не могу выполнить этот запрос.\n\n"
            f"Причина: запрос нарушает редакционный инвариант [{inv.id}].\n"
            f"Правило: {inv.rule}\n\n"
            f"{inv.description}"
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> Any:
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
        }
        for attempt in range(3):
            try:
                return self.client.chat.completions.create(**params)
            except Exception as e:
                if attempt < 2:
                    if self.verbose:
                        print(f"  API ошибка: {e}. Повтор через 2с...")
                    time.sleep(2)
                else:
                    raise

    def _update_tokens(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            self._total_tokens["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            self._total_tokens["completion_tokens"] += getattr(usage, "completion_tokens", 0)
            self._total_tokens["total_tokens"] += getattr(usage, "total_tokens", 0)

    def _snapshot_tokens(self, response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage:
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _print_violation(self, violation: ViolationResult, layer: str) -> None:
        layer_label = "Pre-check (keywords)" if layer == "pre_check" else "LLM reasoning"
        inv = violation.invariant
        print(f"\n  ⛔  НАРУШЕНИЕ ИНВАРИАНТА  [{layer_label}]")
        if inv:
            print(f"       Инвариант: [{inv.id}] {inv.rule}")
        print(f"       Объяснение: {violation.explanation[:200]}")

    def _print_success(self, text: str) -> None:
        if not self.verbose:
            return
        preview = text[:200].replace("\n", " ")
        print(f"\n  ✅  Ответ: {preview}{'...' if len(text) > 200 else ''}")
        tu = self._total_tokens
        print(
            f"  Токены (всего): prompt={tu['prompt_tokens']:,} | "
            f"completion={tu['completion_tokens']:,} | "
            f"total={tu['total_tokens']:,}"
        )
