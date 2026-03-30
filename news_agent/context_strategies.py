"""
context_strategies — три стратегии управления контекстом диалога.

Стратегии:
    SlidingWindowStrategy  — хранит только последние N сообщений
    StickyFactsStrategy    — извлекает key-value факты + sliding window
    BranchingStrategy      — checkpoint'ы и ветки диалога

Базовый класс ContextStrategy определяет интерфейс.
Реестр STRATEGY_REGISTRY позволяет создавать стратегию по имени.
"""

from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from news_agent.token_counter import TokenCounter


# ─────────────────────────────────────────────────────────────
# Промпт для извлечения фактов (StickyFacts)
# ─────────────────────────────────────────────────────────────

FACTS_EXTRACTION_SYSTEM = (
    "Ты — аналитик диалога. Твоя задача: извлечь и обновить ключевые факты "
    "из переписки. Сохраняй только конкретные, важные данные.\n"
    "Категории фактов:\n"
    "- goal (цель проекта/задачи)\n"
    "- constraints (ограничения: бюджет, сроки, технические)\n"
    "- preferences (предпочтения пользователя)\n"
    "- decisions (принятые решения)\n"
    "- agreements (договорённости)\n"
    "- technical (технические детали: стек, платформы)\n"
    "- requirements (функциональные требования)\n"
    "- other (прочие важные факты)\n\n"
    "Верни ТОЛЬКО валидный JSON-объект, где ключи — категории, "
    "значения — строки с фактами. Если категория пуста — не включай её.\n"
    "Пример: {\"goal\": \"мобильное приложение для фитнеса\", "
    "\"constraints\": \"бюджет 2 млн, срок 6 мес\"}"
)

FACTS_EXTRACTION_USER_TEMPLATE = (
    "Текущие известные факты:\n{current_facts}\n\n"
    "Последние сообщения диалога:\n{recent_messages}\n\n"
    "Обнови факты с учётом новой информации. "
    "Сохрани все прежние факты, которые не противоречат новым. "
    "Верни обновлённый JSON."
)

FACTS_CONTEXT_PREFIX = "📌 Ключевые факты из диалога:\n"


# ─────────────────────────────────────────────────────────────
# Базовый класс
# ─────────────────────────────────────────────────────────────

class ContextStrategy(ABC):
    """
    Абстрактная стратегия управления контекстом.

    Каждая стратегия определяет:
      - какие сообщения отправлять в API
      - что делать после нового сообщения пользователя
      - как сериализоваться/десериализоваться
    """

    name: str = "base"

    @abstractmethod
    def get_messages_for_api(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Возвращает список сообщений для отправки в API."""
        ...

    def on_user_message(
        self,
        user_content: str,
        all_messages: List[Dict[str, Any]],
        client: Any = None,
        model: str = "",
    ) -> None:
        """Хук, вызываемый после добавления сообщения пользователя."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает метрики стратегии для логирования."""
        return {"strategy": self.name}

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextStrategy":
        """Десериализация из словаря."""
        ...


# ─────────────────────────────────────────────────────────────
# Стратегия 1: Sliding Window
# ─────────────────────────────────────────────────────────────

class SlidingWindowStrategy(ContextStrategy):
    """
    Хранит только последние window_size сообщений (user+assistant).
    Системные сообщения передаются всегда.
    Всё остальное отбрасывается.
    """

    name = "sliding_window"

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size
        self._messages_dropped: int = 0

    def get_messages_for_api(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        chat_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]

        if len(chat_msgs) <= self.window_size:
            self._messages_dropped = 0
            return system_msgs + chat_msgs

        self._messages_dropped = len(chat_msgs) - self.window_size
        return system_msgs + chat_msgs[-self.window_size:]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "window_size": self.window_size,
            "messages_dropped": self._messages_dropped,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "window_size": self.window_size,
            "messages_dropped": self._messages_dropped,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SlidingWindowStrategy":
        strategy = cls(window_size=data.get("window_size", 10))
        strategy._messages_dropped = data.get("messages_dropped", 0)
        return strategy


# ─────────────────────────────────────────────────────────────
# Стратегия 2: Sticky Facts / Key-Value Memory
# ─────────────────────────────────────────────────────────────

class StickyFactsStrategy(ContextStrategy):
    """
    Хранит отдельный блок facts (ключ-значение) с важными данными.
    После каждого сообщения пользователя обновляет facts через LLM.
    В API отправляет: system + facts_block + последние window_size сообщений.
    """

    name = "sticky_facts"

    def __init__(
        self,
        window_size: int = 6,
        max_facts: int = 20,
    ) -> None:
        self.window_size = window_size
        self.max_facts = max_facts
        self.facts: Dict[str, str] = {}
        self._facts_update_count: int = 0
        self._last_extraction_tokens: int = 0

    def get_messages_for_api(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        chat_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]

        result = list(system_msgs)

        if self.facts:
            facts_text = self._format_facts()
            result.append({
                "role": "system",
                "content": FACTS_CONTEXT_PREFIX + facts_text,
            })

        recent = chat_msgs[-self.window_size:] if len(chat_msgs) > self.window_size else chat_msgs
        result.extend(recent)
        return result

    def on_user_message(
        self,
        user_content: str,
        all_messages: List[Dict[str, Any]],
        client: Any = None,
        model: str = "",
    ) -> None:
        """Извлекает/обновляет факты через LLM после каждого сообщения пользователя."""
        if client is None:
            return

        chat_msgs = [m for m in all_messages if m.get("role") in ("user", "assistant")]
        recent = chat_msgs[-4:] if len(chat_msgs) > 4 else chat_msgs

        recent_text = "\n".join(
            f"{m['role'].upper()}: {m.get('content', '')}" for m in recent
        )
        current_facts_text = json.dumps(self.facts, ensure_ascii=False) if self.facts else "{}"

        extraction_messages = [
            {"role": "system", "content": FACTS_EXTRACTION_SYSTEM},
            {
                "role": "user",
                "content": FACTS_EXTRACTION_USER_TEMPLATE.format(
                    current_facts=current_facts_text,
                    recent_messages=recent_text,
                ),
            },
        ]

        params: Dict[str, Any] = {
            "model": model,
            "messages": extraction_messages,
            "max_completion_tokens": 500,
            "temperature": 0.1,
        }
        if "gpt-5-nano" in model:
            del params["temperature"]

        try:
            response = client.chat.completions.create(**params)
            raw = response.choices[0].message.content.strip()

            usage = getattr(response, "usage", None)
            if usage:
                self._last_extraction_tokens = getattr(usage, "total_tokens", 0)

            # Извлекаем JSON из ответа
            new_facts = self._parse_facts(raw)
            if new_facts:
                self.facts.update(new_facts)
                # Ограничиваем количество фактов
                if len(self.facts) > self.max_facts:
                    keys = list(self.facts.keys())
                    for k in keys[:-self.max_facts]:
                        del self.facts[k]
                self._facts_update_count += 1
        except Exception:
            pass  # Молча продолжаем — факты не обновятся, но диалог не ломается

    def _format_facts(self) -> str:
        """Форматирует факты для системного сообщения."""
        lines = []
        for key, value in self.facts.items():
            lines.append(f"• {key}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _parse_facts(raw: str) -> Optional[Dict[str, str]]:
        """Пытается распарсить JSON из ответа LLM."""
        # Ищем JSON в ответе
        raw = raw.strip()
        # Убираем markdown-обёртку если есть
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items() if v}
        except json.JSONDecodeError:
            # Пробуем найти JSON-объект в тексте
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(raw[start:end + 1])
                    if isinstance(data, dict):
                        return {str(k): str(v) for k, v in data.items() if v}
                except json.JSONDecodeError:
                    pass
        return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "window_size": self.window_size,
            "facts_count": len(self.facts),
            "facts_update_count": self._facts_update_count,
            "last_extraction_tokens": self._last_extraction_tokens,
            "facts": dict(self.facts),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "window_size": self.window_size,
            "max_facts": self.max_facts,
            "facts": dict(self.facts),
            "facts_update_count": self._facts_update_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StickyFactsStrategy":
        strategy = cls(
            window_size=data.get("window_size", 6),
            max_facts=data.get("max_facts", 20),
        )
        strategy.facts = data.get("facts", {})
        strategy._facts_update_count = data.get("facts_update_count", 0)
        return strategy


# ─────────────────────────────────────────────────────────────
# Стратегия 3: Branching (ветки диалога)
# ─────────────────────────────────────────────────────────────

class BranchingStrategy(ContextStrategy):
    """
    Поддерживает checkpoint'ы и ветки диалога.

    - checkpoint: именованная копия текущих сообщений
    - branch: отдельная линия диалога, начинающаяся от checkpoint
    - Каждая ветка хранит сообщения, добавленные после точки ветвления
    - Переключение между ветками мгновенное
    """

    name = "branching"

    def __init__(self) -> None:
        self.checkpoints: Dict[str, List[Dict[str, Any]]] = {}
        self.branches: Dict[str, _Branch] = {}
        self.current_branch: Optional[str] = None

    def save_checkpoint(
        self,
        name: str,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Сохраняет checkpoint — копию текущих сообщений."""
        self.checkpoints[name] = copy.deepcopy(messages)
        return f"Checkpoint '{name}' сохранён ({len(messages)} сообщений)"

    def create_branch(
        self,
        branch_name: str,
        checkpoint_name: str,
    ) -> str:
        """Создаёт новую ветку от указанного checkpoint'а."""
        if checkpoint_name not in self.checkpoints:
            return f"Checkpoint '{checkpoint_name}' не найден"
        if branch_name in self.branches:
            return f"Ветка '{branch_name}' уже существует"

        self.branches[branch_name] = _Branch(
            name=branch_name,
            checkpoint_name=checkpoint_name,
            base_messages=copy.deepcopy(self.checkpoints[checkpoint_name]),
            added_messages=[],
        )
        return (
            f"Ветка '{branch_name}' создана от checkpoint '{checkpoint_name}' "
            f"({len(self.checkpoints[checkpoint_name])} базовых сообщений)"
        )

    def switch_branch(self, branch_name: str) -> str:
        """Переключается на указанную ветку."""
        if branch_name not in self.branches:
            available = ", ".join(self.branches.keys()) or "(нет веток)"
            return f"Ветка '{branch_name}' не найдена. Доступные: {available}"
        self.current_branch = branch_name
        branch = self.branches[branch_name]
        total = len(branch.base_messages) + len(branch.added_messages)
        return f"Переключено на ветку '{branch_name}' ({total} сообщений)"

    def leave_branch(self) -> str:
        """Выходит из ветки обратно в основной диалог."""
        if self.current_branch is None:
            return "Вы уже в основном диалоге"
        name = self.current_branch
        self.current_branch = None
        return f"Вышли из ветки '{name}', вернулись в основной диалог"

    def get_messages_for_api(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self.current_branch is None:
            return list(messages)

        branch = self.branches.get(self.current_branch)
        if branch is None:
            return list(messages)

        return branch.base_messages + branch.added_messages

    def on_user_message(
        self,
        user_content: str,
        all_messages: List[Dict[str, Any]],
        client: Any = None,
        model: str = "",
    ) -> None:
        """Если мы в ветке — добавляем сообщение в ветку."""
        if self.current_branch and self.current_branch in self.branches:
            branch = self.branches[self.current_branch]
            branch.added_messages.append({
                "role": "user",
                "content": user_content,
            })

    def on_assistant_message(self, content: str) -> None:
        """Добавляет ответ ассистента в текущую ветку (если активна)."""
        if self.current_branch and self.current_branch in self.branches:
            self.branches[self.current_branch].added_messages.append({
                "role": "assistant",
                "content": content,
            })

    def list_info(self) -> str:
        """Возвращает информацию о checkpoint'ах и ветках."""
        lines = []
        lines.append("📍 Checkpoints:")
        if not self.checkpoints:
            lines.append("  (нет)")
        else:
            for name, msgs in self.checkpoints.items():
                lines.append(f"  • {name}: {len(msgs)} сообщений")

        lines.append("\n🌿 Ветки:")
        if not self.branches:
            lines.append("  (нет)")
        else:
            for name, branch in self.branches.items():
                marker = " ← текущая" if name == self.current_branch else ""
                total = len(branch.base_messages) + len(branch.added_messages)
                added = len(branch.added_messages)
                lines.append(
                    f"  • {name} (от '{branch.checkpoint_name}'): "
                    f"{total} сообщ. (+{added} новых){marker}"
                )

        active = self.current_branch or "основной диалог"
        lines.append(f"\n🔀 Активно: {active}")
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        branch_info = {}
        for name, branch in self.branches.items():
            branch_info[name] = {
                "checkpoint": branch.checkpoint_name,
                "base_messages": len(branch.base_messages),
                "added_messages": len(branch.added_messages),
                "total_messages": len(branch.base_messages) + len(branch.added_messages),
            }
        return {
            "strategy": self.name,
            "checkpoints_count": len(self.checkpoints),
            "branches_count": len(self.branches),
            "current_branch": self.current_branch,
            "branches": branch_info,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "checkpoints": {k: copy.deepcopy(v) for k, v in self.checkpoints.items()},
            "branches": {
                k: {
                    "name": b.name,
                    "checkpoint_name": b.checkpoint_name,
                    "base_messages": b.base_messages,
                    "added_messages": b.added_messages,
                }
                for k, b in self.branches.items()
            },
            "current_branch": self.current_branch,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BranchingStrategy":
        strategy = cls()
        strategy.checkpoints = data.get("checkpoints", {})
        strategy.current_branch = data.get("current_branch")
        for name, bd in data.get("branches", {}).items():
            strategy.branches[name] = _Branch(
                name=bd["name"],
                checkpoint_name=bd["checkpoint_name"],
                base_messages=bd.get("base_messages", []),
                added_messages=bd.get("added_messages", []),
            )
        return strategy


class _Branch:
    """Внутренний класс — одна ветка диалога."""

    __slots__ = ("name", "checkpoint_name", "base_messages", "added_messages")

    def __init__(
        self,
        name: str,
        checkpoint_name: str,
        base_messages: List[Dict[str, Any]],
        added_messages: List[Dict[str, Any]],
    ) -> None:
        self.name = name
        self.checkpoint_name = checkpoint_name
        self.base_messages = base_messages
        self.added_messages = added_messages


# ─────────────────────────────────────────────────────────────
# Реестр стратегий
# ─────────────────────────────────────────────────────────────

STRATEGY_REGISTRY: Dict[str, type] = {
    "sliding_window": SlidingWindowStrategy,
    "sticky_facts": StickyFactsStrategy,
    "branching": BranchingStrategy,
}


def create_strategy(name: str, **kwargs) -> ContextStrategy:
    """Создаёт стратегию по имени из реестра."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Неизвестная стратегия '{name}'. Доступные: {available}")
    return cls(**kwargs)


def strategy_from_dict(data: Dict[str, Any]) -> Optional[ContextStrategy]:
    """Восстанавливает стратегию из словаря (десериализация)."""
    name = data.get("name")
    if name == "sliding_window":
        return SlidingWindowStrategy.from_dict(data)
    elif name == "sticky_facts":
        return StickyFactsStrategy.from_dict(data)
    elif name == "branching":
        return BranchingStrategy.from_dict(data)
    return None
