"""
Трёхслойная модель памяти ассистента.

Слои:
  1. ShortTermMemory  — текущий диалог (список messages[], ограничен по размеру)
  2. WorkingMemory    — данные текущей задачи (факты, цели, промежуточные результаты)
  3. LongTermMemory   — профиль пользователя, решения, знания (JSON-файл)

MemoryManager координирует все три слоя и предоставляет единый интерфейс.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ShortTermMemory:
    """
    Краткосрочная память — текущий диалог.

    Хранит messages[] для передачи в API.
    При превышении max_messages — старые сообщения (кроме system) удаляются.
    """

    def __init__(self, max_messages: int = 40) -> None:
        self.max_messages = max_messages
        self.messages: List[Dict[str, str]] = []
        self._system_prompt: Optional[str] = None

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self._trim()

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Формирует список messages[] для отправки в API."""
        result: List[Dict[str, str]] = []
        if self._system_prompt:
            result.append({"role": "system", "content": self._system_prompt})
        for msg in self.messages:
            result.append({"role": msg["role"], "content": msg["content"]})
        return result

    def get_last_n(self, n: int = 5) -> List[Dict[str, str]]:
        return self.messages[-n:]

    def clear(self) -> None:
        self.messages.clear()

    def _trim(self) -> None:
        """Удаляет самые старые сообщения, если превышен лимит."""
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def dump(self) -> Dict[str, Any]:
        return {
            "type": "short_term",
            "max_messages": self.max_messages,
            "message_count": self.message_count,
            "messages": list(self.messages),
        }


class WorkingMemory:
    """
    Рабочая память — данные текущей задачи.

    Хранит:
      - current_task: описание текущей задачи
      - facts: список извлечённых фактов
      - goals: цели текущей задачи
      - intermediate: промежуточные результаты
      - context: произвольный контекст задачи (dict)

    Очищается при смене задачи.
    """

    def __init__(self) -> None:
        self.current_task: str = ""
        self.facts: List[str] = []
        self.goals: List[str] = []
        self.intermediate: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}

    def set_task(self, task: str, goals: Optional[List[str]] = None) -> None:
        self.current_task = task
        self.goals = goals or []
        self.facts.clear()
        self.intermediate.clear()
        self.context.clear()

    def add_fact(self, fact: str) -> None:
        if fact not in self.facts:
            self.facts.append(fact)

    def add_facts(self, facts: List[str]) -> None:
        for f in facts:
            self.add_fact(f)

    def add_intermediate(self, label: str, data: Any) -> None:
        self.intermediate.append({
            "label": label,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        })

    def set_context(self, key: str, value: Any) -> None:
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def get_summary(self) -> str:
        """Возвращает текстовую сводку рабочей памяти для вставки в промпт."""
        parts: List[str] = []
        if self.current_task:
            parts.append(f"Текущая задача: {self.current_task}")
        if self.goals:
            parts.append("Цели: " + "; ".join(self.goals))
        if self.facts:
            parts.append("Факты:\n" + "\n".join(f"  - {f}" for f in self.facts))
        if self.intermediate:
            last = self.intermediate[-3:]
            parts.append("Последние результаты:\n" + "\n".join(
                f"  [{r['label']}]: {str(r['data'])[:200]}" for r in last
            ))
        if self.context:
            parts.append("Контекст: " + json.dumps(self.context, ensure_ascii=False, default=str)[:300])
        return "\n\n".join(parts) if parts else "(рабочая память пуста)"

    def clear(self) -> None:
        self.current_task = ""
        self.facts.clear()
        self.goals.clear()
        self.intermediate.clear()
        self.context.clear()

    def dump(self) -> Dict[str, Any]:
        return {
            "type": "working",
            "current_task": self.current_task,
            "goals": list(self.goals),
            "facts": list(self.facts),
            "intermediate_count": len(self.intermediate),
            "context_keys": list(self.context.keys()),
        }


class LongTermMemory:
    """
    Долговременная память — профиль, решения, знания.

    Персистентное хранение в JSON-файле.
    Разделена на категории:
      - profile:   информация о пользователе (имя, предпочтения, стиль)
      - decisions: принятые решения и выводы
      - knowledge: накопленные знания и факты
    """

    def __init__(self, storage_path: str = "memory_data/long_term.json") -> None:
        self._path = Path(storage_path)
        self._data: Dict[str, Any] = {
            "profile": {},
            "decisions": [],
            "knowledge": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self) -> None:
        self._data["updated_at"] = datetime.now().isoformat()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # --- profile ---

    def set_profile(self, key: str, value: Any) -> None:
        self._data["profile"][key] = value
        self._save()

    def get_profile(self, key: str, default: Any = None) -> Any:
        return self._data["profile"].get(key, default)

    def get_full_profile(self) -> Dict[str, Any]:
        return dict(self._data["profile"])

    # --- decisions ---

    def add_decision(self, decision: str, reasoning: str = "") -> None:
        self._data["decisions"].append({
            "decision": decision,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def get_decisions(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        decisions = self._data["decisions"]
        if last_n:
            return decisions[-last_n:]
        return list(decisions)

    # --- knowledge ---

    def add_knowledge(self, topic: str, content: str, source: str = "") -> None:
        self._data["knowledge"].append({
            "topic": topic,
            "content": content,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def search_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Простой поиск по ключевому слову в topic и content."""
        query_lower = query.lower()
        return [
            k for k in self._data["knowledge"]
            if query_lower in k["topic"].lower() or query_lower in k["content"].lower()
        ]

    def get_all_knowledge(self) -> List[Dict[str, Any]]:
        return list(self._data["knowledge"])

    # --- summary ---

    def get_summary(self) -> str:
        """Текстовая сводка долговременной памяти для промпта."""
        parts: List[str] = []
        profile = self._data["profile"]
        if profile:
            profile_lines = [f"  {k}: {v}" for k, v in profile.items()]
            parts.append("Профиль пользователя:\n" + "\n".join(profile_lines))

        decisions = self._data["decisions"]
        if decisions:
            last3 = decisions[-3:]
            dec_lines = [f"  - {d['decision']}" for d in last3]
            parts.append("Последние решения:\n" + "\n".join(dec_lines))

        knowledge = self._data["knowledge"]
        if knowledge:
            last5 = knowledge[-5:]
            kn_lines = [f"  - [{k['topic']}]: {k['content'][:100]}" for k in last5]
            parts.append("Знания:\n" + "\n".join(kn_lines))

        return "\n\n".join(parts) if parts else "(долговременная память пуста)"

    def clear(self) -> None:
        self._data = {
            "profile": {},
            "decisions": [],
            "knowledge": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._save()

    def dump(self) -> Dict[str, Any]:
        return {
            "type": "long_term",
            "profile_keys": list(self._data["profile"].keys()),
            "decisions_count": len(self._data["decisions"]),
            "knowledge_count": len(self._data["knowledge"]),
            "storage_path": str(self._path),
        }


class MemoryManager:
    """
    Координатор всех трёх слоёв памяти.
    Предоставляет единый интерфейс и формирует контекст для LLM.
    """

    def __init__(
        self,
        short_term_limit: int = 40,
        long_term_path: str = "memory_data/long_term.json",
    ) -> None:
        self.short_term = ShortTermMemory(max_messages=short_term_limit)
        self.working = WorkingMemory()
        self.long_term = LongTermMemory(storage_path=long_term_path)
        self._profile_section: str = ""

    def set_profile_section(self, section: str) -> None:
        """Устанавливает секцию профиля пользователя для инъекции в system prompt."""
        self._profile_section = section

    def clear_profile_section(self) -> None:
        """Очищает секцию профиля."""
        self._profile_section = ""

    def build_system_prompt(self, base_prompt: str) -> str:
        """
        Собирает финальный system prompt, обогащённый данными из профиля,
        рабочей и долговременной памяти.
        """
        sections: List[str] = [base_prompt]

        if self._profile_section:
            sections.append(
                f"\n--- ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ ---\n{self._profile_section}"
            )

        lt_summary = self.long_term.get_summary()
        if lt_summary != "(долговременная память пуста)":
            sections.append(f"\n--- ДОЛГОВРЕМЕННАЯ ПАМЯТЬ ---\n{lt_summary}")

        wm_summary = self.working.get_summary()
        if wm_summary != "(рабочая память пуста)":
            sections.append(f"\n--- РАБОЧАЯ ПАМЯТЬ ---\n{wm_summary}")

        return "\n".join(sections)

    def get_full_state(self) -> Dict[str, Any]:
        """Возвращает полное состояние всех слоёв памяти."""
        return {
            "short_term": self.short_term.dump(),
            "working": self.working.dump(),
            "long_term": self.long_term.dump(),
        }

    def save_state_to_file(self, path: str) -> None:
        """Сохраняет полное состояние в JSON-файл для инспекции."""
        state = self.get_full_state()
        state["saved_at"] = datetime.now().isoformat()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
