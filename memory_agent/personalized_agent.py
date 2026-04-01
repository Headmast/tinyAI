"""
PersonalizedAgent — диалоговый агент с персонализацией на основе профиля пользователя.

Расширяет MemoryAgent:
  - Подключает UserProfile к каждому запросу через system prompt
  - Автоматически адаптирует стиль, формат и ограничения ответов
  - Записывает информацию о предпочтениях в долговременную память
  - Поддерживает горячую смену профиля без потери контекста
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from memory_agent.memory import MemoryManager
from memory_agent.profile import UserProfile, ProfileManager, BUILTIN_PROFILES


PERSONALIZED_SYSTEM_PROMPT = """Ты — интеллектуальный ассистент с трёхслойной моделью памяти и персонализацией.

У тебя есть три типа памяти:
1. КРАТКОСРОЧНАЯ — текущий диалог (хранится автоматически)
2. РАБОЧАЯ — данные текущей задачи (факты, цели, промежуточные результаты)
3. ДОЛГОВРЕМЕННАЯ — профиль пользователя, принятые решения, накопленные знания

ВАЖНО: Ты работаешь с конкретным пользователем, чей профиль описан ниже.
Адаптируй ВСЕ свои ответы под его предпочтения:
- Используй указанный стиль и тон
- Придерживайся предпочитаемого формата
- Соблюдай все ограничения
- Учитывай области экспертизы пользователя — не объясняй очевидное для него

После каждого ответа проанализируй диалог и сохрани полезную информацию:

```memory
{
  "working": {
    "facts": ["факт 1", "факт 2"],
    "goals": ["цель 1"],
    "context": {"ключ": "значение"}
  },
  "long_term": {
    "profile": {"ключ": "значение"},
    "decisions": [{"decision": "решение", "reasoning": "причина"}],
    "knowledge": [{"topic": "тема", "content": "содержание"}]
  }
}
```

Правила:
- Блок memory НЕОБЯЗАТЕЛЕН — добавляй только если есть что сохранить
- В profile сохраняй: уточнения предпочтений, новые интересы
- В decisions: выбор, рекомендации, принятые решения
- В knowledge: факты, полезные данные
- Активно используй данные из долговременной памяти в ответах
- ВСЕГДА соблюдай стиль и формат из профиля пользователя"""


MEMORY_BLOCK_PATTERN = re.compile(
    r"```memory\s*\n(\{.*?\})\s*\n```",
    re.DOTALL,
)


class PersonalizedAgent:
    """
    Диалоговый агент с персонализацией.

    Ключевое отличие от MemoryAgent:
      - Профиль пользователя инжектируется в system prompt при каждом запросе
      - Агент адаптирует стиль, формат и глубину ответов под профиль
      - Поддерживает смену профиля на лету
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "zai-org/GLM-4.7",
        profile: Optional[UserProfile] = None,
        memory_manager: Optional[MemoryManager] = None,
        verbose: bool = True,
        max_completion_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> None:
        self.client = client
        self.model = model
        self.profile = profile
        self.memory = memory_manager or MemoryManager()
        self.verbose = verbose
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self._total_tokens: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._turn_count = 0
        self._profile_history: List[Dict[str, Any]] = []

        if profile:
            self._apply_profile(profile)

    def set_profile(self, profile: UserProfile) -> None:
        """
        Устанавливает или меняет профиль пользователя.
        Профиль применяется немедленно к следующему запросу.
        """
        old_name = self.profile.name if self.profile else None
        self.profile = profile
        self._apply_profile(profile)
        self._profile_history.append({
            "action": "set_profile",
            "from": old_name,
            "to": profile.name,
            "turn": self._turn_count,
        })
        if self.verbose:
            print(f"  👤 Профиль установлен: {profile.name} ({profile.role})")

    def clear_profile(self) -> None:
        """Снимает текущий профиль."""
        old_name = self.profile.name if self.profile else None
        self.profile = None
        self.memory.clear_profile_section()
        self._profile_history.append({
            "action": "clear_profile",
            "from": old_name,
            "turn": self._turn_count,
        })
        if self.verbose:
            print("  👤 Профиль снят")

    def _apply_profile(self, profile: UserProfile) -> None:
        """Применяет профиль: инжектирует в MemoryManager и LongTermMemory."""
        self.memory.set_profile_section(profile.build_prompt_section())
        self.memory.long_term.set_profile("name", profile.name)
        self.memory.long_term.set_profile("role", profile.role)
        self.memory.long_term.set_profile("response_style", profile.response_style)
        self.memory.long_term.set_profile("preferred_format", profile.preferred_format)

    def chat(self, user_input: str) -> str:
        """
        Обрабатывает одно сообщение пользователя.
        Возвращает текстовый ответ (без блока memory).
        """
        self._turn_count += 1

        self.memory.short_term.add_message("user", user_input)

        system_prompt = self.memory.build_system_prompt(PERSONALIZED_SYSTEM_PROMPT)
        self.memory.short_term.set_system_prompt(system_prompt)

        messages = self.memory.short_term.get_messages_for_api()

        if self.verbose:
            profile_label = f" | Профиль: {self.profile.name}" if self.profile else ""
            print(f"\n{'─' * 60}")
            print(f"  Ход #{self._turn_count} | Сообщений: {len(messages)}{profile_label}")
            print(f"{'─' * 60}")

        response = self._call_llm(messages)
        raw_content = response.choices[0].message.content or ""

        self._update_token_usage(response)

        visible_text, memory_data = self._extract_memory_block(raw_content)

        if memory_data:
            self._apply_memory_updates(memory_data)

        self.memory.short_term.add_message("assistant", visible_text)

        if self.verbose:
            self._print_status(memory_data)

        return visible_text

    def get_memory_state(self) -> Dict[str, Any]:
        """Возвращает полное состояние памяти + профиль."""
        state = self.memory.get_full_state()
        state["profile"] = self.profile.to_dict() if self.profile else None
        state["profile_history"] = list(self._profile_history)
        return state

    def get_token_usage(self) -> Dict[str, int]:
        return dict(self._total_tokens)

    def reset_working_memory(self) -> None:
        """Очищает рабочую память (смена задачи). Профиль сохраняется."""
        self.memory.working.clear()
        if self.verbose:
            print("  🧹 Рабочая память очищена (профиль сохранён)")

    def reset_all(self) -> None:
        """Полный сброс всех слоёв памяти и профиля."""
        self.memory.short_term.clear()
        self.memory.working.clear()
        self.memory.long_term.clear()
        self.memory.clear_profile_section()
        self.profile = None
        self._turn_count = 0
        self._profile_history.clear()
        if self.verbose:
            print("  🧹 Вся память и профиль очищены")

    def get_profile_info(self) -> Dict[str, Any]:
        """Краткая информация о текущем профиле."""
        if not self.profile:
            return {"active": False}
        return {
            "active": True,
            "name": self.profile.name,
            "role": self.profile.role,
            "tone": self.profile.tone,
            "style": self.profile.response_style[:80] if self.profile.response_style else "",
            "constraints_count": len(self.profile.constraints),
            "expertise_count": len(self.profile.expertise_areas),
        }

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
                        print(f"  ⚠️  API ошибка: {e}. Повтор через 2с...")
                    time.sleep(2)
                else:
                    raise

    def _extract_memory_block(self, raw_text: str) -> tuple:
        """
        Извлекает блок ```memory из ответа.
        Возвращает (видимый_текст, memory_dict или None).
        """
        match = MEMORY_BLOCK_PATTERN.search(raw_text)
        if not match:
            return raw_text.strip(), None

        try:
            memory_data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return raw_text.strip(), None

        visible = MEMORY_BLOCK_PATTERN.sub("", raw_text).strip()
        return visible, memory_data

    def _apply_memory_updates(self, data: Dict[str, Any]) -> None:
        """Применяет обновления памяти из блока memory."""
        working = data.get("working", {})
        if working:
            facts = working.get("facts", [])
            if facts:
                self.memory.working.add_facts(facts)
            goals = working.get("goals", [])
            if goals:
                self.memory.working.goals.extend(goals)
            context = working.get("context", {})
            for k, v in context.items():
                self.memory.working.set_context(k, v)

        lt = data.get("long_term", {})
        if lt:
            profile = lt.get("profile", {})
            for k, v in profile.items():
                self.memory.long_term.set_profile(k, v)

            decisions = lt.get("decisions", [])
            for d in decisions:
                if isinstance(d, dict):
                    self.memory.long_term.add_decision(
                        d.get("decision", ""),
                        d.get("reasoning", ""),
                    )
                elif isinstance(d, str):
                    self.memory.long_term.add_decision(d)

            knowledge = lt.get("knowledge", [])
            for k in knowledge:
                if isinstance(k, dict):
                    self.memory.long_term.add_knowledge(
                        k.get("topic", ""),
                        k.get("content", ""),
                        k.get("source", ""),
                    )

    def _update_token_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            self._total_tokens["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            self._total_tokens["completion_tokens"] += getattr(usage, "completion_tokens", 0)
            self._total_tokens["total_tokens"] += getattr(usage, "total_tokens", 0)

    def _print_status(self, memory_data: Optional[Dict] = None) -> None:
        if not self.verbose:
            return
        state = self.memory.get_full_state()
        st = state["short_term"]
        wk = state["working"]
        lt = state["long_term"]

        print(f"\n  📊 Состояние памяти:")
        print(f"     Краткосрочная:  {st['message_count']} сообщений (лимит {st['max_messages']})")
        print(f"     Рабочая:        факты={len(self.memory.working.facts)}, "
              f"цели={len(self.memory.working.goals)}, "
              f"промежуточных={wk['intermediate_count']}")
        print(f"     Долговременная: профиль={len(lt['profile_keys'])} полей, "
              f"решения={lt['decisions_count']}, "
              f"знания={lt['knowledge_count']}")

        if self.profile:
            print(f"     Персонализация: {self.profile.name} ({self.profile.role})")

        if memory_data:
            print(f"  ✅ Агент обновил память:")
            if "working" in memory_data:
                w = memory_data["working"]
                if w.get("facts"):
                    print(f"     → рабочая.факты: +{len(w['facts'])}")
                if w.get("goals"):
                    print(f"     → рабочая.цели: +{len(w['goals'])}")
                if w.get("context"):
                    print(f"     → рабочая.контекст: {list(w['context'].keys())}")
            if "long_term" in memory_data:
                l = memory_data["long_term"]
                if l.get("profile"):
                    print(f"     → профиль: {list(l['profile'].keys())}")
                if l.get("decisions"):
                    print(f"     → решения: +{len(l['decisions'])}")
                if l.get("knowledge"):
                    print(f"     → знания: +{len(l['knowledge'])}")
        else:
            print(f"  ℹ️  Агент не обновлял память в этом ходе")

        tu = self._total_tokens
        print(f"  🔢 Токены (всего): prompt={tu['prompt_tokens']:,} | "
              f"completion={tu['completion_tokens']:,} | total={tu['total_tokens']:,}")
