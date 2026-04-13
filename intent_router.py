"""
IntentRouter — единый entry-point для всех агентов TinyAI.

Классифицирует запрос пользователя и маршрутизирует к подходящему агенту:
  - "Напиши статью"       → NewsPipeline
  - "Добавь задачу"       → MCPSchedulerAgent / MCPOrchestratorAgent
  - "Покажи логи"         → MCPOrchestratorAgent
  - "Что ты помнишь?"     → MemoryAgent
  - "Проверь текст"       → JournalistAgent
  - Общий вопрос          → прямой LLM-ответ

Использование:
    router = IntentRouter(client)
    result = router.handle("Напиши пост для телеграма про нейросети")
    print(result.agent_used, result.response)
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI


class Intent(str, Enum):
    """Типы намерений пользователя."""
    GENERATE_CONTENT = "generate_content"
    SCHEDULE_TASK = "schedule_task"
    SEARCH_LOGS = "search_logs"
    MEMORY = "memory"
    JOURNALISM = "journalism"
    PIPELINE = "pipeline"
    GENERAL = "general"


@dataclass
class RouteResult:
    """Результат маршрутизации запроса."""
    intent: Intent
    agent_used: str
    response: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Правила классификации (keyword-based, быстрый первый проход) ─────────

_INTENT_KEYWORDS: Dict[Intent, List[str]] = {
    Intent.GENERATE_CONTENT: [
        "напиши", "статья", "пост", "контент", "article",
        "сгенерируй", "создай пост", "write", "generate",
    ],
    Intent.SCHEDULE_TASK: [
        "задач", "scheduler", "расписание", "планировщик",
        "напоминание", "reminder", "бэкап", "backup",
        "запланируй", "добавь задачу",
    ],
    Intent.SEARCH_LOGS: [
        "лог", "логи", "logs", "история", "разговор",
        "поищи в", "найди в", "статистик", "usage",
    ],
    Intent.MEMORY: [
        "помнишь", "запомни", "память", "memory", "забудь",
        "вспомни", "предпочтения", "профиль",
    ],
    Intent.JOURNALISM: [
        "журналист", "проверь текст", "инвариант", "редактор",
        "рецензи", "отредактируй", "review", "invariant",
    ],
    Intent.PIPELINE: [
        "pipeline", "пайплайн", "цепочка", "поиск и суммаризация",
        "шаблон", "template", "дайджест", "digest",
    ],
}

# Маппинг intent → имя агента для отображения
_INTENT_AGENT_NAMES: Dict[Intent, str] = {
    Intent.GENERATE_CONTENT: "NewsPipeline",
    Intent.SCHEDULE_TASK: "MCPOrchestratorAgent",
    Intent.SEARCH_LOGS: "MCPOrchestratorAgent",
    Intent.MEMORY: "MemoryAgent",
    Intent.JOURNALISM: "JournalistAgent",
    Intent.PIPELINE: "PipelineExecutor",
    Intent.GENERAL: "DirectLLM",
}

# System prompt для LLM-классификации (fallback)
_CLASSIFY_PROMPT = """Ты — классификатор намерений. Определи тип запроса пользователя.

Типы:
- generate_content — создание статьи, поста, контента
- schedule_task — управление задачами, расписанием, бэкапами
- search_logs — поиск в логах, истории, статистика использования
- memory — работа с памятью агента, предпочтениями
- journalism — проверка текста, редактирование, рецензирование
- pipeline — запуск пайплайна контента (search→summarize→format→save)
- general — общий вопрос, не попадающий в другие категории

Ответь ОДНИМ СЛОВОМ — тип намерения. Ничего больше."""


def _keyword_classify(text: str) -> Optional[Intent]:
    """Быстрая классификация по ключевым словам. Без API-вызовов."""
    text_lower = text.lower()
    scores: Dict[Intent, int] = {}
    for intent, keywords in _INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[intent] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


class IntentRouter:
    """
    Маршрутизатор запросов пользователя к подходящему агенту.

    Стратегия классификации:
      1. Keyword matching (мгновенно, без API)
      2. LLM fallback (если keywords не дали результата)

    Агенты подключаются через register_handler().
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "zai-org/GLM-4.7",
        verbose: bool = True,
    ):
        self.client = client
        self.model = model
        self.verbose = verbose
        self._handlers: Dict[Intent, Callable[[str], str]] = {}
        self._route_log: List[RouteResult] = []

    def register_handler(self, intent: Intent, handler: Callable[[str], str]):
        """
        Регистрирует обработчик для конкретного намерения.

        handler: функция (user_input: str) -> str
        """
        self._handlers[intent] = handler

    def classify(self, text: str) -> Intent:
        """Классифицирует намерение пользователя."""
        # Фаза 1: keyword matching
        intent = _keyword_classify(text)
        if intent is not None:
            if self.verbose:
                print(f"  🏷️  Intent: {intent.value} (keywords)")
            return intent

        # Фаза 2: LLM fallback
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _CLASSIFY_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=20,
                temperature=0.0,
            )
            raw = (response.choices[0].message.content or "").strip().lower()
            try:
                intent = Intent(raw)
            except ValueError:
                intent = Intent.GENERAL
            if self.verbose:
                print(f"  🏷️  Intent: {intent.value} (LLM: '{raw}')")
            return intent
        except Exception:
            return Intent.GENERAL

    def handle(self, user_input: str) -> RouteResult:
        """
        Классифицирует запрос и передаёт обработчику соответствующего агента.

        Если обработчик для intent не зарегистрирован — отвечает напрямую через LLM.
        """
        intent = self.classify(user_input)
        agent_name = _INTENT_AGENT_NAMES.get(intent, "DirectLLM")

        handler = self._handlers.get(intent)
        if handler is None:
            # Fallback: прямой ответ от LLM
            response_text = self._direct_llm(user_input)
            agent_name = "DirectLLM"
        else:
            try:
                response_text = handler(user_input)
            except Exception as e:
                response_text = f"Ошибка агента {agent_name}: {e}"

        result = RouteResult(
            intent=intent,
            agent_used=agent_name,
            response=response_text,
        )
        self._route_log.append(result)
        return result

    def _direct_llm(self, text: str) -> str:
        """Прямой ответ от LLM без агентов."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Отвечай кратко и по существу."},
                    {"role": "user", "content": text},
                ],
                max_tokens=2000,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Ошибка LLM: {e}"

    @property
    def route_history(self) -> List[RouteResult]:
        """История маршрутизации."""
        return list(self._route_log)

    def get_stats(self) -> Dict[str, Any]:
        """Статистика маршрутизации по типам intent."""
        counts: Dict[str, int] = {}
        for r in self._route_log:
            counts[r.intent.value] = counts.get(r.intent.value, 0) + 1
        return {
            "total_requests": len(self._route_log),
            "by_intent": counts,
            "agents_used": list(dict.fromkeys(
                r.agent_used for r in self._route_log
            )),
        }
