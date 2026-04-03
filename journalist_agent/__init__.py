"""
journalist_agent — ассистент-журналист с системой редакционных инвариантов и FSM.

Два режима работы:
  JournalistAgent     — диалоговый агент с двухуровневой проверкой инвариантов
  JournalistFSMAgent  — агент с контролируемым жизненным циклом задачи (FSM)

Поддерживаемые типы контента (JournalistFSMAgent):
  ContentType.ARTICLE        — статья (6 шагов)
  ContentType.NEWS_RESEARCH  — новостное исследование (6 шагов)
  ContentType.REVIEW         — ревью (6 шагов)
  ContentType.NOTE           — заметка (3 шага)
  ContentType.BLOG_ANALYSIS  — анализ блога (7 шагов)
"""

from journalist_agent.invariants import Invariant, InvariantStore, ViolationResult
from journalist_agent.agent import AgentResponse, JournalistAgent
from journalist_agent.workflow import (
    ContentType,
    JournalistTask,
    JournalistTaskStorage,
    TransitionError,
    WORKFLOWS,
    STATE_PAUSED,
    STATE_PUBLISHED,
)
from journalist_agent.fsm_agent import JournalistFSMAgent

__all__ = [
    "Invariant",
    "InvariantStore",
    "ViolationResult",
    "AgentResponse",
    "JournalistAgent",
    "ContentType",
    "JournalistTask",
    "JournalistTaskStorage",
    "TransitionError",
    "WORKFLOWS",
    "STATE_PAUSED",
    "STATE_PUBLISHED",
    "JournalistFSMAgent",
]
