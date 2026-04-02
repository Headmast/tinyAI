"""
journalist_agent — ассистент-журналист с системой редакционных инвариантов.

Инварианты — абсолютные ограничения, которые ассистент не имеет права нарушать.
Хранятся отдельно от диалога в journalist_agent/invariants.json.

Экспортируемые классы:
  JournalistAgent   — основной агент
  InvariantStore    — хранилище инвариантов
  Invariant         — единичный инвариант
  ViolationResult   — результат проверки на нарушение
  AgentResponse     — структурированный ответ агента
"""

from journalist_agent.invariants import Invariant, InvariantStore, ViolationResult
from journalist_agent.agent import AgentResponse, JournalistAgent

__all__ = [
    "Invariant",
    "InvariantStore",
    "ViolationResult",
    "AgentResponse",
    "JournalistAgent",
]
