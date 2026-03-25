"""
news_agent — AI-агент для автоматической генерации новостных постов
и диалоговых сессий с поддержкой истории контекста.

Архитектура:
  - NewsPipeline: многошаговый конвейер (Planner → Researcher → Writer → Editor → SEO)
  - AgentLoop: ReAct-агент для автономного режима
  - SessionStorage / ConversationSession: диалоговые сессии с персистентной историей
"""

from news_agent.storage import PostStorage
from news_agent.pipeline import NewsPipeline
from news_agent.agent import AgentLoop
from news_agent.session_manager import ConversationSession, SessionStorage

__all__ = ["PostStorage", "NewsPipeline", "AgentLoop", "ConversationSession", "SessionStorage"]
__version__ = "2.0.0"
