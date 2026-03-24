"""
news_agent — AI-агент для автоматической генерации новостных постов.

Архитектура: многошаговый pipeline (Planner → Researcher → Writer → Editor → SEO)
с ReAct-агентом для автономного режима.
"""

from news_agent.storage import PostStorage
from news_agent.pipeline import NewsPipeline
from news_agent.agent import AgentLoop

__all__ = ["PostStorage", "NewsPipeline", "AgentLoop"]
__version__ = "1.0.0"
