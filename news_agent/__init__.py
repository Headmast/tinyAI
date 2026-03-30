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
from news_agent.usage_tracker import UsageTracker, RequestTimer, RequestRecord
from news_agent.token_counter import TokenCounter, DialogTokenTracker, TokenBudget
from news_agent.context_compressor import ContextCompressor, CompressionStats
from news_agent.context_strategies import (
    ContextStrategy, SlidingWindowStrategy, StickyFactsStrategy,
    BranchingStrategy, create_strategy, strategy_from_dict,
)
from news_agent.strategy_logger import StrategyLogger

__all__ = [
    "PostStorage", "NewsPipeline", "AgentLoop",
    "ConversationSession", "SessionStorage",
    "UsageTracker", "RequestTimer", "RequestRecord",
    "TokenCounter", "DialogTokenTracker", "TokenBudget",
    "ContextCompressor", "CompressionStats",
    "ContextStrategy", "SlidingWindowStrategy", "StickyFactsStrategy",
    "BranchingStrategy", "create_strategy", "strategy_from_dict",
    "StrategyLogger",
]
__version__ = "6.0.0"
