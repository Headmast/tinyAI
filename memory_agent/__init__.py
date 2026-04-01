"""
memory_agent — агент с явной моделью памяти (memory layers) и персонализацией.

Три слоя памяти:
  - ShortTermMemory:  текущий диалог (контекстное окно)
  - WorkingMemory:    данные текущей задачи (факты, цели, промежуточные результаты)
  - LongTermMemory:   профиль, решения, знания (персистентное хранение)

Персонализация:
  - UserProfile:         профиль пользователя (стиль, формат, ограничения)
  - ProfileManager:      управление профилями (встроенные + пользовательские)
  - PersonalizedAgent:   агент с адаптацией под профиль пользователя
"""

from memory_agent.memory import ShortTermMemory, WorkingMemory, LongTermMemory, MemoryManager
from memory_agent.agent import MemoryAgent
from memory_agent.profile import UserProfile, ProfileManager, BUILTIN_PROFILES
from memory_agent.personalized_agent import PersonalizedAgent

__all__ = [
    "ShortTermMemory",
    "WorkingMemory",
    "LongTermMemory",
    "MemoryManager",
    "MemoryAgent",
    "UserProfile",
    "ProfileManager",
    "BUILTIN_PROFILES",
    "PersonalizedAgent",
]
__version__ = "2.0.0"
