"""
memory_agent — агент с явной моделью памяти (memory layers).

Три слоя памяти:
  - ShortTermMemory:  текущий диалог (контекстное окно)
  - WorkingMemory:    данные текущей задачи (факты, цели, промежуточные результаты)
  - LongTermMemory:   профиль, решения, знания (персистентное хранение)

Агент на модели GLM-4.7 явно управляет маршрутизацией данных между слоями.
"""

from memory_agent.memory import ShortTermMemory, WorkingMemory, LongTermMemory, MemoryManager
from memory_agent.agent import MemoryAgent

__all__ = [
    "ShortTermMemory",
    "WorkingMemory",
    "LongTermMemory",
    "MemoryManager",
    "MemoryAgent",
]
__version__ = "1.0.0"
