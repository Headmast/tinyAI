"""
Фабрики тестовых данных для TinyAI.

Использование:
    from tests.factories import make_message, make_conversation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def make_message(role: str = "user", content: str = "тестовое сообщение") -> Dict[str, str]:
    """Создаёт сообщение в формате OpenAI API."""
    return {"role": role, "content": content}


def make_conversation(n: int = 3) -> List[Dict[str, str]]:
    """Создаёт тестовый диалог из n пар user/assistant сообщений."""
    messages = [make_message("system", "Ты — тестовый ассистент.")]
    for i in range(n):
        messages.append(make_message("user", f"Вопрос {i + 1}"))
        messages.append(make_message("assistant", f"Ответ {i + 1}"))
    return messages


def make_journalist_task(
    topic: str = "Искусственный интеллект",
    content_type: str = "ARTICLE",
) -> Dict[str, Any]:
    """Создаёт тестовую задачу для FSM-агента журналиста."""
    return {
        "topic": topic,
        "content_type": content_type,
        "status": "planning",
        "steps_completed": [],
    }


def make_rag_document(
    text: Optional[str] = None,
    source: str = "test_source.md",
    chunk_id: int = 0,
) -> Dict[str, Any]:
    """Создаёт тестовый RAG-документ (чанк)."""
    return {
        "text": text or "Тестовый документ об искусственном интеллекте и машинном обучении.",
        "source": source,
        "chunk_id": chunk_id,
        "metadata": {"source": source, "chunk_id": chunk_id},
    }


def make_tool_call(
    name: str = "test_tool",
    arguments: Optional[Dict[str, Any]] = None,
    call_id: str = "call_test_123",
) -> Dict[str, Any]:
    """Создаёт тестовый tool_call в формате OpenAI API."""
    import json
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments or {}),
        },
    }
