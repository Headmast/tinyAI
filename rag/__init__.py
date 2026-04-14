"""
RAG (Retrieval-Augmented Generation) — индексация документов с эмбеддингами.

Модули:
    chunker      — стратегии разбиения текста на чанки
    embedder     — генерация эмбеддингов через OpenAI
    index_store  — FAISS-индекс + SQLite-метаданные
    indexer      — оркестрация: загрузка → chunking → embedding → сохранение
    search       — поиск по индексу
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Document:
    """Загруженный документ с текстом и метаданными."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Чанк текста с метаданными."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Результат поиска по индексу."""
    chunk: Chunk
    score: float
    rank: int
