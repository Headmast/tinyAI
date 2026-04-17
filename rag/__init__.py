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
from typing import Any, Dict, List, Optional


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


@dataclass
class SourceRef:
    """Ссылка на источник, использованный в ответе."""
    index: int
    source: str          # путь к файлу
    section: str         # иерархический раздел
    chunk_id: str        # уникальный идентификатор чанка
    score: float         # similarity score


@dataclass
class Quote:
    """Дословная цитата из чанка."""
    index: int
    text: str
    verified: bool = False  # подтверждена ли как подстрока чанка


@dataclass
class CitedAnswer:
    """Структурированный ответ с источниками и цитатами."""
    answer: str
    sources: List[SourceRef] = field(default_factory=list)
    quotes: List[Quote] = field(default_factory=list)
    is_uncertain: bool = False   # True если "не знаю"
    raw_response: str = ""
    confidence: Optional[float] = None  # max score of retrieved chunks
    search_results: List[SearchResult] = field(default_factory=list)  # raw chunks for fast mode
