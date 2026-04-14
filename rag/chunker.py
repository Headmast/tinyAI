"""
Стратегии разбиения текста на чанки (chunking).

Две стратегии:
    FixedSizeChunker    — фиксированный размер в токенах с перекрытием
    StructureChunker    — по структуре Markdown (заголовки/секции)

Общий интерфейс: метод chunk(doc: Document) -> List[Chunk]
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from news_agent.token_counter import TokenCounter
from rag import Document, Chunk


class BaseChunker(ABC):
    """Абстрактный базовый класс для стратегий chunking."""

    @abstractmethod
    def chunk(self, doc: Document) -> List[Chunk]:
        """Разбивает документ на чанки."""
        ...


def _file_hash(text: str) -> str:
    """Короткий хеш текста для генерации chunk_id."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


# ── Strategy 1: Fixed-size ────────────────────────────────────────────────────


class FixedSizeChunker(BaseChunker):
    """
    Разбиение на чанки фиксированного размера (в токенах) с перекрытием.

    Параметры:
        chunk_size    — целевой размер чанка в токенах (по умолчанию 512)
        chunk_overlap — перекрытие между соседними чанками (по умолчанию 64)
        model         — модель для TokenCounter (по умолчанию cl100k_base)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        model: str = "gpt-5.4",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._counter = TokenCounter(model=model)

    def chunk(self, doc: Document) -> List[Chunk]:
        text = doc.text
        if not text.strip():
            return []

        file_hash = _file_hash(text)
        title = doc.metadata.get("title", "")
        source = doc.metadata.get("source", "")

        # Разбиваем текст на слова, чтобы собирать чанки по токен-лимиту
        words = text.split()
        chunks: List[Chunk] = []
        start = 0

        while start < len(words):
            # Наращиваем окно пока не превысим chunk_size токенов
            end = start
            while end < len(words):
                candidate = " ".join(words[start : end + 1])
                token_count = self._counter.count_text(candidate)
                if token_count > self.chunk_size and end > start:
                    break
                end += 1

            chunk_text = " ".join(words[start:end])
            token_count = self._counter.count_text(chunk_text)
            chunk_index = len(chunks)

            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "chunk_id": f"fs_{file_hash}_{chunk_index}",
                    "source": source,
                    "title": title,
                    "section": "N/A",
                    "strategy": "fixed_size",
                    "token_count": token_count,
                    "char_count": len(chunk_text),
                    "chunk_index": chunk_index,
                },
            ))

            # Сдвигаем старт с учётом overlap
            overlap_words = 0
            if self.chunk_overlap > 0 and end < len(words):
                # Отсчитываем overlap_words от конца текущего чанка назад
                overlap_text = ""
                for i in range(end - 1, start - 1, -1):
                    test = words[i] + (" " + overlap_text if overlap_text else "")
                    if self._counter.count_text(test) > self.chunk_overlap:
                        break
                    overlap_text = test
                    overlap_words += 1

            start = end - overlap_words

        return chunks


# ── Strategy 2: Structure-based ───────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class StructureChunker(BaseChunker):
    """
    Разбиение по структуре Markdown (заголовки/секции).

    Парсит заголовки (# ## ###) и разбивает текст по секциям.
    Если секция превышает max_chunk_tokens — делит по параграфам.
    Если параграф слишком большой — fallback на fixed-size внутри.

    Параметры:
        max_chunk_tokens — максимальный размер чанка в токенах
        model            — модель для TokenCounter
    """

    def __init__(
        self,
        max_chunk_tokens: int = 1024,
        model: str = "gpt-5.4",
    ) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self._counter = TokenCounter(model=model)

    def chunk(self, doc: Document) -> List[Chunk]:
        text = doc.text
        if not text.strip():
            return []

        file_hash = _file_hash(text)
        title = doc.metadata.get("title", "")
        source = doc.metadata.get("source", "")

        sections = self._split_by_headings(text)
        chunks: List[Chunk] = []
        chunk_index = 0

        for section_title, section_text in sections:
            section_path = section_title if section_title else title

            sub_chunks = self._split_section(section_text, section_path)

            for sub_text in sub_chunks:
                token_count = self._counter.count_text(sub_text)
                chunks.append(Chunk(
                    text=sub_text,
                    metadata={
                        "chunk_id": f"st_{file_hash}_{chunk_index}",
                        "source": source,
                        "title": title,
                        "section": section_path,
                        "strategy": "structure",
                        "token_count": token_count,
                        "char_count": len(sub_text),
                        "chunk_index": chunk_index,
                    },
                ))
                chunk_index += 1

        return chunks

    def _split_by_headings(self, text: str) -> List[tuple]:
        """
        Разбивает текст на секции по заголовкам Markdown.

        Возвращает список (section_path, section_text).
        Иерархия: заголовки разделяются через ' > '.
        """
        lines = text.split("\n")
        sections: List[tuple] = []
        current_headings: Dict[int, str] = {}
        current_lines: List[str] = []
        current_path = ""

        for line in lines:
            match = _HEADING_RE.match(line)
            if match:
                # Сохраняем предыдущую секцию
                if current_lines:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        sections.append((current_path, section_text))
                    current_lines = []

                level = len(match.group(1))
                heading_text = match.group(2).strip()

                # Обновляем иерархию: убираем все заголовки глубже текущего
                current_headings[level] = heading_text
                for k in list(current_headings.keys()):
                    if k > level:
                        del current_headings[k]

                # Формируем путь: H1 > H2 > H3
                path_parts = [
                    current_headings[k]
                    for k in sorted(current_headings.keys())
                ]
                current_path = " > ".join(path_parts)

                # Добавляем заголовок в текст секции
                current_lines.append(line)
            else:
                current_lines.append(line)

        # Последняя секция
        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append((current_path, section_text))

        return sections if sections else [("", text)]

    def _split_section(self, text: str, section_path: str) -> List[str]:
        """
        Разбивает секцию на под-чанки, если она слишком большая.

        Стратегия:
            1. Если помещается в max_chunk_tokens — возвращаем как есть
            2. Иначе — делим по параграфам (\\n\\n)
            3. Если параграф слишком большой — fixed-size fallback
        """
        token_count = self._counter.count_text(text)
        if token_count <= self.max_chunk_tokens:
            return [text]

        # Разбиваем по параграфам
        paragraphs = re.split(r"\n\n+", text)
        result: List[str] = []
        buffer = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            candidate_tokens = self._counter.count_text(candidate)

            if candidate_tokens <= self.max_chunk_tokens:
                buffer = candidate
            else:
                if buffer:
                    result.append(buffer)
                    buffer = ""

                # Проверяем, помещается ли параграф сам по себе
                para_tokens = self._counter.count_text(para)
                if para_tokens <= self.max_chunk_tokens:
                    buffer = para
                else:
                    # Fallback: разбиваем длинный параграф по словам
                    result.extend(self._split_long_text(para))

        if buffer:
            result.append(buffer)

        return result if result else [text]

    def _split_long_text(self, text: str) -> List[str]:
        """Fallback: разбивает длинный текст по словам до max_chunk_tokens."""
        words = text.split()
        parts: List[str] = []
        start = 0

        while start < len(words):
            end = start
            while end < len(words):
                candidate = " ".join(words[start : end + 1])
                if self._counter.count_text(candidate) > self.max_chunk_tokens and end > start:
                    break
                end += 1

            parts.append(" ".join(words[start:end]))
            start = end

        return parts
