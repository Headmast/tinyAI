"""Тесты для rag/chunker.py — стратегии разбиения текста на чанки."""

import pytest
from rag import Document, Chunk
from rag.chunker import FixedSizeChunker, StructureChunker


class TestFixedSizeChunker:
    """Тесты FixedSizeChunker."""

    def test_empty_text_returns_empty(self):
        chunker = FixedSizeChunker(chunk_size=100)
        doc = Document(text="", metadata={"source": "test"})
        assert chunker.chunk(doc) == []

    def test_whitespace_only_returns_empty(self):
        chunker = FixedSizeChunker(chunk_size=100)
        doc = Document(text="   \n\t  ", metadata={"source": "test"})
        assert chunker.chunk(doc) == []

    def test_short_text_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=512)
        doc = Document(text="Короткий текст.", metadata={"source": "test.md"})
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Короткий текст."
        assert chunks[0].metadata["strategy"] == "fixed_size"
        assert chunks[0].metadata["source"] == "test.md"

    def test_long_text_produces_multiple_chunks(self):
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=0)
        # Создаём текст длиннее 20 токенов
        text = " ".join([f"слово{i}" for i in range(100)])
        doc = Document(text=text, metadata={"source": "long.md"})
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_chunk_metadata_has_required_fields(self):
        chunker = FixedSizeChunker(chunk_size=512)
        doc = Document(text="Тестовый текст для проверки.", metadata={
            "source": "src.md", "title": "Заголовок"
        })
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        meta = chunks[0].metadata
        assert "chunk_id" in meta
        assert meta["source"] == "src.md"
        assert meta["title"] == "Заголовок"
        assert "token_count" in meta
        assert "char_count" in meta
        assert meta["chunk_index"] == 0

    def test_overlap_produces_overlapping_text(self):
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
        text = " ".join([f"word{i}" for i in range(100)])
        doc = Document(text=text, metadata={"source": "test"})
        chunks = chunker.chunk(doc)
        # С overlap чанков должно быть больше чем без
        chunker_no_overlap = FixedSizeChunker(chunk_size=20, chunk_overlap=0)
        chunks_no_overlap = chunker_no_overlap.chunk(doc)
        assert len(chunks) >= len(chunks_no_overlap)

    def test_all_chunks_are_chunk_type(self):
        chunker = FixedSizeChunker(chunk_size=50)
        doc = Document(text="Текст " * 50, metadata={"source": "test"})
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert isinstance(c, Chunk)


class TestStructureChunker:
    """Тесты StructureChunker."""

    def test_empty_text_returns_empty(self):
        chunker = StructureChunker()
        doc = Document(text="", metadata={"source": "test"})
        assert chunker.chunk(doc) == []

    def test_markdown_with_headings(self):
        text = "# Введение\nТекст введения.\n\n# Основная часть\nТекст основной части."
        chunker = StructureChunker()
        doc = Document(text=text, metadata={"source": "article.md"})
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1
        # Проверяем что метаданные strategy = "structure"
        for c in chunks:
            assert c.metadata.get("strategy") == "structure"

    def test_no_headings_single_chunk(self):
        chunker = StructureChunker()
        doc = Document(text="Просто текст без заголовков.", metadata={"source": "test"})
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1
