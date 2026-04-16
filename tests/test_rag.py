"""
Тесты для RAG-модуля: chunking, index_store, search.

Тесты chunker и index_store работают offline (без API).
Тесты embedder/search помечены @pytest.mark.api — требуют OPENAI_API_KEY.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import Document, Chunk, SearchResult
from rag.chunker import FixedSizeChunker, StructureChunker, _file_hash
from rag.index_store import (
    FAISSIndexStore,
    init_db,
    save_chunks,
    get_chunks_by_faiss_ids,
    get_all_chunks,
    get_chunk_stats,
)


# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_MD = """# Test Document

## Section One

This is the first section with some content about testing.
It contains multiple sentences to ensure we have enough text.
The quick brown fox jumps over the lazy dog repeatedly.

## Section Two

### Subsection A

Here we discuss subsection A topics in detail.
More text to fill the subsection properly.

### Subsection B

And here is subsection B with different content.
Additional information about subsection B goes here.

## Section Three

Final section with concluding remarks about the document.
This section wraps up all the topics discussed above.
End of document content.
"""

LONG_TEXT = " ".join(["word"] * 2000)  # ~500 tokens


def _make_doc(text: str = SAMPLE_MD, source: str = "test.md", title: str = "Test") -> Document:
    return Document(text=text, metadata={"source": source, "title": title})


# ── FixedSizeChunker ──────────────────────────────────────────────────────────

class TestFixedSizeChunker:
    def test_basic_chunking(self):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        doc = _make_doc()
        chunks = chunker.chunk(doc)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_metadata_fields(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        doc = _make_doc()
        chunks = chunker.chunk(doc)

        required_fields = [
            "chunk_id", "source", "title", "section",
            "strategy", "token_count", "char_count", "chunk_index",
        ]
        for chunk in chunks:
            for field in required_fields:
                assert field in chunk.metadata, f"Missing field: {field}"

    def test_strategy_label(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(_make_doc())
        for c in chunks:
            assert c.metadata["strategy"] == "fixed_size"

    def test_chunk_size_respected(self):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=0)
        doc = _make_doc(LONG_TEXT)
        chunks = chunker.chunk(doc)

        for c in chunks:
            # Допускаем небольшое превышение (± 1 слово)
            assert c.metadata["token_count"] <= 55, (
                f"Chunk too large: {c.metadata['token_count']} tokens"
            )

    def test_empty_document(self):
        chunker = FixedSizeChunker()
        doc = _make_doc("")
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_chunk_ids_unique(self):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(_make_doc())
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"

    def test_chunk_index_sequential(self):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(_make_doc())
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_source_propagated(self):
        chunker = FixedSizeChunker(chunk_size=100)
        doc = _make_doc(source="docs/test.md", title="My Title")
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.metadata["source"] == "docs/test.md"
            assert c.metadata["title"] == "My Title"


# ── StructureChunker ──────────────────────────────────────────────────────────

class TestStructureChunker:
    def test_basic_chunking(self):
        chunker = StructureChunker(max_chunk_tokens=200)
        chunks = chunker.chunk(_make_doc())
        assert len(chunks) > 0

    def test_section_detection(self):
        chunker = StructureChunker(max_chunk_tokens=500)
        chunks = chunker.chunk(_make_doc())

        sections = [c.metadata["section"] for c in chunks]
        # Должны быть иерархические пути
        assert any("Section One" in s for s in sections)
        assert any("Section Two" in s for s in sections)

    def test_hierarchical_sections(self):
        chunker = StructureChunker(max_chunk_tokens=500)
        chunks = chunker.chunk(_make_doc())

        sections = [c.metadata["section"] for c in chunks]
        # Должны быть подсекции через " > "
        has_hierarchy = any(" > " in s for s in sections)
        assert has_hierarchy, f"No hierarchical sections found: {sections}"

    def test_strategy_label(self):
        chunker = StructureChunker(max_chunk_tokens=500)
        chunks = chunker.chunk(_make_doc())
        for c in chunks:
            assert c.metadata["strategy"] == "structure"

    def test_metadata_fields(self):
        chunker = StructureChunker(max_chunk_tokens=500)
        chunks = chunker.chunk(_make_doc())

        required_fields = [
            "chunk_id", "source", "title", "section",
            "strategy", "token_count", "char_count", "chunk_index",
        ]
        for chunk in chunks:
            for field in required_fields:
                assert field in chunk.metadata, f"Missing field: {field}"

    def test_long_section_split(self):
        # Документ с одной огромной секцией
        big_section = "# Big\n\n" + "\n\n".join(
            [f"Paragraph {i}. " + " ".join(["text"] * 100) for i in range(10)]
        )
        doc = _make_doc(big_section)
        chunker = StructureChunker(max_chunk_tokens=200)
        chunks = chunker.chunk(doc)

        # Должен разбить на несколько чанков
        assert len(chunks) > 1

    def test_empty_document(self):
        chunker = StructureChunker()
        chunks = chunker.chunk(_make_doc(""))
        assert chunks == []


# ── SQLite Store ──────────────────────────────────────────────────────────────

class TestSQLiteStore:
    def test_init_and_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            chunks = [
                Chunk(text="hello world", metadata={
                    "chunk_id": "test_0",
                    "source": "test.md",
                    "title": "Test",
                    "section": "Intro",
                    "strategy": "fixed_size",
                    "chunk_index": 0,
                    "token_count": 2,
                    "char_count": 11,
                }),
            ]
            save_chunks(db_path, chunks, embedding_dim=1536)

            result = get_all_chunks(db_path)
            assert len(result) == 1
            assert result[0]["chunk_id"] == "test_0"
            assert result[0]["text"] == "hello world"

    def test_filter_by_strategy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            chunks = [
                Chunk(text="chunk A", metadata={
                    "chunk_id": "a_0", "strategy": "fixed_size",
                    "source": "", "title": "", "section": "",
                    "chunk_index": 0, "token_count": 2, "char_count": 7,
                }),
                Chunk(text="chunk B", metadata={
                    "chunk_id": "b_0", "strategy": "structure",
                    "source": "", "title": "", "section": "",
                    "chunk_index": 0, "token_count": 2, "char_count": 7,
                }),
            ]
            save_chunks(db_path, chunks, embedding_dim=1536)

            fixed = get_all_chunks(db_path, strategy="fixed_size")
            assert len(fixed) == 1
            assert fixed[0]["strategy"] == "fixed_size"

            struct = get_all_chunks(db_path, strategy="structure")
            assert len(struct) == 1
            assert struct[0]["strategy"] == "structure"

    def test_get_by_faiss_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            chunks = [
                Chunk(text=f"chunk {i}", metadata={
                    "chunk_id": f"c_{i}", "strategy": "fixed_size",
                    "source": "", "title": "", "section": "",
                    "chunk_index": i, "token_count": 2, "char_count": 7,
                })
                for i in range(5)
            ]
            save_chunks(db_path, chunks, embedding_dim=1536)

            result = get_chunks_by_faiss_ids(db_path, [0, 2, 4])
            assert len(result) == 3

    def test_chunk_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            chunks = [
                Chunk(text="a " * 50, metadata={
                    "chunk_id": "s_0", "strategy": "fixed_size",
                    "source": "", "title": "", "section": "",
                    "chunk_index": 0, "token_count": 50, "char_count": 100,
                }),
                Chunk(text="b " * 100, metadata={
                    "chunk_id": "s_1", "strategy": "fixed_size",
                    "source": "", "title": "", "section": "",
                    "chunk_index": 1, "token_count": 100, "char_count": 200,
                }),
            ]
            save_chunks(db_path, chunks, embedding_dim=1536)

            stats = get_chunk_stats(db_path)
            assert "fixed_size" in stats
            assert stats["fixed_size"]["count"] == 2
            assert stats["fixed_size"]["avg_tokens"] == 75.0


# ── FAISS IndexStore ──────────────────────────────────────────────────────────

class TestFAISSIndexStore:
    def test_build_and_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FAISSIndexStore(index_dir=tmpdir)
            dim = 16
            n = 5
            embeddings = np.random.randn(n, dim).astype(np.float32)
            # L2-normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

            chunks = [
                Chunk(text=f"chunk {i}", metadata={
                    "chunk_id": f"c_{i}", "strategy": "test",
                    "source": "", "title": "", "section": "",
                    "chunk_index": i, "token_count": 2, "char_count": 7,
                })
                for i in range(n)
            ]

            store.build_index(embeddings, chunks, strategy="test")
            assert store.index_size("test") == n

            # Search with first vector — should find itself
            query = embeddings[0:1]
            scores, indices = store.search(query, strategy="test", top_k=3)
            assert indices.shape == (1, 3)
            assert indices[0, 0] == 0  # should be itself

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FAISSIndexStore(index_dir=tmpdir)
            dim = 16
            n = 3
            embeddings = np.random.randn(n, dim).astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

            chunks = [
                Chunk(text=f"c{i}", metadata={
                    "chunk_id": f"c_{i}", "strategy": "test",
                    "source": "", "title": "", "section": "",
                    "chunk_index": i, "token_count": 1, "char_count": 2,
                })
                for i in range(n)
            ]

            store.build_index(embeddings, chunks, strategy="test")
            store.save("test")

            # New store, load from disk
            store2 = FAISSIndexStore(index_dir=tmpdir)
            store2.load("test")
            assert store2.index_size("test") == n

            scores, indices = store2.search(embeddings[1:2], strategy="test", top_k=1)
            assert indices[0, 0] == 1


# ── Embedder (mocked) ────────────────────────────────────────────────────────

class TestEmbedderMocked:
    def test_embed_texts_shape(self):
        from rag.embedder import OpenAIEmbedder

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]

        embedder = OpenAIEmbedder()
        with patch.object(embedder._client.embeddings, "create", return_value=mock_response):
            result = embedder.embed_texts(["hello", "world"])
            assert result.shape == (2, 1536)
            # Check normalization
            norms = np.linalg.norm(result, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_embed_query_shape(self):
        from rag.embedder import OpenAIEmbedder

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.5] * 1536)]

        embedder = OpenAIEmbedder()
        with patch.object(embedder._client.embeddings, "create", return_value=mock_response):
            result = embedder.embed_query("test query")
            assert result.shape == (1, 1536)


# ── Document Loading ──────────────────────────────────────────────────────────

class TestDocumentLoading:
    def test_load_documents(self):
        from rag.indexer import load_documents

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test markdown files
            (Path(tmpdir) / "test1.md").write_text("# Title One\n\nContent one.", encoding="utf-8")
            (Path(tmpdir) / "test2.md").write_text("# Title Two\n\nContent two.", encoding="utf-8")
            (Path(tmpdir) / "empty.md").write_text("", encoding="utf-8")

            docs = load_documents(tmpdir)
            assert len(docs) == 2  # empty.md skipped
            titles = [d.metadata["title"] for d in docs]
            assert "Title One" in titles
            assert "Title Two" in titles

    def test_load_subdirectories(self):
        from rag.indexer import load_documents

        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "sub"
            sub.mkdir()
            (sub / "nested.md").write_text("# Nested\n\nContent.", encoding="utf-8")

            docs = load_documents(tmpdir)
            assert len(docs) == 1
            assert docs[0].metadata["title"] == "Nested"


# ── Search pipeline (offline) ────────────────────────────────────────────────

class TestSearchPipeline:
    def _fake_chunk_rows(self):
        return [
            {
                "faiss_id": 0,
                "chunk_id": "c0",
                "source": "docs/ARCHITECTURE.md",
                "title": "Arch",
                "section": "A",
                "strategy": "structure",
                "token_count": 10,
                "char_count": 20,
                "chunk_index": 0,
                "text": "high relevance",
            },
            {
                "faiss_id": 1,
                "chunk_id": "c1",
                "source": "docs/ARCHITECTURE.md",
                "title": "Arch",
                "section": "B",
                "strategy": "structure",
                "token_count": 11,
                "char_count": 21,
                "chunk_index": 1,
                "text": "medium relevance",
            },
            {
                "faiss_id": 2,
                "chunk_id": "c2",
                "source": "docs/ARCHITECTURE.md",
                "title": "Arch",
                "section": "C",
                "strategy": "structure",
                "token_count": 12,
                "char_count": 22,
                "chunk_index": 2,
                "text": "low relevance",
            },
        ]

    def test_threshold_and_top_k_after(self):
        from rag.search import search

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.ones((1, 16), dtype=np.float32)

        mock_store = MagicMock()
        mock_store.search.return_value = (
            np.array([[0.90, 0.50, 0.20]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )

        with patch("rag.search.OpenAIEmbedder", return_value=mock_embedder), patch(
            "rag.search.FAISSIndexStore", return_value=mock_store
        ), patch(
            "rag.search.get_chunks_by_faiss_ids", return_value=self._fake_chunk_rows()
        ):
            results = search(
                query="test",
                strategy="structure",
                index_dir="rag_data",
                top_k_before=3,
                top_k_after=2,
                similarity_threshold=0.40,
            )

        assert len(results) == 2
        assert results[0].score >= 0.5
        assert results[1].score >= 0.5
        assert all(r.rank in (1, 2) for r in results)

    def test_reranker_changes_order(self):
        from rag.search import search

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.ones((1, 16), dtype=np.float32)

        mock_store = MagicMock()
        mock_store.search.return_value = (
            np.array([[0.90, 0.80, 0.70]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )

        class _FakeReranker:
            def rerank(self, query, candidates):
                out = list(reversed(candidates))
                out[0].score = 0.99
                return out

        with patch("rag.search.OpenAIEmbedder", return_value=mock_embedder), patch(
            "rag.search.FAISSIndexStore", return_value=mock_store
        ), patch(
            "rag.search.get_chunks_by_faiss_ids", return_value=self._fake_chunk_rows()
        ):
            results = search(
                query="test",
                strategy="structure",
                index_dir="rag_data",
                top_k_before=3,
                top_k_after=2,
                similarity_threshold=0.0,
                reranker=_FakeReranker(),
            )

        assert len(results) == 2
        assert results[0].score == pytest.approx(0.99)
        assert results[0].rank == 1
        assert results[1].rank == 2
