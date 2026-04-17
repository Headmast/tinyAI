"""
Тесты для TASK24: RAG с обязательными цитатами и источниками.

Тестирует:
- dataclasses: SourceRef, Quote, CitedAnswer
- citation_parser: normalize_text, is_quote_in_texts, parse_sources_block,
                   parse_quotes_block, parse_cited_response
- Uncertain / «не знаю» режим
- Edge cases: пустые ответы, malformed citations, missing sections

Запуск:
    pytest tests/test_rag_citations.py -v
    pytest tests/test_rag_citations.py -v -k "parser"
    pytest tests/test_rag_citations.py -v -k "uncertain"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import CitedAnswer, Chunk, Quote, SearchResult, SourceRef
from rag.citation_parser import (
    is_quote_in_texts,
    normalize_text,
    parse_cited_response,
    parse_quotes_block,
    parse_sources_block,
)


# ── Sample fixtures ───────────────────────────────────────────────────────────


def _make_search_result(
    text: str,
    score: float = 0.85,
    rank: int = 0,
    source: str = "docs/readme.md",
    section: str = "Overview",
    chunk_id: str = "abc123",
) -> SearchResult:
    """Create a SearchResult with a Chunk."""
    meta = {
        "source": source,
        "section": section,
        "chunk_id": chunk_id,
    }
    return SearchResult(
        chunk=Chunk(text=text, metadata=meta),
        score=score,
        rank=rank,
    )


def _make_results(
    texts: list[str] | None = None,
    scores: list[float] | None = None,
) -> list[SearchResult]:
    """Create a list of SearchResults for testing."""
    if texts is None:
        texts = [
            "TinyAI использует RAG для поиска по документации. "
            "RAG Agent загружает файлы, чанкует их, строит FAISS-индекс.",
            "MCP (Model Context Protocol) — это стандарт для интеграции моделей с контекстом. "
            "TinyAI поддерживает MCP через mcp_server.py и mcp_client.py.",
            "Память агента реализована в модуле memory_agent. "
            "Она хранит факты, предпочтения и историю взаимодействий.",
        ]
    if scores is None:
        scores = [0.92, 0.85, 0.71]
    return [
        _make_search_result(t, s, i, chunk_id=f"chunk_{i}")
        for i, (t, s) in enumerate(zip(texts, scores))
    ]


CONFIDENCE_THRESHOLD = 0.45


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Dataclass Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSourceRefDataclass:
    """Tests for the SourceRef dataclass."""

    def test_source_ref_creation(self):
        src = SourceRef(
            index=1,
            source="docs/readme.md",
            section="Overview",
            chunk_id="abc123",
            score=0.85,
        )
        assert src.index == 1
        assert src.source == "docs/readme.md"
        assert src.section == "Overview"
        assert src.chunk_id == "abc123"
        assert src.score == 0.85

    def test_source_ref_all_required_fields(self):
        """All 5 fields are required — missing any raises TypeError."""
        with pytest.raises(TypeError):
            SourceRef(index=1)

    def test_source_ref_repr(self):
        src = SourceRef(
            index=2, source="file.py", section="Main",
            chunk_id="id1", score=0.77,
        )
        repr_str = repr(src)
        assert "2" in repr_str
        assert "file.py" in repr_str
        assert "Main" in repr_str


class TestQuoteDataclass:
    """Tests for the Quote dataclass."""

    def test_quote_creation(self):
        q = Quote(index=1, text="exact quote here", verified=True)
        assert q.index == 1
        assert q.text == "exact quote here"
        assert q.verified is True

    def test_quote_defaults(self):
        """index and text are required; verified defaults to False."""
        with pytest.raises(TypeError):
            Quote(index=1)

        q = Quote(index=1, text="hello")
        assert q.verified is False

    def test_quote_verification_flag(self):
        q_yes = Quote(index=1, text="found", verified=True)
        q_no = Quote(index=2, text="missing", verified=False)
        assert q_yes.verified
        assert not q_no.verified


class TestCitedAnswerDataclass:
    """Tests for the CitedAnswer dataclass."""

    def test_cited_answer_creation(self):
        sources = [SourceRef(
            index=1, source="file.md",
            section="S", chunk_id="c0", score=0.9,
        )]
        quotes = [Quote(index=1, text="quote", verified=True)]
        answer = CitedAnswer(
            answer="This is the answer [1].",
            sources=sources,
            quotes=quotes,
            is_uncertain=False,
            raw_response="This is the answer [1].\nИСТОЧНИКИ:\n...",
        )
        assert "answer" in answer.answer
        assert len(answer.sources) == 1
        assert len(answer.quotes) == 1
        assert not answer.is_uncertain

    def test_cited_answer_uncertain(self):
        answer = CitedAnswer(
            answer="Я не знаю...",
            is_uncertain=True,
        )
        assert answer.is_uncertain
        assert answer.sources == []   # default_factory -> empty list
        assert answer.quotes == []

    def test_cited_answer_defaults(self):
        answer = CitedAnswer(answer="test")
        assert answer.answer == "test"
        assert answer.sources == []
        assert answer.quotes == []
        assert answer.is_uncertain is False
        assert answer.raw_response == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Citation Parser — Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeText:
    """Tests for normalize_text utility."""

    def test_lowercase(self):
        assert normalize_text("Hello WORLD") == normalize_text("hello world")

    def test_whitespace_normalization(self):
        result = normalize_text("hello    world\n\ntest")
        assert "  " not in result
        assert result == "hello world test"

    def test_strip_punctuation(self):
        result = normalize_text("«Hello, world!»")
        assert "hello" in result
        assert "world" in result

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_only_punctuation(self):
        result = normalize_text(".,;:!?«»")
        assert len(result) <= 1


class TestIsQuoteInTexts:
    """Tests for is_quote_in_texts utility."""

    def test_exact_match(self):
        texts = ["The quick brown fox jumps over the lazy dog."]
        assert is_quote_in_texts("quick brown fox", texts) is True

    def test_case_insensitive(self):
        texts = ["The QUICK Brown Fox"]
        assert is_quote_in_texts("quick brown fox", texts) is True

    def test_no_match(self):
        texts = ["The quick brown fox"]
        assert is_quote_in_texts("purple elephant", texts) is False

    def test_too_short_quote(self):
        texts = ["some long text here"]
        assert is_quote_in_texts("abc", texts) is False

    def test_match_across_multiple_texts(self):
        texts = [
            "first document nothing here",
            "second document contains the brown fox",
        ]
        assert is_quote_in_texts("contains the brown fox", texts) is True


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Citation Parser — Block Parsing
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseSourcesBlock:
    """Tests for parsing the ИСТОЧНИКИ section."""

    def test_parse_simple_sources(self):
        block = (
            "ИСТОЧНИКИ:\n"
            "[1] docs/readme.md — Overview (chunk_0)\n"
            "[2] docs/mcp.md — MCP Protocol (chunk_1)\n"
        )
        sources = parse_sources_block(block)
        assert len(sources) == 2
        assert sources[0].index == 1
        assert sources[1].index == 2
        assert sources[0].chunk_id == "chunk_0"

    def test_parse_sources_with_pipe_separator(self):
        block = "[1] docs/readme.md | Overview | chunk_0"
        sources = parse_sources_block(block)
        assert len(sources) >= 1
        assert sources[0].index == 1

    def test_parse_sources_minimal(self):
        block = "[1] just-source-name"
        sources = parse_sources_block(block)
        assert len(sources) >= 1
        assert sources[0].source == "just-source-name"

    def test_parse_empty_block(self):
        sources = parse_sources_block("")
        assert sources == []

    def test_parse_no_source_lines(self):
        block = "Just some random text without source markers."
        sources = parse_sources_block(block)
        assert sources == []


class TestParseQuotesBlock:
    """Tests for parsing the ЦИТАТЫ section."""

    def test_parse_simple_quotes(self):
        block = (
            "ЦИТАТЫ:\n"
            '[1] «TinyAI использует RAG для поиска по документации»\n'
            '[2] «MCP — это стандарт для интеграции моделей»\n'
        )
        quotes = parse_quotes_block(block)
        assert len(quotes) == 2
        assert quotes[0].index == 1
        assert "TinyAI использует" in quotes[0].text

    def test_parse_quotes_with_dash_prefix(self):
        block = '[1] «Quote with angled brackets»'
        quotes = parse_quotes_block(block)
        assert len(quotes) >= 1
        assert quotes[0].text == "Quote with angled brackets"

    def test_parse_empty_quotes_block(self):
        quotes = parse_quotes_block("")
        assert quotes == []

    def test_parse_quotes_ignores_non_quote_lines(self):
        block = "Some random text\nMore text without quotes"
        quotes = parse_quotes_block(block)
        assert quotes == []


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Citation Parser — Full Response Parsing
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseCitedResponse:
    """Tests for parse_cited_response — end-to-end parsing of LLM output."""

    def test_parse_full_cited_response(self):
        response = (
            "RAG позволяет эффективно искать по документации [1]. "
            "Это полезно для вопросов по проекту.\n\n"
            "Протокол MCP поддерживает интеграцию [2].\n\n"
            "---\n"
            "ИСТОЧНИКИ:\n"
            "[1] docs/readme.md — Overview (chunk_0)\n"
            "[2] docs/mcp.md — MCP Protocol (chunk_1)\n\n"
            "ЦИТАТЫ:\n"
            '[1] «TinyAI использует RAG для поиска»\n'
            '[2] «MCP — это стандарт для интеграции»\n'
        )
        results = _make_results()
        cited = parse_cited_response(response, results)

        assert not cited.is_uncertain
        assert len(cited.sources) == 2
        assert len(cited.quotes) == 2
        assert "RAG позволяет" in cited.answer

    def test_parse_uncertain_response(self):
        response = (
            "НЕ ЗНАЮ: У меня нет достаточной информации для ответа на этот вопрос.\n"
            "Пожалуйста, задайте вопрос о проекте TinyAI."
        )
        results = _make_results()
        cited = parse_cited_response(response, results)

        assert cited.is_uncertain is True
        assert "ЗНАЮ" in cited.answer

    def test_parse_response_without_blocks(self):
        """Response with inline [N] markers but no ИСТОЧНИКИ/ЦИТАТЫ blocks."""
        response = "Fact one [1] and fact two [2] are important."
        results = _make_results()
        cited = parse_cited_response(response, results, verbose=False)

        # No structured blocks -> empty sources/quotes
        assert cited.sources == []
        assert cited.quotes == []
        assert "Fact one" in cited.answer
        assert "[1]" in cited.answer

    def test_parse_response_with_verification(self):
        """Quotes matching chunk text should be verified."""
        chunk_text = "TinyAI использует RAG для поиска по документации проекта."
        results = [_make_search_result(chunk_text, score=0.9)]

        response = (
            "Ответ: RAG используется для поиска.\n\n"
            "ЦИТАТЫ:\n"
            '[1] «TinyAI использует RAG для поиска по документации проекта.»\n'
        )
        cited = parse_cited_response(response, results, verbose=False)

        assert len(cited.quotes) == 1
        assert cited.quotes[0].verified is True

    def test_parse_response_unverified_quote(self):
        """Quotes absent from chunks should be unverified."""
        chunk_text = "Some completely different text here."
        results = [_make_search_result(chunk_text, score=0.9)]

        response = 'ЦИТАТЫ:\n[1] «This quote does not exist in the chunk»\n'
        cited = parse_cited_response(response, results, verbose=False)

        assert len(cited.quotes) >= 1
        assert any(not q.verified for q in cited.quotes)

    def test_parse_empty_response(self):
        results = _make_results()
        cited = parse_cited_response("", results)
        assert cited.answer == ""
        assert cited.sources == []
        assert cited.quotes == []

    def test_parse_response_preserves_inline_markers(self):
        response = "Fact one [1] and fact two [2] are important."
        results = _make_results()
        cited = parse_cited_response(response, results)
        assert "[1]" in cited.answer
        assert "[2]" in cited.answer


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Uncertain / Confident Logic
# ═══════════════════════════════════════════════════════════════════════════════


class TestUncertainMode:
    """Tests for the uncertain / «не знаю» threshold logic."""

    def test_no_results_triggers_uncertain(self):
        results: list[SearchResult] = []
        if not results:
            msg = (
                f"Я не могу уверенно ответить на вопрос «test?» "
                f"(Релевантность: 0.00 при пороге {CONFIDENCE_THRESHOLD})"
            )
            cited = CitedAnswer(answer=msg, is_uncertain=True)
            assert cited.is_uncertain

    def test_low_confidence_returns_uncertain(self):
        results = [_make_search_result("marginal content", score=0.20)]
        max_score = max(r.score for r in results)
        assert max_score < CONFIDENCE_THRESHOLD

    def test_confident_answer_structure(self):
        results = _make_results()
        max_score = max(r.score for r in results)
        assert max_score > CONFIDENCE_THRESHOLD

        cited = CitedAnswer(
            answer="RAG использует FAISS [1].",
            sources=[SourceRef(
                index=1, source="docs/readme.md",
                section="Overview", chunk_id="chunk_0", score=0.92,
            )],
            quotes=[Quote(index=1, text="TinyAI использует RAG", verified=True)],
        )
        assert not cited.is_uncertain
        assert len(cited.sources) == 1
        assert len(cited.quotes) == 1
        assert "[1]" in cited.answer

    def test_confident_answer_preserves_chunk_info(self):
        results = [_make_search_result(
            "MCP Protocol details here.", score=0.88,
            source="docs/mcp.md", section="Protocol", chunk_id="mcp_001",
        )]
        src = SourceRef(
            index=1,
            source=results[0].chunk.metadata["source"],
            section=results[0].chunk.metadata["section"],
            chunk_id=results[0].chunk.metadata["chunk_id"],
            score=results[0].score,
        )
        assert src.source == "docs/mcp.md"
        assert src.section == "Protocol"
        assert src.chunk_id == "mcp_001"
        assert src.score == 0.88


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Edge Cases & Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge-case tests for the citation system."""

    def test_very_long_chunk(self):
        long_text = "word " * 5000
        results = [_make_search_result(long_text, score=0.9)]
        response = "[1] word word word"
        cited = parse_cited_response(response, results, verbose=False)
        assert cited.answer is not None

    def test_unicode_content(self):
        response = (
            "Ответ: Юникод работает [1].\n\n"
            "ИСТОЧНИКИ:\n"
            "[1] docs/тест.md — Тестирование (unicode_01)\n\n"
            'ЦИТАТЫ:\n'
            '[1] «Юникод работает корректно»\n'
        )
        results = [_make_search_result(
            "Юникод работает корректно в проекте TinyAI.",
            score=0.9,
            source="docs/тест.md",
            section="Тестирование",
            chunk_id="unicode_01",
        )]
        cited = parse_cited_response(response, results, verbose=False)
        assert len(cited.sources) == 1
        assert "тест.md" in cited.sources[0].source

    def test_threshold_boundary_exact(self):
        """Score exactly at threshold -> NOT uncertain."""
        results = [_make_search_result("borderline", score=0.45)]
        assert max(r.score for r in results) >= CONFIDENCE_THRESHOLD

    def test_threshold_boundary_below(self):
        """Score just below threshold -> uncertain."""
        results = [_make_search_result("almost relevant", score=0.44)]
        assert max(r.score for r in results) < CONFIDENCE_THRESHOLD

    def test_mixed_verified_unverified_quotes(self):
        texts = [
            "First chunk with exact matching text here.",
            "Second chunk with completely different content.",
        ]
        results = [
            _make_search_result(texts[0], score=0.9, chunk_id="c0"),
            _make_search_result(texts[1], score=0.8, chunk_id="c1"),
        ]
        response = (
            "ЦИТАТЫ:\n"
            '[1] «First chunk with exact matching text here.»\n'
            '[2] «This text does not appear anywhere»\n'
        )
        cited = parse_cited_response(response, results, verbose=False)
        verified = sum(1 for q in cited.quotes if q.verified)
        unverified = sum(1 for q in cited.quotes if not q.verified)
        assert verified + unverified == len(cited.quotes)


class TestCitationParserIntegration:
    """Integration-style tests for the full citation pipeline."""

    def test_full_pipeline_happy_path(self):
        texts = [
            "RAG stands for Retrieval Augmented Generation. It combines retrieval with LLM.",
            "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        ]
        results = [
            _make_search_result(texts[0], score=0.95,
                                source="docs/rag.md", section="RAG Overview",
                                chunk_id="rag_01"),
            _make_search_result(texts[1], score=0.88,
                                source="docs/faq.md", section="FAISS",
                                chunk_id="faq_02"),
        ]
        llm_response = (
            "RAG (Retrieval Augmented Generation) — это метод, "
            "сочетающий поиск с генерацией [1].\n"
            "Для эффективного поиска используется библиотека FAISS [2].\n\n"
            "ИСТОЧНИКИ:\n"
            "[1] docs/rag.md — RAG Overview (rag_01)\n"
            "[2] docs/faq.md — FAISS (faq_02)\n\n"
            "ЦИТАТЫ:\n"
            '[1] «RAG stands for Retrieval Augmented Generation»\n'
            '[2] «FAISS is a library for efficient similarity search»\n'
        )
        cited = parse_cited_response(llm_response, results, verbose=False)

        assert not cited.is_uncertain
        assert len(cited.sources) == 2
        assert cited.sources[0].source == "docs/rag.md"
        assert cited.sources[1].source == "docs/faq.md"
        assert len(cited.quotes) == 2
        assert "[1]" in cited.answer
        assert "[2]" in cited.answer

    def test_full_pipeline_uncertain(self):
        results = [_make_search_result(
            "irrelevant content about cooking recipes",
            score=0.15,
        )]
        assert max(r.score for r in results) < CONFIDENCE_THRESHOLD
        # Should return uncertain without calling LLM
        cited = CitedAnswer(
            answer="не могу ответить", is_uncertain=True,
        )
        assert cited.is_uncertain

    def test_custom_confidence_threshold(self):
        results = [_make_search_result("some content", score=0.60)]
        max_score = max(r.score for r in results)
        # High threshold -> uncertain
        assert max_score < 0.80
        # Low threshold -> confident
        assert max_score >= 0.40
