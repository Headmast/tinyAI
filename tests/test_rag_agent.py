"""
Тесты для RagAgent: ask_without_rag, ask_with_rag, compare.
Тесты answer_comparison и benchmark (без API).

Тесты помечены @pytest.mark.api — требуют CLOUD_API_KEY + готового индекса.
Все остальные тесты работают offline (mock).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import Chunk, SearchResult
from rag.rag_agent import RagAgent, _SYSTEM_NO_RAG, _SYSTEM_WITH_RAG_TMPL
from rag.answer_comparison import (
    build_comparison,
    compare_all,
    print_comparison_report,
    build_modes_comparison,
    compare_modes_all,
)
from rag.benchmark import CONTROL_QUESTIONS, evaluate_answer


# ── Фикстуры ─────────────────────────────────────────────────────────────────

def _make_openai_response(content: str = "test answer", total_tokens: int = 100) -> Any:
    """Создаёт mock-объект ответа OpenAI."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = 80
    usage.completion_tokens = 20
    usage.total_tokens = total_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_search_result(
    text: str = "chunk text",
    source: str = "docs/ARCHITECTURE.md",
    section: str = "Overview",
    score: float = 0.9,
) -> SearchResult:
    chunk = Chunk(
        text=text,
        metadata={
            "source": source,
            "section": section,
            "chunk_id": "test_001",
            "token_count": 50,
        },
    )
    return SearchResult(chunk=chunk, score=score, rank=1)


def _make_agent_with_mock_client() -> tuple[RagAgent, MagicMock]:
    """Создаёт RagAgent с mock-клиентом без реального API."""
    with patch.dict(os.environ, {"CLOUD_API_KEY": "test-key"}):
        agent = RagAgent(strategy="structure", top_k=3, verbose=False)
    mock_client = MagicMock()
    agent.client = mock_client
    return agent, mock_client


# ── TestRagAgentInit ──────────────────────────────────────────────────────────

class TestRagAgentInit:
    def test_defaults(self):
        with patch.dict(os.environ, {"CLOUD_API_KEY": "test-key"}):
            agent = RagAgent()
        assert agent.strategy == "structure"
        assert agent.top_k == 5
        assert agent.top_k_before == 10
        assert agent.top_k_after == 5
        assert agent.similarity_threshold == 0.30
        assert agent.enable_query_rewrite is True
        assert agent.enable_rerank is True
        assert agent.temperature == 0.3
        assert agent.verbose is False

    def test_custom_params(self):
        with patch.dict(os.environ, {"CLOUD_API_KEY": "test-key"}):
            agent = RagAgent(
                model="custom-model",
                strategy="fixed_size",
                top_k=3,
                top_k_before=9,
                top_k_after=4,
                similarity_threshold=0.45,
                enable_query_rewrite=False,
                enable_rerank=False,
                temperature=0.3,
                verbose=True,
            )
        assert agent.model == "custom-model"
        assert agent.strategy == "fixed_size"
        assert agent.top_k == 3
        assert agent.top_k_before == 9
        assert agent.top_k_after == 4
        assert agent.similarity_threshold == 0.45
        assert agent.enable_query_rewrite is False
        assert agent.enable_rerank is False
        assert agent.temperature == 0.3
        assert agent.verbose is True

    def test_missing_api_key_raises(self):
        env_without_keys = {k: v for k, v in os.environ.items()
                            if k not in ("CLOUD_API_KEY", "OPENAI_API_KEY")}
        with patch.dict(os.environ, env_without_keys, clear=True):
            with pytest.raises(EnvironmentError, match="API-ключ не найден"):
                RagAgent()

    def test_openai_api_key_fallback(self):
        env = {k: v for k, v in os.environ.items() if k != "CLOUD_API_KEY"}
        env["OPENAI_API_KEY"] = "openai-key"
        with patch.dict(os.environ, env, clear=True):
            agent = RagAgent()
        assert agent.client is not None


# ── TestAskWithoutRag ─────────────────────────────────────────────────────────

class TestAskWithoutRag:
    def test_returns_expected_keys(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response("answer text")

        result = agent.ask_without_rag("test question")

        assert set(result.keys()) == {"answer", "token_usage", "elapsed_ms", "mode"}

    def test_mode_is_no_rag(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        result = agent.ask_without_rag("q")
        assert result["mode"] == "no_rag"

    def test_answer_content(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response("my answer")

        result = agent.ask_without_rag("q")
        assert result["answer"] == "my answer"

    def test_messages_structure(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        agent.ask_without_rag("my question")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "my question"

    def test_system_prompt_no_rag(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        agent.ask_without_rag("q")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        system_msg = call_kwargs["messages"][0]["content"]
        assert system_msg == _SYSTEM_NO_RAG

    def test_token_usage_returned(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response(total_tokens=150)

        result = agent.ask_without_rag("q")
        assert result["token_usage"]["total_tokens"] == 150

    def test_elapsed_ms_positive(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        result = agent.ask_without_rag("q")
        assert result["elapsed_ms"] >= 0


# ── TestAskWithRag ────────────────────────────────────────────────────────────

class TestAskWithRag:
    def test_returns_expected_keys(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        mock_results = [_make_search_result("chunk1"), _make_search_result("chunk2")]
        with patch("rag.rag_agent.rag_search", return_value=mock_results):
            result = agent.ask_with_rag("q")

        assert {
            "answer",
            "sources",
            "chunks_used",
            "token_usage",
            "elapsed_ms",
            "mode",
        }.issubset(set(result.keys()))

    def test_mode_is_rag(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        with patch("rag.rag_agent.rag_search", return_value=[_make_search_result()]):
            result = agent.ask_with_rag("q")
        assert result["mode"] == "rag"

    def test_sources_extracted(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        r1 = _make_search_result(source="docs/ARCHITECTURE.md")
        r2 = _make_search_result(source="docs/MODELS_INFO.md")
        with patch("rag.rag_agent.rag_search", return_value=[r1, r2]):
            result = agent.ask_with_rag("q")

        assert "docs/ARCHITECTURE.md" in result["sources"]
        assert "docs/MODELS_INFO.md" in result["sources"]

    def test_duplicate_sources_deduplicated(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        r1 = _make_search_result(source="docs/ARCHITECTURE.md")
        r2 = _make_search_result(source="docs/ARCHITECTURE.md")
        with patch("rag.rag_agent.rag_search", return_value=[r1, r2]):
            result = agent.ask_with_rag("q")

        assert result["sources"].count("docs/ARCHITECTURE.md") == 1

    def test_chunks_used_count(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        mock_results = [_make_search_result() for _ in range(3)]
        with patch("rag.rag_agent.rag_search", return_value=mock_results):
            result = agent.ask_with_rag("q")

        assert result["chunks_used"] == 3

    def test_context_injected_into_system_prompt(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        r = _make_search_result(text="important content", source="docs/ARCHITECTURE.md")
        with patch("rag.rag_agent.rag_search", return_value=[r]):
            agent.ask_with_rag("q")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        system_msg = call_kwargs["messages"][0]["content"]
        assert "important content" in system_msg
        assert "docs/ARCHITECTURE.md" in system_msg

    def test_top_k_override(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        with patch("rag.rag_agent.rag_search", return_value=[]) as mock_search:
            agent.ask_with_rag("q", top_k=7)
            assert mock_search.call_args[1]["top_k"] == 7

    def test_query_rewrite_applied(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        agent.query_rewriter.rewrite = MagicMock(return_value="rewritten question")
        with patch("rag.rag_agent.rag_search", return_value=[]) as mock_search:
            result = agent.ask_with_rag("original question")

        assert mock_search.call_args[1]["query"] == "rewritten question"
        assert result["rewrite_applied"] is True
        assert result["original_query"] == "original question"
        assert result["rewritten_query"] == "rewritten question"

    def test_disable_rewrite_per_call(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        agent.query_rewriter.rewrite = MagicMock(return_value="rewritten question")
        with patch("rag.rag_agent.rag_search", return_value=[]) as mock_search:
            result = agent.ask_with_rag("original question", enable_query_rewrite=False)

        assert mock_search.call_args[1]["query"] == "original question"
        assert result["rewrite_applied"] is False

    def test_disable_rerank_per_call(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        with patch("rag.rag_agent.rag_search", return_value=[]) as mock_search:
            result = agent.ask_with_rag("q", enable_rerank=False)

        assert mock_search.call_args[1]["reranker"] is None
        assert result["rerank_applied"] is False

    def test_empty_search_results(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response("fallback")

        with patch("rag.rag_agent.rag_search", return_value=[]):
            result = agent.ask_with_rag("q")

        assert result["chunks_used"] == 0
        assert result["sources"] == []


# ── TestCompare ───────────────────────────────────────────────────────────────

class TestCompare:
    def test_returns_expected_keys(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        with patch("rag.rag_agent.rag_search", return_value=[]):
            result = agent.compare("q")

        assert "question" in result
        assert "no_rag" in result
        assert "rag" in result

    def test_question_preserved(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = _make_openai_response()

        with patch("rag.rag_agent.rag_search", return_value=[]):
            result = agent.compare("my question")

        assert result["question"] == "my question"

    def test_both_modes_called(self):
        agent, _ = _make_agent_with_mock_client()

        no_rag_result = {"answer": "a", "token_usage": {"total_tokens": 10}, "elapsed_ms": 1.0, "mode": "no_rag"}
        rag_result = {"answer": "b", "sources": [], "chunks_used": 0,
                      "token_usage": {"total_tokens": 20}, "elapsed_ms": 2.0, "mode": "rag"}

        agent.ask_without_rag = MagicMock(return_value=no_rag_result)
        agent.ask_with_rag = MagicMock(return_value=rag_result)

        result = agent.compare("q")

        agent.ask_without_rag.assert_called_once_with("q", stream=False)
        agent.ask_with_rag.assert_called_once_with("q", stream=False)
        assert result["no_rag"] == no_rag_result
        assert result["rag"] == rag_result

    def test_compare_modes_calls_four_configs(self):
        agent, _ = _make_agent_with_mock_client()
        mode_result = {
            "answer": "x",
            "sources": [],
            "chunks_used": 0,
            "token_usage": {"total_tokens": 1},
            "elapsed_ms": 1.0,
            "mode": "rag",
        }

        agent.ask_with_rag = MagicMock(return_value=mode_result)
        result = agent.compare_modes("q")

        assert result["question"] == "q"
        assert set(result.keys()) == {
            "question", "baseline", "rewrite_only", "rerank_only", "combined"
        }
        assert agent.ask_with_rag.call_count == 4


# ── TestAnswerComparison ──────────────────────────────────────────────────────

class TestAnswerComparison:
    def _make_compare_result(
        self,
        no_rag_answer: str = "",
        rag_answer: str = "",
        sources: list | None = None,
    ) -> dict:
        return {
            "question": "test question",
            "no_rag": {
                "answer": no_rag_answer,
                "token_usage": {"total_tokens": 50},
                "elapsed_ms": 100.0,
                "mode": "no_rag",
            },
            "rag": {
                "answer": rag_answer,
                "sources": sources or [],
                "chunks_used": 2,
                "token_usage": {"total_tokens": 200},
                "elapsed_ms": 300.0,
                "mode": "rag",
            },
        }

    def _make_spec(
        self,
        keywords: list | None = None,
        expected_sources: list | None = None,
    ) -> dict:
        return {
            "id": 1,
            "question": "test question",
            "expected_keywords": keywords or ["keyword1", "keyword2"],
            "expected_sources": expected_sources or ["docs/ARCHITECTURE.md"],
            "notes": "",
        }

    def test_keyword_hits_counted(self):
        result = self._make_compare_result(
            no_rag_answer="some answer with keyword1",
            rag_answer="answer with keyword1 and keyword2",
        )
        spec = self._make_spec(keywords=["keyword1", "keyword2"])
        comparison = build_comparison(result, spec)
        ev = comparison["evaluation"]
        assert ev["keyword_hits_no_rag"] == 1
        assert ev["keyword_hits_rag"] == 2

    def test_rag_wins_when_more_keywords(self):
        result = self._make_compare_result(
            no_rag_answer="one keyword1 here",
            rag_answer="both keyword1 and keyword2 here",
        )
        spec = self._make_spec(keywords=["keyword1", "keyword2"])
        comparison = build_comparison(result, spec)
        assert comparison["evaluation"]["rag_wins"] is True

    def test_rag_does_not_win_when_equal(self):
        result = self._make_compare_result(
            no_rag_answer="keyword1 keyword2",
            rag_answer="keyword1 keyword2",
        )
        spec = self._make_spec(keywords=["keyword1", "keyword2"])
        comparison = build_comparison(result, spec)
        assert comparison["evaluation"]["rag_wins"] is False

    def test_source_hits_counted(self):
        result = self._make_compare_result(
            sources=["docs/ARCHITECTURE.md", "docs/MODELS_INFO.md"]
        )
        spec = self._make_spec(expected_sources=["docs/ARCHITECTURE.md"])
        comparison = build_comparison(result, spec)
        assert comparison["evaluation"]["source_hits"] == 1
        assert comparison["evaluation"]["total_sources"] == 1

    def test_partial_source_match(self):
        result = self._make_compare_result(sources=["docs/ARCHITECTURE.md"])
        spec = self._make_spec(expected_sources=["docs/ARCHITECTURE.md", "docs/MODELS_INFO.md"])
        comparison = build_comparison(result, spec)
        assert comparison["evaluation"]["source_hits"] == 1
        assert comparison["evaluation"]["total_sources"] == 2

    def test_compare_all_aggregation(self):
        spec = self._make_spec(keywords=["kw1", "kw2"])
        c1 = build_comparison(
            self._make_compare_result("kw1", "kw1 kw2", ["docs/ARCHITECTURE.md"]),
            spec,
        )
        c2 = build_comparison(
            self._make_compare_result("no match", "kw1", []),
            spec,
        )
        agg = compare_all([c1, c2])
        assert agg["total_questions"] == 2
        assert agg["rag_wins"] == 2   # rag has more hits in both
        assert agg["rag_win_rate"] == 1.0

    def test_compare_all_empty(self):
        assert compare_all([]) == {}

    def test_build_modes_comparison(self):
        spec = self._make_spec(keywords=["kw1", "kw2"], expected_sources=["docs/ARCHITECTURE.md"])
        modes = {
            "question": "q",
            "baseline": {
                "answer": "kw1",
                "sources": [],
                "chunks_used": 1,
                "token_usage": {"total_tokens": 100},
                "elapsed_ms": 120,
            },
            "rewrite_only": {
                "answer": "kw1 kw2",
                "sources": ["docs/ARCHITECTURE.md"],
                "chunks_used": 2,
                "token_usage": {"total_tokens": 120},
                "elapsed_ms": 130,
            },
            "rerank_only": {
                "answer": "kw1",
                "sources": [],
                "chunks_used": 2,
                "token_usage": {"total_tokens": 110},
                "elapsed_ms": 125,
            },
            "combined": {
                "answer": "kw1 kw2",
                "sources": ["docs/ARCHITECTURE.md"],
                "chunks_used": 2,
                "token_usage": {"total_tokens": 130},
                "elapsed_ms": 140,
            },
        }

        result = build_modes_comparison(modes, spec)
        assert result["question_id"] == 1
        assert result["modes"]["baseline"]["keyword_hits"] == 1
        assert result["modes"]["rewrite_only"]["keyword_hits"] == 2
        assert result["wins_vs_baseline"]["rewrite_only"] is True
        assert result["wins_vs_baseline"]["combined"] is True

    def test_compare_modes_all(self):
        spec = self._make_spec(keywords=["kw1", "kw2"], expected_sources=["docs/ARCHITECTURE.md"])
        modes_1 = build_modes_comparison(
            {
                "question": "q1",
                "baseline": {"answer": "kw1", "sources": [], "token_usage": {"total_tokens": 100}, "elapsed_ms": 100, "chunks_used": 1},
                "rewrite_only": {"answer": "kw1 kw2", "sources": ["docs/ARCHITECTURE.md"], "token_usage": {"total_tokens": 110}, "elapsed_ms": 110, "chunks_used": 2},
                "rerank_only": {"answer": "kw1", "sources": [], "token_usage": {"total_tokens": 105}, "elapsed_ms": 105, "chunks_used": 2},
                "combined": {"answer": "kw1 kw2", "sources": ["docs/ARCHITECTURE.md"], "token_usage": {"total_tokens": 120}, "elapsed_ms": 120, "chunks_used": 2},
            },
            spec,
        )
        modes_2 = build_modes_comparison(
            {
                "question": "q2",
                "baseline": {"answer": "", "sources": [], "token_usage": {"total_tokens": 90}, "elapsed_ms": 90, "chunks_used": 1},
                "rewrite_only": {"answer": "kw1", "sources": [], "token_usage": {"total_tokens": 95}, "elapsed_ms": 95, "chunks_used": 1},
                "rerank_only": {"answer": "kw2", "sources": [], "token_usage": {"total_tokens": 94}, "elapsed_ms": 94, "chunks_used": 1},
                "combined": {"answer": "kw1 kw2", "sources": ["docs/ARCHITECTURE.md"], "token_usage": {"total_tokens": 102}, "elapsed_ms": 102, "chunks_used": 2},
            },
            spec,
        )

        agg = compare_modes_all([modes_1, modes_2])
        assert agg["total_questions"] == 2
        assert "baseline" in agg["modes"]
        assert "combined" in agg["modes"]
        assert agg["wins_vs_baseline"]["combined"]["count"] >= 1


# ── TestBenchmarkEvaluate ─────────────────────────────────────────────────────

class TestBenchmarkEvaluate:
    def test_control_questions_count(self):
        assert len(CONTROL_QUESTIONS) == 10

    def test_control_questions_have_required_fields(self):
        required = {"id", "question", "expected_keywords", "expected_sources", "notes"}
        for q in CONTROL_QUESTIONS:
            assert required.issubset(q.keys()), f"Q{q.get('id')} missing fields"

    def test_ids_are_sequential(self):
        ids = [q["id"] for q in CONTROL_QUESTIONS]
        assert ids == list(range(1, 11))

    def test_evaluate_answer_structure(self):
        compare_result = {
            "question": "test",
            "no_rag": {"answer": "pipeline agent scheduler", "token_usage": {"total_tokens": 10}, "elapsed_ms": 50.0},
            "rag": {"answer": "pipeline agent scheduler memory journalist",
                    "sources": ["docs/ARCHITECTURE.md"],
                    "chunks_used": 3, "token_usage": {"total_tokens": 200}, "elapsed_ms": 200.0},
        }
        ev = evaluate_answer(compare_result, CONTROL_QUESTIONS[0])
        assert "keyword_hits_no_rag" in ev
        assert "keyword_hits_rag" in ev
        assert "source_hits" in ev
        assert "rag_wins" in ev
        # Q1 keywords: pipeline, agent, scheduler, memory, journalist
        assert ev["keyword_hits_no_rag"] == 3  # pipeline, agent, scheduler
        assert ev["keyword_hits_rag"] == 5      # all 5

    def test_print_report_runs_without_error(self, capsys):
        """print_comparison_report не должен падать на пустом списке."""
        print_comparison_report([])
        captured = capsys.readouterr()
        assert captured.err == ""


# ── TestStreaming ──────────────────────────────────────────────────────────────

class TestStreaming:
    """Тесты streaming-вывода: _stream_llm, ask_without_rag(stream=True), ask_with_rag(stream=True)."""

    def _make_stream_chunks(self, tokens: list[str]):
        """Helper: создаёт итератор фейковых streaming chunk-ов."""
        chunks = []
        for t in tokens:
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = t
            choice = MagicMock()
            choice.delta = delta
            chunk.choices = [choice]
            chunks.append(chunk)
        return iter(chunks)

    def test_ask_without_rag_stream_returns_answer(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = self._make_stream_chunks(
            ["Hello", " world", "!"]
        )

        result = agent.ask_without_rag("q", stream=True)

        assert result["mode"] == "no_rag"
        assert "Hello" in result["answer"]
        assert "world" in result["answer"]
        assert result["token_usage"]["total_tokens"] == 0  # streaming has no usage

    def test_ask_without_rag_stream_calls_with_stream_param(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = self._make_stream_chunks(["ok"])

        agent.ask_without_rag("q", stream=True)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True

    def test_ask_with_rag_stream_returns_answer(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = self._make_stream_chunks(
            ["RAG", " answer"]
        )

        with patch("rag.rag_agent.rag_search", return_value=[]):
            result = agent.ask_with_rag("q", stream=True)

        assert result["mode"] == "rag"
        assert "RAG" in result["answer"]
        assert result["token_usage"]["total_tokens"] == 0

    def test_stream_thinking_tags_stripped_from_answer(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = self._make_stream_chunks(
            ["<thinking>", "reasoning here", "</thinking>", "final answer"]
        )

        result = agent.ask_without_rag("q", stream=True)

        assert "final answer" in result["answer"]
        assert "<thinking>" not in result["answer"]
        assert "reasoning here" not in result["answer"]

    def test_stream_elapsed_ms_positive(self):
        agent, mock_client = _make_agent_with_mock_client()
        mock_client.chat.completions.create.return_value = self._make_stream_chunks(["ok"])

        result = agent.ask_without_rag("q", stream=True)

        assert result["elapsed_ms"] >= 0


# ── TestBenchmarkRun ──────────────────────────────────────────────────────────

class TestBenchmarkRun:
    """Тесты для run_benchmark и print_final_summary из rag.benchmark."""

    def test_evaluate_answer_rag_wins_with_more_keywords(self):
        compare_result = {
            "question": "Какие агенты есть в TinyAI?",
            "no_rag": {"answer": "pipeline", "token_usage": {"total_tokens": 10}, "elapsed_ms": 50.0},
            "rag": {"answer": "pipeline agent scheduler memory journalist",
                    "sources": ["docs/ARCHITECTURE.md"],
                    "chunks_used": 3, "token_usage": {"total_tokens": 200}, "elapsed_ms": 200.0},
        }
        ev = evaluate_answer(compare_result, CONTROL_QUESTIONS[0])
        assert ev["rag_wins"] is True
        assert ev["keyword_hits_rag"] > ev["keyword_hits_no_rag"]

    def test_evaluate_answer_returns_all_fields(self):
        compare_result = {
            "question": "test",
            "no_rag": {"answer": "", "token_usage": {"total_tokens": 5}, "elapsed_ms": 10.0},
            "rag": {"answer": "", "sources": [], "chunks_used": 0,
                    "token_usage": {"total_tokens": 20}, "elapsed_ms": 100.0},
        }
        ev = evaluate_answer(compare_result, CONTROL_QUESTIONS[0])
        required_fields = {"keyword_hits_no_rag", "keyword_hits_rag", "source_hits", "rag_wins"}
        assert required_fields.issubset(set(ev.keys()))

    def test_evaluate_answer_source_matching(self):
        compare_result = {
            "question": "test",
            "no_rag": {"answer": "", "token_usage": {"total_tokens": 5}, "elapsed_ms": 10.0},
            "rag": {"answer": "answer with keywords",
                    "sources": ["docs/ARCHITECTURE.md"],
                    "chunks_used": 2, "token_usage": {"total_tokens": 50}, "elapsed_ms": 100.0},
        }
        q = CONTROL_QUESTIONS[0]
        ev = evaluate_answer(compare_result, q)
        # source_hits should count matching expected sources
        if "docs/ARCHITECTURE.md" in q.get("expected_sources", []):
            assert ev["source_hits"] >= 1

    def test_print_comparison_report_with_data(self, capsys):
        spec = {
            "id": 1, "question": "q", "expected_keywords": ["kw1"],
            "expected_sources": [], "notes": "",
        }
        result = {
            "question": "q",
            "no_rag": {"answer": "kw1", "token_usage": {"total_tokens": 10}, "elapsed_ms": 50.0},
            "rag": {"answer": "kw1", "sources": [], "chunks_used": 0,
                    "token_usage": {"total_tokens": 20}, "elapsed_ms": 100.0},
        }
        comparison = build_comparison(result, spec)
        print_comparison_report([comparison])
        captured = capsys.readouterr()
        assert captured.err == ""
        assert len(captured.out) > 0


# ── Интеграционные тесты (требуют реального API) ──────────────────────────────

@pytest.mark.api
class TestRagAgentIntegration:
    """Требует CLOUD_API_KEY и готового индекса в rag_data/."""

    def test_ask_without_rag_real(self):
        agent = RagAgent(verbose=False)
        result = agent.ask_without_rag("Что такое TinyAI?")
        assert result["mode"] == "no_rag"
        assert len(result["answer"]) > 10
        assert result["token_usage"]["total_tokens"] > 0

    def test_ask_with_rag_returns_sources(self):
        agent = RagAgent(strategy="structure", top_k=3, verbose=False)
        result = agent.ask_with_rag("Как устроен pipeline для генерации постов?")
        assert result["mode"] == "rag"
        assert len(result["sources"]) > 0
        assert result["chunks_used"] > 0

    def test_compare_both_modes(self):
        agent = RagAgent(strategy="structure", top_k=3, verbose=False)
        result = agent.compare("Как работает memory_agent?")
        assert "question" in result
        assert result["no_rag"]["mode"] == "no_rag"
        assert result["rag"]["mode"] == "rag"
