"""
RagAgent — агент с режимами: без RAG, с RAG, и с цитатами/источниками.

Режимы:
    ask_without_rag      — прямой запрос к LLM без дополнительного контекста
    ask_with_rag         — поиск релевантных чанков → инжекция контекста → запрос к LLM
    ask_with_citations   — RAG с обязательными inline-цитатами, источниками и верификацией
    compare              — запускает оба режима, возвращает объединённый результат

CLI:
    python -m rag.rag_agent "Как устроен MCP Agent?" --strategy structure --citations
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import CitedAnswer, Quote, SearchResult, SourceRef
from rag.citation_parser import parse_cited_response, normalize_text
from rag.embedder import Embedder
from rag.search import search as rag_search
from rag.query_rewrite import QueryRewriter, DEFAULT_REWRITE_MODEL
from rag.reranker import LLMReranker, MathReranker, DEFAULT_RERANKER_MODEL

load_dotenv()

DEFAULT_MODEL = os.getenv("LLM_MODEL", "zai-org/GLM-4.7")
DEFAULT_BASE_URL = os.getenv("BASE_URL", "https://foundation-models.api.cloud.ru/v1")
DEFAULT_INDEX_DIR = Path(__file__).parent.parent / "rag_data"
DEFAULT_REWRITE_ENABLED = True
DEFAULT_RERANK_ENABLED = True
DEFAULT_TOP_K_BEFORE = 8
DEFAULT_TOP_K_AFTER = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.30
DEFAULT_CONFIDENCE_THRESHOLD = 0.40
DEFAULT_RAG_TEMPERATURE = 0.3
DEFAULT_CITATION_TEMPERATURE = 0.2

_GPT_MODELS = {"gpt-5.4", "gpt-5.4-mini", "gpt-5-nano", "gpt-5", "gpt-4o", "gpt-4o-mini"}
_GPT_BASE_URL = "https://api.openai.com/v1"

_SYSTEM_NO_RAG = """\
You are a helpful assistant. Answer the user's question concisely and accurately.
If you don't know the answer, say so rather than making things up.
"""

_SYSTEM_WITH_RAG_TMPL = """\
You are an expert assistant answering questions based ONLY on the provided context.

### Правила:
1. Отвечай на языке вопроса.
2. Используй ТОЛЬКО информацию из контекста ниже.
3. Если информации недостаточной — скажи об этом честно.
4. Пиши живым, понятным языком — не сухо и не как справочник. Используй абзацы и списки где уместно.
5. В конце ответа, после разделителя "───", перечисли источники, на которые опирался:
   [N] Имя файла — Раздел

Источники контекста: {sources}

### Контекст:
{context}
"""

_SYSTEM_WITH_RAG_CITED_TMPL = """\
You are an expert assistant answering questions based ONLY on the provided context.

### Правила:
1. Отвечай на языке вопроса.
2. Используй ТОЛЬКО информацию из контекста.
3. Пиши ответ в живом, понятном стиле — как объясняешь коллеге, а не как сухой справочник.
4. После основного ответа приведи 2-3 важней цитаты из документов, которые подтверждают твои выводы. Формат:
   > «цитата из текста» — [N] имя файла
5. В конце перечисли все источники в формате:
   [N] Имя файла — Раздел

### Пример формата:
Архитектура RAG в TinyAI включает несколько компонентов: индексацию документов, векторный поиск через FAISS, и генерацию ответов через LLM.

> «Система получает запрос и выполняет поиск по базе документов» — [1] ARCHITECTURE.md

> «FAISS используется для быстрого поиска ближайших векторов» — [2] ARCHITECTURE.md

───
📚 ИСТОЧНИКИ:
[1] ARCHITECTURE.md — TinyAI Архитектура
[2] ARCHITECTURE.md — TinyAI Архитектура

### Контекст:
{context}
"""


class RagAgent:
    """
    Агент с режимами: без RAG, с RAG, и с цитатами/источниками.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        strategy: Optional[str] = "structure",
        top_k: int = 5,
        top_k_before: int = DEFAULT_TOP_K_BEFORE,
        top_k_after: Optional[int] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        enable_query_rewrite: bool = DEFAULT_REWRITE_ENABLED,
        enable_rerank: bool = DEFAULT_RERANK_ENABLED,
        rewrite_model: str = DEFAULT_REWRITE_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        temperature: float = DEFAULT_RAG_TEMPERATURE,
        verbose: bool = False,
        provider: str = "cloud",
    ) -> None:
        self.model = model
        self.index_dir = Path(index_dir)
        self.strategy = strategy
        self.top_k = top_k
        self.top_k_before = top_k_before
        self.top_k_after = top_k_after if top_k_after is not None else top_k
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.enable_query_rewrite = enable_query_rewrite
        self.enable_rerank = enable_rerank
        self.rewrite_model = rewrite_model
        self.reranker_model = reranker_model
        self.temperature = temperature
        self.verbose = verbose
        self.provider = provider

        self._gpt_client: Optional[OpenAI] = None
        self._embedder: Optional[Embedder] = None

        if provider == "ollama":
            from core.config import get_llm_client
            self.client = get_llm_client("ollama")
            # Локальный эмбеддер
            from rag.local_embedder import OllamaEmbedder
            self._embedder = OllamaEmbedder()
            # MathReranker для локального режима (без LLM-вызовов для ранжирования)
            self.reranker = MathReranker(
                enabled=self.enable_rerank,
                verbose=self.verbose,
            )
            self.query_rewriter = QueryRewriter(
                client=self.client,
                model=self.model,
                enabled=self.enable_query_rewrite,
                verbose=self.verbose,
            )
        else:
            api_key = os.getenv("CLOUD_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "API-ключ не найден. Задайте CLOUD_API_KEY или OPENAI_API_KEY в .env"
                )

            self.client = OpenAI(
                base_url=os.getenv("BASE_URL", DEFAULT_BASE_URL),
                api_key=api_key,
                timeout=120.0,
            )
            self.query_rewriter = QueryRewriter(
                client=self.client,
                model=self.rewrite_model,
                enabled=self.enable_query_rewrite,
                verbose=self.verbose,
            )
            self.reranker = MathReranker(
                enabled=self.enable_rerank,
                verbose=self.verbose,
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_client(self) -> OpenAI:
        """Return the appropriate OpenAI client for the configured model."""
        if self.provider == "ollama":
            return self.client
        if self.model in _GPT_MODELS:
            if self._gpt_client is None:
                openai_key = os.getenv("OPENAI_API_KEY")
                if not openai_key:
                    raise EnvironmentError(
                        "OPENAI_API_KEY не найден для GPT-модели. "
                        "Задайте его в .env файле."
                    )
                self._gpt_client = OpenAI(
                    base_url=_GPT_BASE_URL,
                    api_key=openai_key,
                    timeout=120.0,
                )
            return self._gpt_client
        return self.client

    def _build_numbered_context(
        self, results: List[SearchResult]
    ) -> Tuple[str, List[SourceRef]]:
        """Build numbered context with [N] markers for citation prompting."""
        context_parts: List[str] = []
        sources: List[SourceRef] = []

        for i, r in enumerate(results, start=1):
            source = r.chunk.metadata.get("source", "unknown")
            section = r.chunk.metadata.get("section", "")
            chunk_id = r.chunk.metadata.get("chunk_id", "")

            sources.append(SourceRef(
                index=i,
                source=source,
                section=section,
                chunk_id=chunk_id,
                score=round(r.score, 4),
            ))

            header = (
                f"[{i}] Источник: {source} | Раздел: {section} | "
                f"ID: {chunk_id}"
            )
            context_parts.append(f"{header}\n{r.chunk.text}")

        return "\n\n---\n\n".join(context_parts), sources

    def _build_uncertain_message(self, question: str, max_score: float) -> str:
        """Build a pre-formatted 'I don't know' response."""
        return (
            f"Я не могу уверенно ответить на вопрос «{question}» на основе "
            f"имеющейся документации проекта TinyAI. "
            f"(Релевантность найденных фрагментов: {max_score:.2f} "
            f"при пороге {self.confidence_threshold})\n\n"
            f"Пожалуйста, уточните вопрос или задайте вопрос о компонентах, "
            f"архитектуре или возможностях проекта TinyAI."
        )

    # ── Публичный интерфейс ───────────────────────────────────────────────────

    def ask_without_rag(self, question: str, stream: bool = False) -> Dict[str, Any]:
        """Прямой запрос к LLM без поиска по документам."""
        messages = [
            {"role": "system", "content": _SYSTEM_NO_RAG},
            {"role": "user", "content": question},
        ]

        t0 = time.monotonic()
        if stream:
            answer = self._stream_llm(messages)
            elapsed_ms = (time.monotonic() - t0) * 1000
            token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        else:
            response = self._call_llm(messages)
            elapsed_ms = (time.monotonic() - t0) * 1000
            answer = response.choices[0].message.content or ""
            token_usage = self._extract_usage(response)

        if self.verbose and not stream:
            print(
                f"  [no_rag] tokens={token_usage['total_tokens']} "
                f"elapsed={elapsed_ms:.0f}ms"
            )

        return {
            "answer": answer,
            "token_usage": token_usage,
            "elapsed_ms": round(elapsed_ms, 1),
            "mode": "no_rag",
        }

    def ask_with_rag(
        self,
        question: str,
        top_k: Optional[int] = None,
        stream: bool = False,
        enable_query_rewrite: Optional[bool] = None,
        enable_rerank: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Поиск релевантных чанков → инжекция контекста → запрос к LLM."""
        k = top_k if top_k is not None else self.top_k_after
        use_rewrite = (
            self.enable_query_rewrite
            if enable_query_rewrite is None
            else enable_query_rewrite
        )
        use_rerank = (
            self.enable_rerank if enable_rerank is None else enable_rerank
        )

        original_query = question
        rewritten_query = question
        if use_rewrite:
            rewritten_query = self.query_rewriter.rewrite(question)

        results: List[SearchResult] = rag_search(
            query=rewritten_query,
            top_k=k,
            strategy=self.strategy,
            index_dir=self.index_dir,
            top_k_before=self.top_k_before,
            top_k_after=k,
            similarity_threshold=self.similarity_threshold,
            reranker=self.reranker if use_rerank else None,
            embedder=self._embedder,
        )

        context_parts: List[str] = []
        sources_set: set[str] = set()

        for r in results:
            source = r.chunk.metadata.get("source", "unknown")
            section = r.chunk.metadata.get("section", "")
            sources_set.add(source)

            header = f"[{source}]"
            if section:
                header += f" / {section}"
            context_parts.append(f"{header}\n{r.chunk.text}")

        context = "\n\n---\n\n".join(context_parts)
        sources_str = ", ".join(sorted(sources_set))

        system_prompt = _SYSTEM_WITH_RAG_TMPL.format(
            context=context,
            sources=sources_str,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        t0 = time.monotonic()
        if stream:
            answer = self._stream_llm(messages)
            elapsed_ms = (time.monotonic() - t0) * 1000
            token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        else:
            response = self._call_llm(messages)
            elapsed_ms = (time.monotonic() - t0) * 1000
            answer = response.choices[0].message.content or ""
            token_usage = self._extract_usage(response)

        if self.verbose and not stream:
            print(
                f"  [rag] chunks={len(results)} sources={len(sources_set)} "
                f"tokens={token_usage['total_tokens']} "
                f"elapsed={elapsed_ms:.0f}ms"
            )

        return {
            "answer": answer,
            "sources": sorted(sources_set),
            "chunks_used": len(results),
            "token_usage": token_usage,
            "elapsed_ms": round(elapsed_ms, 1),
            "mode": "rag",
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "rewrite_applied": use_rewrite and rewritten_query != original_query,
            "rerank_applied": use_rerank,
            "top_k_before": self.top_k_before,
            "top_k_after": k,
            "similarity_threshold": self.similarity_threshold,
        }

    def ask_with_citations(
        self,
        question: str,
        top_k: Optional[int] = None,
        stream: bool = False,
        enable_query_rewrite: Optional[bool] = None,
        enable_rerank: Optional[bool] = None,
        confidence_threshold: Optional[float] = None,
    ) -> CitedAnswer:
        """RAG с обязательными inline-цитатами, источниками и верификацией."""
        k = top_k if top_k is not None else 10  # More context for better answers
        use_rewrite = (
            self.enable_query_rewrite
            if enable_query_rewrite is None
            else enable_query_rewrite
        )
        use_rerank = (
            self.enable_rerank if enable_rerank is None else enable_rerank
        )
        conf_thresh = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        original_query = question
        rewritten_query = question
        if use_rewrite:
            rewritten_query = self.query_rewriter.rewrite(question)

        results: List[SearchResult] = rag_search(
            query=rewritten_query,
            top_k=k,
            strategy=self.strategy,
            index_dir=self.index_dir,
            top_k_before=self.top_k_before,
            top_k_after=k,
            similarity_threshold=self.similarity_threshold,
            reranker=self.reranker if use_rerank else None,
            embedder=self._embedder,
        )

        # Anti-hallucination guard
        if not results:
            if self.verbose:
                print("  [citations] no results --> uncertain")
            uncertain_msg = self._build_uncertain_message(question, 0.0)
            return CitedAnswer(
                answer=uncertain_msg,
                is_uncertain=True,
                raw_response=uncertain_msg,
                confidence=0.0,
            )

        max_score = max(r.score for r in results)
        if max_score < conf_thresh:
            if self.verbose:
                print(
                    f"  [citations] max_score={max_score:.4f} "
                    f"< {conf_thresh} --> uncertain"
                )
            uncertain_msg = self._build_uncertain_message(question, max_score)
            return CitedAnswer(
                answer=uncertain_msg,
                is_uncertain=True,
                raw_response=uncertain_msg,
                confidence=max_score,
            )

        context, source_refs = self._build_numbered_context(results)
        system_prompt = _SYSTEM_WITH_RAG_CITED_TMPL.format(context=context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        t0 = time.monotonic()
        if stream:
            raw_response = self._stream_llm(messages, temperature=0.2)
        else:
            response = self._call_llm(messages, temperature=0.2)
            raw_response = response.choices[0].message.content or ""

        elapsed_ms = (time.monotonic() - t0) * 1000

        if self.verbose:
            print(
                f"  [citations] chunks={len(results)} "
                f"max_score={max_score:.4f} "
                f"elapsed={elapsed_ms:.0f}ms"
            )

        cited = parse_cited_response(raw_response, results, verbose=self.verbose)

        # If LLM didn't produce ИСТОЧНИКИ block, auto-extract sources from search results
        if not cited.sources:
            cited.sources = self._extract_auto_sources(results)
            if self.verbose:
                print(f"  [parser] auto-extracted {len(cited.sources)} sources from search results")

        # If LLM didn't provide quotes, extract them automatically
        if not cited.quotes and not cited.is_uncertain:
            cited.quotes = self._extract_auto_quotes(results)
            if self.verbose:
                print(
                    f"  [citations] auto-extracted {len(cited.quotes)} quotes"
                )

        if self.verbose:
            status = "uncertain" if cited.is_uncertain else "confident"
            print(
                f"  [citations] {status} "
                f"sources={len(cited.sources)} "
                f"quotes={len(cited.quotes)}"
            )

        return cited

    def _extract_auto_sources(
        self,
        results: List[SearchResult],
        max_sources: int = 5,
    ) -> List[SourceRef]:
        """
        Auto-extract SourceRef from SearchResult when LLM doesn't output ИСТОЧНИКИ block.
        This ensures sources are always available for the RAG pipeline.
        """
        sources: List[SourceRef] = []
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

        for i, r in enumerate(sorted_results[:max_sources], start=1):
            source_name = r.chunk.metadata.get('source') or "Unknown"
            section = r.chunk.metadata.get('section', '')
            chunk_id = r.chunk.metadata.get("chunk_id", "")

            sources.append(SourceRef(
                index=i,
                source=source_name,
                section=section,
                chunk_id=chunk_id,
                score=getattr(r, 'score', 0.0),
            ))

        return sources

    def _extract_auto_quotes(
        self,
        results: List[SearchResult],
        num_quotes: int = 3,
    ) -> List[Quote]:
        """Extract key sentences from top chunks as fallback quotes."""
        quotes: List[Quote] = []
        # Take top chunks by score
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        sentences_seen: set[str] = set()
        
        for r in sorted_results[:num_quotes]:
            # Split chunk into sentences (simple heuristic: by periods)
            chunk_text = r.chunk.text.replace('\n', ' ')
            sentences = [s.strip() for s in chunk_text.split('.') if len(s.strip()) > 20]
            
            for sent in sentences:
                sent_full = sent + '.'
                sent_norm = normalize_text(sent_full)
                if sent_norm not in sentences_seen and sent_norm:
                    sentences_seen.add(sent_norm)
                    quotes.append(Quote(
                        index=r.rank if hasattr(r, 'rank') else 0,
                        text=sent_full.strip(),
                        verified=True,  # Extracted from actual chunk, so verified
                    ))
                    break  # One quote per chunk
            
            if len(quotes) >= num_quotes:
                break
        
        return quotes

    def compare(self, question: str, stream: bool = False) -> Dict[str, Any]:
        """Запускает оба режима и возвращает объединённый результат."""
        return {
            "question": question,
            "no_rag": self.ask_without_rag(question, stream=stream),
            "rag": self.ask_with_rag(question, stream=stream),
        }

    def compare_modes(self, question: str) -> Dict[str, Any]:
        """Сравнение 4 режимов retrieval-пайплайна."""
        return {
            "question": question,
            "baseline": self.ask_with_rag(
                question, enable_query_rewrite=False, enable_rerank=False
            ),
            "rewrite_only": self.ask_with_rag(
                question, enable_query_rewrite=True, enable_rerank=False
            ),
            "rerank_only": self.ask_with_rag(
                question, enable_query_rewrite=False, enable_rerank=True
            ),
            "combined": self.ask_with_rag(
                question, enable_query_rewrite=True, enable_rerank=True
            ),
        }

    # ── Внутренние методы ─────────────────────────────────────────────────────

    def _prepare_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prepare messages for the LLM. Disables thinking for Ollama qwen3."""
        if self.provider == "ollama" and "qwen3" in self.model:
            messages = [m.copy() for m in messages]
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += " /nothink"
        return messages

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> Any:
        """Call LLM with retry logic."""
        client = self._get_client()
        temp = temperature if temperature is not None else self.temperature
        messages = self._prepare_messages(messages)
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1200,
        }
        if self.model not in ("gpt-5-nano", "gpt-5.4", "gpt-5.4-mini", "gpt-5"):
            params["temperature"] = temp

        for attempt in range(3):
            try:
                return client.chat.completions.create(**params)
            except Exception as exc:
                if attempt < 2:
                    delay = min(2 ** attempt + random.uniform(0, 1), 30)
                    if self.verbose:
                        print(f"  API error: {exc}. Retry in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise

    def _stream_llm(
        self,
        messages: List[Dict[str, str]],
        indent: str = "  ",
        temperature: Optional[float] = None,
    ) -> str:
        """Stream LLM response, rendering <thinking> blocks specially."""
        THINK_OPEN = "<thinking>"
        THINK_CLOSE = "</thinking>"
        LONGEST_TAG = max(len(THINK_OPEN), len(THINK_CLOSE))

        full_text = ""
        answer_text = ""
        in_think = False
        buf = ""

        sys.stdout.write(indent)
        sys.stdout.flush()

        client = self._get_client()
        temp = temperature if temperature is not None else self.temperature
        messages = self._prepare_messages(messages)
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1200,
            "stream": True,
        }
        if self.model not in ("gpt-5-nano", "gpt-5.4", "gpt-5.4-mini", "gpt-5"):
            params["temperature"] = temp

        stream_resp = client.chat.completions.create(**params)

        for chunk in stream_resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            token = (delta.content or "") if delta else ""
            if not token:
                continue

            full_text += token
            buf += token

            while buf:
                if not in_think:
                    idx = buf.find(THINK_OPEN)
                    if idx != -1:
                        before = buf[:idx]
                        if before:
                            sys.stdout.write(before.replace("\n", f"\n{indent}"))
                            answer_text += before
                        sys.stdout.write(
                            f"\n{indent}\033[33mThinking:\033[0m"
                            f"\n{indent}\033[2m"
                        )
                        sys.stdout.flush()
                        buf = buf[idx + len(THINK_OPEN):]
                        in_think = True
                    else:
                        safe = max(0, len(buf) - LONGEST_TAG + 1)
                        if safe > 0:
                            chunk_out = buf[:safe]
                            sys.stdout.write(
                                chunk_out.replace("\n", f"\n{indent}")
                            )
                            sys.stdout.flush()
                            answer_text += chunk_out
                            buf = buf[safe:]
                        break
                else:
                    idx = buf.find(THINK_CLOSE)
                    if idx != -1:
                        think_chunk = buf[:idx]
                        if think_chunk:
                            sys.stdout.write(
                                think_chunk.replace("\n", f"\n{indent}")
                            )
                        sys.stdout.write(
                            f"\033[0m\n{indent}"
                            f"{'─' * 50}\n{indent}"
                        )
                        sys.stdout.flush()
                        buf = buf[idx + len(THINK_CLOSE):]
                        in_think = False
                    else:
                        safe = max(0, len(buf) - LONGEST_TAG + 1)
                        if safe > 0:
                            sys.stdout.write(
                                buf[:safe].replace("\n", f"\n{indent}")
                            )
                            sys.stdout.flush()
                            buf = buf[safe:]
                        break

        if buf:
            sys.stdout.write(buf.replace("\n", f"\n{indent}"))
            if not in_think:
                answer_text += buf
        if in_think:
            sys.stdout.write("\033[0m")
        sys.stdout.write("\n")
        sys.stdout.flush()

        return answer_text.strip()

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage:
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="RagAgent")
    p.add_argument("question", help="Question for the agent")
    p.add_argument(
        "--strategy",
        default="structure",
        choices=["fixed_size", "structure"],
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--top-k-before", type=int, default=DEFAULT_TOP_K_BEFORE
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    p.add_argument("--disable-rewrite", action="store_true")
    p.add_argument("--disable-rerank", action="store_true")
    p.add_argument(
        "--citations",
        action="store_true",
        help="Enable citation mode",
    )
    args = p.parse_args()

    agent = RagAgent(
        strategy=args.strategy,
        top_k=args.top_k,
        top_k_before=args.top_k_before,
        similarity_threshold=args.threshold,
        confidence_threshold=args.confidence,
        enable_query_rewrite=not args.disable_rewrite,
        enable_rerank=not args.disable_rerank,
        verbose=True,
    )

    if args.citations:
        result = agent.ask_with_citations(args.question, stream=True)
        tag = "!" if result.is_uncertain else "OK"
        print(
            f"\n  [{tag}] sources={len(result.sources)} "
            f"quotes={len(result.quotes)} uncertain={result.is_uncertain}"
        )
    else:
        result = agent.compare(args.question, stream=True)
        print("\n" + "=" * 70)
        print(f"  QUESTION: {result['question']}")
        print("=" * 70)
        rag_info = result["rag"]
        print(
            f"\n  Chunks: {rag_info['chunks_used']}, "
            f"Sources: {', '.join(rag_info['sources'])}"
        )


if __name__ == "__main__":
    main()
