"""
RagChatAgent — диалоговый агент с RAG-поиском, цитатами и памятью задачи.

Объединяет:
  - RagAgent (RAG-поиск с цитатами и источниками)
  - PersistentTaskMemory (состояние задачи: goal, clarifications, constraints)
  - Историю диалога

Каждый turn:
  1. Сохраняет ввод пользователя в память
  2. Автоматически извлекает goal/clarifications/constraints
  3. Выполняет RAG-поиск с цитатами
  4. Формирует ответ с учётом контекста задачи
  5. Возвращает ответ с источниками
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from rag import CitedAnswer, SearchResult, SourceRef, Quote
from rag.rag_agent import RagAgent
from rag.task_memory import PersistentTaskMemory
from rag.reranker import MathReranker
from rag.query_rewrite import QueryRewriter
from rag.search import search as rag_search

load_dotenv()

DEFAULT_MODEL = os.getenv("LLM_MODEL", "zai-org/GLM-4.7")
DEFAULT_BASE_URL = os.getenv("BASE_URL", "https://foundation-models.api.cloud.ru/v1")
DEFAULT_INDEX_DIR = Path(__file__).parent.parent / "rag_data"
DEFAULT_STORAGE_DIR = Path(__file__).parent.parent / "rag_data"
_DEFAULT_TASK_MEMORY = DEFAULT_STORAGE_DIR / "task_memory.json"
MAX_HISTORY_TURNS = 10

_GPT_MODELS = {"gpt-5.4", "gpt-5.4-mini", "gpt-5-nano", "gpt-5", "gpt-4o", "gpt-4o-mini"}
_GPT_BASE_URL = "https://api.openai.com/v1"

SYSTEM_PROMPT_TEMPLATE = """\
Ты — интеллектуальный ассистент, который помогает пользователям разобраться в сложных темах.
Ты отвечаешь на вопросы опираясь на предоставленный контекст из базы знаний, но говоришь живым, понятным языком.

### Правила:
1. Используй ТОЛЬКО информацию из контекста ниже.
2. Если информации недостаточно — скажи об этом прямо и честно.
3. Отвечай на языке вопроса (русский или английский).
4. Отвечай как знающий эксперт, но простым, понятным языком — не сухо и не как справочник.

### Формат ответа:
1. **Основной ответ** — дай полный, связный текст, объясни тему как человеку. Используй абзацы, списки, примеры где уместно.
2. **Цитаты из документов** — приведи 2-3 наиболее уместных цитаты прямо из текста контекста, чтобы подтвердить ответ. Цитаты оборачивай в знаки кавычек «» и указывай откуда они:
   > «цитата» — _документ_
3. **Источники** — в конце после разделителя "───" перечисли все документы, на которые опираешься, в формате:
   [N] Имя файла — Раздел

{memory_context_block}

{context}

Отвечай на вопрос пользователя, придерживаясь формата выше. Начни сразу с объяснения, без вступлений вроде "конечно" или "вот".
"""

CONVERSATION_PREFIX = """\
Ниже — история диалога (последние сообщения). Помни контекст задачи при ответе.

### Последние сообщения:
{history}

───
Новый вопрос пользователя: {user_input}
"""


class RagChatAgent:
    """
    Диалоговый агент с RAG, цитатами и памятью задачи.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        index_dir: Optional[str | Path] = None,
        memory_path: Optional[str | Path] = None,
        strategy: Optional[str] = "structure",
        top_k: int = 10,
        top_k_before: int = 15,
        similarity_threshold: float = 0.30,
        enable_query_rewrite: bool = True,
        enable_rerank: bool = True,
        fast_mode: bool = False,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.index_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
        self.strategy = strategy
        self.top_k = top_k
        self.top_k_before = top_k_before
        self.similarity_threshold = similarity_threshold
        self.enable_query_rewrite = enable_query_rewrite
        self.enable_rerank = enable_rerank
        self.fast_mode = fast_mode
        self.verbose = verbose

        # ── API client ──
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
        self._gpt_client: Optional[OpenAI] = None

        # ── RAG Agent (reused for search + citation building) ──
        self.rag_agent = RagAgent(
            model=model,
            index_dir=self.index_dir,
            strategy=strategy,
            top_k=top_k,
            top_k_before=top_k_before,
            similarity_threshold=similarity_threshold,
            enable_query_rewrite=enable_query_rewrite,
            enable_rerank=enable_rerank,
            verbose=verbose,
        )

        # ── Fast mode: math reranker + heuristic rewriter (no LLM) ──
        if self.fast_mode:
            self._math_reranker = MathReranker(
                cosine_weight=0.6,
                bm25_weight=0.4,
                verbose=verbose,
            )
            self._heuristic_rewriter = QueryRewriter(
                client=None,  # не нужен — будем использовать только эвристику
                enabled=False,  # отключаем LLM rewrite
                verbose=verbose,
            )

        # ── Task Memory ──
        mem_path = memory_path if memory_path else _DEFAULT_TASK_MEMORY
        self.memory = PersistentTaskMemory(
            storage_path=str(mem_path),
            max_history=MAX_HISTORY_TURNS * 2,
        )

        # ── Token tracking ──
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_elapsed_ms = 0.0

    # ── Client helpers ────────────────────────────────────────────────────────

    def _get_client(self) -> OpenAI:
        if self.model in _GPT_MODELS:
            if self._gpt_client is None:
                openai_key = os.getenv("OPENAI_API_KEY")
                if not openai_key:
                    raise EnvironmentError(
                        "OPENAI_API_KEY не найден для GPT-модели."
                    )
                self._gpt_client = OpenAI(
                    base_url=_GPT_BASE_URL,
                    api_key=openai_key,
                    timeout=120.0,
                )
            return self._gpt_client
        return self.client

    # ── RAG search ────────────────────────────────────────────────────────────

    def _search(self, query: str) -> CitedAnswer:
        """Запустить RAG-поиск с цитатами через RagAgent."""
        return self.rag_agent.ask_with_citations(
            question=query,
            top_k=self.top_k,
            enable_query_rewrite=self.enable_query_rewrite,
            enable_rerank=self.enable_rerank,
        )

    def _fast_search(self, query: str) -> CitedAnswer:
        """
        Fast-режим: поиск без LLM-вызовов.

        1. Heuristic query rewrite (regex, без LLM)
        2. Embed query (кэшируется)
        3. FAISS search
        4. MathReranker (BM25 + cosine, без LLM)
        5. Возвращает CitedAnswer с найденными чанками (без LLM-генерации)
        """
        # Heuristic rewrite
        rewritten = self._heuristic_rewriter._heuristic_rewrite(query)
        if self.verbose:
            print(f"  [fast] rewrite: {query[:50]}... → {rewritten[:50]}...")

        # Search with math reranker
        results: list[SearchResult] = rag_search(
            query=rewritten,
            top_k=self.top_k,
            strategy=self.strategy,
            index_dir=self.index_dir,
            top_k_before=self.top_k_before,
            top_k_after=self.top_k,
            similarity_threshold=self.similarity_threshold,
            reranker=self._math_reranker,
        )

        if not results:
            return CitedAnswer(
                answer="",
                is_uncertain=True,
                raw_response="",
                confidence=0.0,
            )

        max_score = max(r.score for r in results)

        # Строим source_refs и контекст
        sources: list[SourceRef] = []
        for i, r in enumerate(results, start=1):
            sources.append(SourceRef(
                index=i,
                source=r.chunk.metadata.get("source", "unknown"),
                section=r.chunk.metadata.get("section", ""),
                chunk_id=r.chunk.metadata.get("chunk_id", ""),
                score=round(r.score, 4),
            ))

        return CitedAnswer(
            answer="",  # не генерируем — будет один вызов LLM в chat_turn
            is_uncertain=max_score < 0.40,
            raw_response="",
            confidence=max_score,
            sources=sources,
            quotes=[],
            search_results=results,  # сохраняем для построения контекста
        )

    # ── Prompt building ───────────────────────────────────────────────────────

    def _build_context_from_cited_answer(self, cited: CitedAnswer) -> str:
        """Извлечь текстовый контекст из CitedAnswer для prompt."""
        parts: List[str] = []

        # Fast mode: используем raw search results с полным текстом чанков
        if cited.search_results:
            for i, r in enumerate(cited.search_results, start=1):
                source = r.chunk.metadata.get("source", "unknown")
                section = r.chunk.metadata.get("section", "")
                header = f"[{i}] Источник: {source}"
                if section:
                    header += f" | Раздел: {section}"
                header += f" (score: {r.score:.4f})"
                parts.append(f"{header}\n{r.chunk.text}")
            return "\n\n---\n\n".join(parts)

        # Обычный режим: источники + цитаты + ответ
        if hasattr(cited, "sources") and cited.sources:
            for src in cited.sources:
                header = f"[{src.index}] {src.source}"
                if hasattr(src, "section") and src.section:
                    header += f" / {src.section}"
                header += f" (score: {src.score:.4f})"
                parts.append(header)

            if cited.quotes:
                for i, q in enumerate(cited.quotes, start=1):
                    parts.append(f"  Цитата {i}: «{q.text}»")

        parts.append(f"\nОтвет ассистента:\n{cited.answer}")

        return "\n\n".join(parts)

    def _build_messages(
        self,
        user_input: str,
        cited_answer: CitedAnswer,
    ) -> List[Dict[str, str]]:
        """
        Построить messages[] для финального вызова LLM, включая:
        - system prompt с контекстом задачи и RAG-контекстом
        - историю диалога
        - текущий вопрос
        """
        # Memory context
        memory_block = self.memory.get_memory_context_block()

        # RAG context from cited answer
        rag_context = self._build_context_from_cited_answer(cited_answer)

        # System prompt
        system_content = SYSTEM_PROMPT_TEMPLATE.format(
            memory_context_block=memory_block,
            context=rag_context,
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]

        # History
        history_text = ""
        exchanges = self.memory.get_last_n_exchanges(MAX_HISTORY_TURNS)
        for ex in exchanges:
            prefix = "Пользователь" if ex["role"] == "user" else "Ассистент"
            history_text += f"• {prefix}: {ex['content']}\n\n"
        
        if history_text:
            # Add history as a user message (the assistant's own past answers)
            # This keeps conversation context flowing
            messages.append({
                "role": "user",
                "content": CONVERSATION_PREFIX.format(
                    history=history_text.strip(),
                    user_input=user_input,
                ),
            })
        else:
            messages.append({"role": "user", "content": user_input})

        return messages

    # ── Main entry point ──────────────────────────────────────────────────────

    def chat_turn(self, user_input: str) -> Dict[str, Any]:
        """
        Один поворот диалога.

        Шаги:
          1. Сохранить ввод пользователя
          2. Автоматически извлечь goal/clarifications/constraints
          3. RAG-поиск с цитатами
          4. Финальный вызов LLM с полным контекстом
          5. Сохранить ответ
          6. Вернуть результат
        """
        t0 = time.monotonic()

        # Step 1: Save user message
        turn_num = self.memory.add_user_message(user_input)

        # Step 2: Auto-extract task state
        self.memory.auto_extract_user_input(user_input)

        if self.verbose:
            mem = self.memory.dump()
            print(f"\n  📝 Turn #{turn_num}")
            print(f"  Цель: {mem['goal'] or '(не задана)'}")
            print(f"  Уточнения: {len(mem['clarifications'])}")
            print(f"  Ограничения: {len(mem['constraints'])}")

        # Step 3: RAG search with citations
        try:
            if self.fast_mode:
                cited = self._fast_search(user_input)
            else:
                cited = self._search(user_input)
            rag_status = "found" if not cited.is_uncertain else "uncertain"
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ RAG search error: {e}")
            cited = CitedAnswer(
                answer=f"Ошибка RAG-поиска: {e}",
                is_uncertain=True,
                raw_response="",
                confidence=0.0,
            )
            rag_status = "error"

        if self.verbose:
            print(f"  🔍 RAG: {rag_status}, score={cited.confidence:.4f}")
            print(f"  📚 Sources: {len(cited.sources)}")
            print(f"  💬 Quotes: {len(cited.quotes)}")

        # Step 4: Final LLM call with full context
        messages = self._build_messages(user_input, cited)
        final_answer = self._call_llm(messages)

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Step 5: Save assistant response
        self.memory.add_assistant_message(final_answer)

        # Token tracking
        self._total_elapsed_ms += elapsed_ms

        if self.verbose:
            print(f"  ⏱️  Total: {elapsed_ms:.0f}ms")

        # Step 6: Return result
        return {
            "answer": final_answer,
            "cited_answer": cited,
            "turn": turn_num,
            "goal": self.memory.goal,
            "sources": [
                {
                    "index": s.index,
                    "source": s.source,
                    "section": getattr(s, "section", ""),
                    "score": getattr(s, "score", 0.0),
                }
                for s in cited.sources
            ],
            "quotes": [
                {"text": q.text, "verified": q.verified}
                for q in cited.quotes
            ],
            "is_uncertain": cited.is_uncertain,
            "confidence": cited.confidence,
            "elapsed_ms": round(elapsed_ms, 1),
            "memory_summary": self.memory.get_summary(),
        }

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Вызвать LLM с retry."""
        client = self._get_client()
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 3000,
        }
        if self.model not in ("gpt-5-nano", "gpt-5.4", "gpt-5.4-mini", "gpt-5"):
            params["temperature"] = 0.4

        for attempt in range(3):
            try:
                response = client.chat.completions.create(**params)
                usage = getattr(response, "usage", None)
                if usage:
                    self._total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
                    self._total_completion_tokens += getattr(usage, "completion_tokens", 0)
                return response.choices[0].message.content or ""
            except Exception as exc:
                if attempt < 2:
                    if self.verbose:
                        print(f"  ⚠️  API error: {exc}. Retry in 2s...")
                    time.sleep(2)
                else:
                    raise
        return ""  # safety fallback

    # ── Utility methods ───────────────────────────────────────────────────────

    def set_goal(self, goal: str) -> None:
        """Явно установить цель диалога."""
        self.memory.set_goal(goal)

    def get_memory_state(self) -> Dict[str, Any]:
        """Получить текущее состояние памяти."""
        return self.memory.dump()

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику использования."""
        return {
            "turns": self.memory.turn_count,
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "total_elapsed_ms": round(self._total_elapsed_ms, 1),
            "goal_set": bool(self.memory.goal),
        }

    def reset_memory(self) -> None:
        """Сбросить память задачи."""
        self.memory.reset()
        if self.verbose:
            print("  🧹 Память задачи очищена")
