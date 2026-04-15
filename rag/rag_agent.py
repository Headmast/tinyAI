"""
RagAgent — агент с двумя режимами ответа: без RAG и с RAG.

Режимы:
    ask_without_rag  — прямой запрос к LLM без дополнительного контекста
    ask_with_rag     — поиск релевантных чанков → инжекция контекста → запрос к LLM
    compare          — запускает оба режима, возвращает объединённый результат

CLI:
    python -m rag.rag_agent "Как устроен MCP Agent?" --strategy structure
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.search import search as rag_search
from rag import SearchResult

load_dotenv()

DEFAULT_MODEL = os.getenv("LLM_MODEL", "zai-org/GLM-4.7")
DEFAULT_BASE_URL = os.getenv("BASE_URL", "https://foundation-models.api.cloud.ru/v1")
DEFAULT_INDEX_DIR = Path(__file__).parent.parent / "rag_data"

_SYSTEM_NO_RAG = (
    "Ты — ассистент, помогающий разобраться в устройстве проекта TinyAI. "
    "Отвечай точно и по существу."
)

_SYSTEM_WITH_RAG_TMPL = (
    "Ты — ассистент, помогающий разобраться в устройстве проекта TinyAI. "
    "Используй ТОЛЬКО следующий контекст для ответа на вопрос пользователя. "
    "Не придумывай информацию, которой нет в контексте.\n\n"
    "=== КОНТЕКСТ ===\n"
    "{context}\n"
    "=== КОНЕЦ КОНТЕКСТА ===\n\n"
    "Источники: {sources}"
)


class RagAgent:
    """
    Агент с двумя режимами: без RAG и с RAG.

    Args:
        model:       название LLM-модели
        index_dir:   директория с FAISS-индексом и SQLite-базой
        strategy:    стратегия RAG-поиска ('fixed_size', 'structure' или None — оба)
        top_k:       количество чанков для RAG-контекста
        temperature: температура генерации
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        strategy: Optional[str] = "structure",
        top_k: int = 5,
        temperature: float = 0.7,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.index_dir = Path(index_dir)
        self.strategy = strategy
        self.top_k = top_k
        self.temperature = temperature
        self.verbose = verbose

        api_key = os.getenv("CLOUD_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "API-ключ не найден. Задайте CLOUD_API_KEY или OPENAI_API_KEY в .env"
            )

        self.client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=api_key)

    # ── Публичный интерфейс ───────────────────────────────────────────────────

    def ask_without_rag(self, question: str) -> Dict[str, Any]:
        """
        Прямой запрос к LLM без поиска по документам.

        Returns:
            dict с ключами: answer, token_usage, elapsed_ms, mode
        """
        messages = [
            {"role": "system", "content": _SYSTEM_NO_RAG},
            {"role": "user", "content": question},
        ]

        t0 = time.monotonic()
        response = self._call_llm(messages)
        elapsed_ms = (time.monotonic() - t0) * 1000

        answer = response.choices[0].message.content or ""
        token_usage = self._extract_usage(response)

        if self.verbose:
            print(f"  [no_rag] tokens={token_usage['total_tokens']} "
                  f"elapsed={elapsed_ms:.0f}ms")

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
    ) -> Dict[str, Any]:
        """
        Поиск релевантных чанков → инжекция контекста → запрос к LLM.

        Returns:
            dict с ключами: answer, sources, chunks_used, token_usage, elapsed_ms, mode
        """
        k = top_k or self.top_k

        # Поиск
        results: List[SearchResult] = rag_search(
            query=question,
            top_k=k,
            strategy=self.strategy,
            index_dir=self.index_dir,
        )

        # Строим контекст
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
        response = self._call_llm(messages)
        elapsed_ms = (time.monotonic() - t0) * 1000

        answer = response.choices[0].message.content or ""
        token_usage = self._extract_usage(response)

        if self.verbose:
            print(f"  [rag] chunks={len(results)} sources={len(sources_set)} "
                  f"tokens={token_usage['total_tokens']} elapsed={elapsed_ms:.0f}ms")

        return {
            "answer": answer,
            "sources": sorted(sources_set),
            "chunks_used": len(results),
            "token_usage": token_usage,
            "elapsed_ms": round(elapsed_ms, 1),
            "mode": "rag",
        }

    def compare(self, question: str) -> Dict[str, Any]:
        """
        Запускает оба режима и возвращает объединённый результат.

        Returns:
            dict с ключами: question, no_rag, rag
        """
        return {
            "question": question,
            "no_rag": self.ask_without_rag(question),
            "rag": self.ask_with_rag(question),
        }

    # ── Внутренние методы ─────────────────────────────────────────────────────

    def _call_llm(self, messages: List[Dict[str, str]]) -> Any:
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": 2000,
            "temperature": self.temperature,
        }

        for attempt in range(3):
            try:
                return self.client.chat.completions.create(**params)
            except Exception as e:
                if attempt < 2:
                    if self.verbose:
                        print(f"  ⚠️  API ошибка: {e}. Повтор через 2с...")
                    time.sleep(2)
                else:
                    raise

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage:
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RagAgent: сравнение ответов с RAG и без")
    parser.add_argument("question", help="Вопрос к агенту")
    parser.add_argument("--strategy", default="structure",
                        choices=["fixed_size", "structure"],
                        help="Стратегия RAG-поиска (default: structure)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Количество чанков для контекста (default: 5)")
    args = parser.parse_args()

    agent = RagAgent(strategy=args.strategy, top_k=args.top_k, verbose=True)
    result = agent.compare(args.question)

    print("\n" + "═" * 70)
    print(f"  ВОПРОС: {result['question']}")
    print("═" * 70)
    print("\n▶ БЕЗ RAG:")
    print(result["no_rag"]["answer"])
    print(f"  (tokens: {result['no_rag']['token_usage']['total_tokens']}, "
          f"elapsed: {result['no_rag']['elapsed_ms']:.0f}ms)")

    print("\n▶ С RAG:")
    print(result["rag"]["answer"])
    print(f"  (tokens: {result['rag']['token_usage']['total_tokens']}, "
          f"elapsed: {result['rag']['elapsed_ms']:.0f}ms, "
          f"chunks: {result['rag']['chunks_used']}, "
          f"sources: {', '.join(result['rag']['sources'])})")


if __name__ == "__main__":
    main()
