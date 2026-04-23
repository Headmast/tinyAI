"""
Поиск по RAG-индексу.

Embed query → FAISS search → fetch metadata → return results.

CLI:
    python -m rag.search "Как работает MCP Agent?" --top-k 5 --strategy structure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import Chunk, SearchResult
from rag.embedder import Embedder, OpenAIEmbedder
from rag.index_store import FAISSIndexStore, get_chunks_by_faiss_ids
from rag.reranker import RerankerProtocol

DEFAULT_INDEX_DIR = Path(__file__).parent.parent / "rag_data"


def search(
    query: str,
    top_k: int = 5,
    strategy: Optional[str] = None,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    top_k_before: Optional[int] = None,
    top_k_after: Optional[int] = None,
    similarity_threshold: float = 0.0,
    reranker: Optional[RerankerProtocol] = None,
    embedder: Optional[Embedder] = None,
) -> List[SearchResult]:
    """
    Поиск релевантных чанков по текстовому запросу.

    Args:
        query: текст запроса
        top_k: количество результатов (legacy-параметр, эквивалент top_k_after)
        strategy: 'fixed_size' или 'structure' (если None — оба, нужен конкретный)
        index_dir: директория с индексом
        top_k_before: кол-во кандидатов до этапа rerank/filter
        top_k_after: кол-во результатов после rerank/filter
        similarity_threshold: порог similarity для отсечения нерелевантных результатов
        reranker: второй этап ранжирования (опционально)
    """
    final_k = top_k_after if top_k_after is not None else top_k
    initial_k = top_k_before if top_k_before is not None else final_k

    if initial_k <= 0:
        raise ValueError("top_k_before должен быть > 0")
    if final_k <= 0:
        raise ValueError("top_k_after/top_k должен быть > 0")

    index_path = Path(index_dir)
    store = FAISSIndexStore(index_dir=index_path)
    if embedder is None:
        embedder = OpenAIEmbedder()

    # Определяем стратегии для поиска
    if strategy:
        strategies = [strategy]
    else:
        strategies = []
        for s in ["fixed_size", "structure"]:
            if (index_path / f"index_{s}.faiss").exists():
                strategies.append(s)

    if not strategies:
        raise FileNotFoundError(f"Индексы не найдены в {index_path}")

    # Embed query
    query_embedding = embedder.embed_query(query)

    all_results: List[SearchResult] = []

    for strat in strategies:
        scores, indices = store.search(query_embedding, strategy=strat, top_k=initial_k)

        # Получаем FAISS id → metadata из SQLite
        faiss_ids = [int(idx) for idx in indices[0] if idx >= 0]
        if not faiss_ids:
            continue

        chunk_rows = get_chunks_by_faiss_ids(
            store.db_path, faiss_ids, strategy=strat
        )

        # Маппинг faiss_id → row для сохранения порядка
        row_map = {row["faiss_id"]: row for row in chunk_rows}

        for rank_idx, (score, faiss_id) in enumerate(
            zip(scores[0], indices[0])
        ):
            faiss_id = int(faiss_id)
            if faiss_id < 0 or faiss_id not in row_map:
                continue

            row = row_map[faiss_id]
            chunk = Chunk(
                text=row["text"],
                metadata={
                    "chunk_id": row["chunk_id"],
                    "source": row["source"],
                    "title": row["title"],
                    "section": row["section"],
                    "strategy": row["strategy"],
                    "token_count": row["token_count"],
                    "char_count": row["char_count"],
                    "chunk_index": row["chunk_index"],
                },
            )
            all_results.append(SearchResult(
                chunk=chunk,
                score=float(score),
                rank=rank_idx + 1,
            ))

    # Сортируем по score (убывание) и пере-нумеруем
    all_results.sort(key=lambda r: r.score, reverse=True)

    # Общий top-k до второго этапа (после merge по стратегиям).
    all_results = all_results[:initial_k]

    if similarity_threshold > 0:
        all_results = [r for r in all_results if r.score >= similarity_threshold]

    if reranker and all_results:
        all_results = reranker.rerank(query=query, candidates=all_results)

    all_results = all_results[:final_k]
    for i, r in enumerate(all_results):
        r.rank = i + 1

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Search")
    parser.add_argument("query", help="Поисковый запрос")
    parser.add_argument("--top-k", type=int, default=5, help="Кол-во результатов")
    parser.add_argument(
        "--strategy",
        choices=["fixed_size", "structure"],
        default=None,
        help="Стратегия (по умолчанию: все доступные)",
    )
    parser.add_argument("--index-dir", default="rag_data/")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    index_dir = project_root / args.index_dir

    results = search(
        query=args.query,
        top_k=args.top_k,
        strategy=args.strategy,
        index_dir=index_dir,
    )

    if not results:
        print("Результатов не найдено.")
        return

    print(f"\n🔍 Запрос: {args.query}")
    print(f"{'='*60}")

    for r in results:
        m = r.chunk.metadata
        print(f"\n#{r.rank} [score: {r.score:.4f}] [{m['strategy']}]")
        print(f"  📁 {m['source']}")
        print(f"  📑 Section: {m['section']}")
        print(f"  🔢 Tokens: {m['token_count']}")
        preview = r.chunk.text[:200].replace("\n", " ")
        print(f"  📝 {preview}...")
        print(f"  {'─'*50}")


if __name__ == "__main__":
    main()
