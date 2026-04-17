"""
Быстрый бенчмарк поиска RAG — проверка релевантности без LLM-запросов.

Тестирует:
1. Находит ли RAG релевантные чанки на внутренние вопросы
2. Возвращает ли low-score/пустые результаты на внешние вопросы
3. Статистику по каждой категории

Запуск:
    python demos/benchmark_rag_retrieval.py
    python demos/benchmark_rag_retrieval.py --verbose
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import SearchResult
from rag.search import search as rag_search

SEP = "═" * 80
SUB = "─" * 80

# 10 вопросов с ожиданием результатов поиска
QUESTIONS: List[Dict[str, Any]] = [
    # --- Внутренние вопросы: ожидаем > 0 результатов с score > threshold ---
    {"id": 1, "question": "Как работает RAG в проекте TinyAI?", "category": "rag", "expected_hits": True},
    {"id": 2, "question": "Какие стратегии поиска поддерживает RagAgent?", "category": "rag", "expected_hits": True},
    {"id": 3, "question": "Что такое протокол MCP и как он используется в TinyAI?", "category": "mcp", "expected_hits": True},
    {"id": 4, "question": "Как устроен журналист-агент в TinyAI?", "category": "agent", "expected_hits": True},
    {"id": 5, "question": "Как работает memory_agent?", "category": "memory", "expected_hits": True},
    {"id": 6, "question": "Как работает scheduler в TinyAI?", "category": "scheduler", "expected_hits": True},
    # --- Внешние вопросы: ожидаем низкие скоры или 0 результатов ---
    {"id": 7, "question": "Какой сейчас курс биткоина?", "category": "external", "expected_hits": False},
    {"id": 8, "question": "Кто выиграл World Series 2025 по бейсболу?", "category": "external", "expected_hits": False},
    {"id": 9, "question": "Какой рецепт лучше для торта Наполеон?", "category": "external", "expected_hits": False},
    # --- Граничный ---
    {"id": 10, "question": "Расскажи про RAG и про MCP одновременно.", "category": "multi_topic", "expected_hits": True},
]


def print_header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_subheader(title: str) -> None:
    print(f"\n{SUB}")
    print(f"  {title}")
    print(SUB)


def run_retrieval_benchmark(
    strategy: str = "structure",
    top_k: int = 5,
    similarity_threshold: float = 0.30,
    confidence_threshold: float = 0.45,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Запустить retrieval-only тест без LLM."""
    results: List[Dict[str, Any]] = []

    for q in QUESTIONS:
        qnum = q["id"]
        question = q["question"]
        category = q["category"]
        expected_hits = q["expected_hits"]

        print(f"\n[{'='*38}] Вопрос {qnum}/10 [{category}]")
        print(f"  Вопрос: {question}")
        print(f"  Ожидание hits: {'Yes' if expected_hits else 'No'}")

        t0 = time.monotonic()
        try:
            search_results: List[SearchResult] = rag_search(
                query=question,
                top_k=top_k,
                strategy=strategy,
                top_k_before=top_k,
                top_k_after=top_k,
                similarity_threshold=similarity_threshold,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000

            has_hits = len(search_results) > 0
            max_score = max((r.score for r in search_results), default=0.0)
            is_confident = max_score >= confidence_threshold and has_hits

            match = "✓" if (has_hits == expected_hits) else "✗ MISMATCH"
            icon = "✅" if has_hits else "⚠️"

            print(f"  {icon} Результаты: {len(search_results)} чанков, max_score={max_score:.4f} ({match})")
            print(f"  Время поиска: {elapsed_ms:.0f}ms")

            if verbose and search_results:
                for r in search_results[:3]:
                    src = r.chunk.metadata.get("source", "unknown")
                    section = r.chunk.metadata.get("section", "")
                    print(f"    [{r.rank}] {src} / {section} score={r.score:.4f}")
                    preview = r.chunk.text[:120].replace("\n", " ")
                    print(f"       {preview}...")

            record = {
                "id": qnum,
                "question": question,
                "category": category,
                "expected_hits": expected_hits,
                "actual_hits": has_hits,
                "match": has_hits == expected_hits,
                "num_chunks": len(search_results),
                "max_score": round(max_score, 4),
                "is_confident": is_confident,
                "search_time_ms": round(elapsed_ms, 1),
                "scores": [round(r.score, 4) for r in search_results],
                "sources": [
                    {
                        "source": r.chunk.metadata.get("source", "unknown"),
                        "section": r.chunk.metadata.get("section", ""),
                        "score": round(r.score, 4),
                    }
                    for r in search_results[:5]
                ],
            }
            results.append(record)

        except Exception as exc:
            print(f"  ❌ ОШИБКА: {exc}")
            results.append({
                "id": qnum,
                "question": question,
                "category": category,
                "expected_hits": expected_hits,
                "actual_hits": False,
                "match": False,
                "error": str(exc),
                "num_chunks": 0,
                "max_score": 0.0,
                "is_confident": False,
            })

    return results


def print_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Напечатать итоги и вернуть статистику."""
    total = len(results)
    matches = sum(1 for r in results if r.get("match"))
    hit_questions = sum(1 for r in results if r.get("actual_hits"))
    no_hit_questions = sum(1 for r in results if not r.get("actual_hits"))

    avg_time = sum(r.get("search_time_ms", 0) for r in results) / max(total, 1)
    scores_with_hits = [r["max_score"] for r in results if r.get("max_score", 0) > 0]
    avg_score = sum(scores_with_hits) / max(len(scores_with_hits), 1)

    print_header("ИТОГИ RETRIEVAL BENCHMARK")
    print(f"\n  Всего вопросов:         {total}")
    print(f"  Совпадений с ожидаемым:  {matches}/{total} ({100*matches/total:.0f}%)")
    print(f"  Найдены хиты:            {hit_questions}")
    print(f"  Нет хитов:               {no_hit_questions}")
    print(f"  Среднее время поиска:    {avg_time:.0f}ms")
    print(f"  Средний max_score:       {avg_score:.4f}")

    # По категориям
    print_subheader("Результаты по категориям")
    categories: Dict[str, List[Dict]] = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, []).append(r)
    for cat, items in sorted(categories.items()):
        cat_matches = sum(1 for i in items if i.get("match"))
        cat_total = len(items)
        cat_pct = 100 * cat_matches / cat_total if cat_total else 0
        cat_scores = [i["max_score"] for i in items if i.get("max_score", 0) > 0]
        avg_cat = sum(cat_scores) / max(len(cat_scores), 1)
        print(f"  {cat:15s}: {cat_matches}/{cat_total} ({cat_pct:.0f}%)  avg_score={avg_cat:.4f}")

    # Таблица
    print_subheader("Таблица результатов")
    print(f"  {'№':>2} {'Expected':>9} {'Actual':>7} {'Match':>6} {'Chunks':>7} {'MaxScore':>9} {'Time(ms)':>9} Вопрос")
    print(f"  {'─'*2} {'─'*9} {'─'*7} {'─'*6} {'─'*7} {'─'*9} {'─'*9} {'─'*40}")
    for r in results:
        exp = "Yes" if r["expected_hits"] else "No"
        actual = "Yes" if r["actual_hits"] else "No"
        match_str = "✓" if r.get("match") else "✗"
        error = r.get("error", "")
        print(
            f"  {r['id']:>2} {exp:>9} {actual:>7} {match_str:>6} {r['num_chunks']:>7} {r['max_score']:>9.4f} {r.get('search_time_ms', '-'):>9} {r['question'][:40]}"
        )
        if error:
            print(f"     ERROR: {error}")

    return {
        "total": total,
        "matches": matches,
        "match_rate": matches / total if total else 0,
        "hit_questions": hit_questions,
        "no_hit_questions": no_hit_questions,
        "avg_search_time_ms": avg_time,
        "avg_max_score": avg_score,
    }


def save_results(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    config: Dict[str, Any],
) -> Path:
    """Сохранить результаты в JSON."""
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = logs_dir / f"benchmark_retrieval_{timestamp}.json"

    payload = {
        "benchmark": "RAG Retrieval (TASK24)",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "summary": summary,
        "questions": results,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n  Сохранено: {filepath}")
    return filepath


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Benchmark RAG Retrieval только retrieval (без LLM)")
    p.add_argument("--verbose", action="store_true", help="Показать детали чанков")
    p.add_argument("--strategy", default="structure", choices=["fixed_size", "structure"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--similarity-threshold", type=float, default=0.30)
    p.add_argument("--confidence-threshold", type=float, default=0.45, help="Threshold для confident")
    args = p.parse_args()

    print_header(f"Retrieval Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Strategy: {args.strategy}, top_k: {args.top_k}, threshold: {args.similarity_threshold}, confidence: {args.confidence_threshold}")

    config = {
        "strategy": args.strategy,
        "top_k": args.top_k,
        "similarity_threshold": args.similarity_threshold,
        "confidence_threshold": args.confidence_threshold,
    }

    results = run_retrieval_benchmark(
        strategy=args.strategy,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        confidence_threshold=args.confidence_threshold,
        verbose=args.verbose,
    )
    summary = print_summary(results)
    save_results(results, summary, config)


if __name__ == "__main__":
    main()
