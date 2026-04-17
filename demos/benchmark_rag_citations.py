"""
Benchmark RAG с цитатами — запуск 10 вопросов с логами и сохранением результатов.

Запуск:
    python demos/benchmark_rag_citations.py
    python demos/benchmark_rag_citations.py --verbose
    python demos/benchmark_rag_citations.py --top-k 7 --threshold 0.35

Результат сохраняется в logs/benchmark_citations_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_agent import RagAgent, DEFAULT_CONFIDENCE_THRESHOLD

SEP = "═" * 80
SUB = "─" * 80

# 10 вопросов, покрывающих уверенные и неуверенные ответы
QUESTIONS: List[Dict[str, Any]] = [
    # --- Вопросы из документации (ожидание: confident) ---
    {
        "id": 1,
        "question": "Как работает RAG в проекте TinyAI?",
        "category": "rag",
        "expected": "confident",
        "note": "Внутренний вопрос по RAG-модулю.",
    },
    {
        "id": 2,
        "question": "Какие стратегии поиска поддерживает RagAgent?",
        "category": "rag",
        "expected": "confident",
        "note": "Прямой вопрос по стратегии поиска.",
    },
    {
        "id": 3,
        "question": "Что такое протокол MCP и как он используется в TinyAI?",
        "category": "mcp",
        "expected": "confident",
        "note": "Архитектурный вопрос по MCP.",
    },
    {
        "id": 4,
        "question": "Как устроен журналист-агент в TinyAI?",
        "category": "agent",
        "expected": "confident",
        "note": "Вопрос по архитектуре journalist_agent.",
    },
    {
        "id": 5,
        "question": "Как работает memory_agent?",
        "category": "memory",
        "expected": "confident",
        "note": "Вопрос по модулю памяти.",
    },
    {
        "id": 6,
        "question": "Как работает scheduler в TinyAI?",
        "category": "scheduler",
        "expected": "confident",
        "note": "Вопрос по планировщику задач.",
    },
    # --- Внешние вопросы (ожидание: uncertain) ---
    {
        "id": 7,
        "question": "Какой сейчас курс биткоина?",
        "category": "external",
        "expected": "uncertain",
        "note": "Внешний факт — не в документации TinyAI.",
    },
    {
        "id": 8,
        "question": "Кто выиграл World Series 2025 по бейсболу?",
        "category": "external",
        "expected": "uncertain",
        "note": "Спортивный результат — не в документации.",
    },
    {
        "id": 9,
        "question": "Какой рецепт лучше для торта 'Наполеон'?",
        "category": "external",
        "expected": "uncertain",
        "note": "Кулинарный вопрос — не в документации.",
    },
    # --- Граничный вопрос для проверки качества цитат ---
    {
        "id": 10,
        "question": "Расскажи про RAG и про MCP одновременно.",
        "category": "multi_topic",
        "expected": "confident",
        "note": "Две темы из документации — должен найти по обеим.",
    },
]


def print_header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_subheader(title: str) -> None:
    print(f"\n{SUB}")
    print(f"  {title}")
    print(SUB)


def run_benchmark(
    agent: RagAgent, questions: List[Dict[str, Any]], verbose: bool = False
) -> List[Dict[str, Any]]:
    """Запустить все вопросы и вернуть список результатов."""
    results: List[Dict[str, Any]] = []

    for i, q in enumerate(questions, 1):
        qnum = q["id"]
        question = q["question"]
        category = q["category"]
        expected = q["expected"]
        note = q["note"]

        print(f"\n[{'='*38}] Вопрос {qnum}/10 [{category}]")
        print(f"  Вопрос: {question}")
        print(f"  Ожидание: {expected} | Примечание: {note}")
        print()

        try:
            t0 = time.monotonic()
            result = agent.ask_with_citations(question, stream=False)
            elapsed_ms = (time.monotonic() - t0) * 1000

            status_icon = "⚠️" if result.is_uncertain else "✅"
            status_text = "uncertain" if result.is_uncertain else "confident"
            match = "✓" if (result.is_uncertain == (expected == "uncertain")) else "✗ MISMATCH"

            print(f"  {status_icon} Результат: {status_text} ({match})")
            print(f"  Скоры чанков: {[f'{r.score:.3f}' for r in result.sources]}")

            # Печать первых 200 символов ответа
            preview = (result.answer or "")[:200].replace("\n", " ")
            print(f"  Ответ: {preview}{'...' if len(result.answer or '') > 200 else ''}")

            record = {
                "id": qnum,
                "question": question,
                "category": category,
                "note": note,
                "expected": expected,
                "actual": "uncertain" if result.is_uncertain else "confident",
                "match": result.is_uncertain == (expected == "uncertain"),
                "is_uncertain": result.is_uncertain,
                "confidence": result.confidence,
                "elapsed_ms": round(elapsed_ms, 1),
                "sources": [
                    {
                        "index": s.index,
                        "source": s.source,
                        "section": s.section,
                        "score": s.score,
                    }
                    for s in result.sources[:5]
                ],
                "quotes": [
                    {
                        "text": qt.text[:80],
                        "verified": qt.verified,
                    }
                    for qt in result.quotes[:5]
                ],
                "full_answer": result.answer,
                "raw_response": result.raw_response,
            }
            results.append(record)

            if verbose:
                print(f"\n  [RAW] {result.raw_response[:300]}")

        except Exception as exc:
            print(f"  ❌ ОШИБКА: {exc}")
            record = {
                "id": qnum,
                "question": question,
                "category": category,
                "expected": expected,
                "actual": "error",
                "match": False,
                "error": str(exc),
                "is_uncertain": None,
                "sources": [],
                "quotes": [],
            }
            results.append(record)

    return results


def print_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Напечатать итоговую сводку и вернуть статистику."""
    total = len(results)
    confident = sum(1 for r in results if r.get("actual") == "confident")
    uncertain = sum(1 for r in results if r.get("actual") == "uncertain")
    errors = sum(1 for r in results if r.get("actual") == "error")
    matches = sum(1 for r in results if r.get("match"))

    avg_time = sum(r.get("elapsed_ms", 0) for r in results) / max(total, 1)
    avg_confidence = sum(
        r.get("confidence", 0)
        for r in results
        if r.get("confidence") is not None
    ) / max(confident, 1)

    sources_count = [len(r.get("sources", [])) for r in results]
    quotes_count = [len(r.get("quotes", [])) for r in results]

    print_header("ИТОГИ BENCHMARK")

    print(f"\n  Всего вопросов:      {total}")
    print(f"  Confident ответов:   {confident}")
    print(f"  Uncertain ответов:   {uncertain}")
    print(f"  Ошибок:              {errors}")
    print(f"  Совпадений с ожидаемым: {matches}/{total} ({100*matches/total:.0f}%)")
    print(f"  Среднее время:       {avg_time:.0f}ms")
    print(f"  Средняя уверенность:  {avg_confidence:.3f} (confident ответы)")
    print(f"  Источников/вопрос:   ~{sum(sources_count)/max(total,1):.1f}")
    print(f"  Цитат/вопрос:        ~{sum(quotes_count)/max(total,1):.1f}")

    # По категориям
    print_subheader("Результаты по категориям")
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, []).append(r)
    for cat, items in sorted(categories.items()):
        cat_matches = sum(1 for i in items if i.get("match"))
        cat_total = len(items)
        cat_pct = 100 * cat_matches / cat_total if cat_total else 0
        print(f"  {cat:15s}: {cat_matches}/{cat_total} ({cat_pct:.0f}%)")

    # Таблица результатов
    print_subheader("Таблица результатов")
    print(f"  {'№':>2} {'Status':>10} {'Exp':>10} {'Match':>6} {'Conf':>6} {'Sources':>8} {'Quotes':>7} {'Time(ms)':>9} Вопрос")
    print(f"  {'─'*2} {'─'*10} {'─'*10} {'─'*6} {'─'*6} {'─'*8} {'─'*7} {'─'*9} {'─'*40}")
    for r in results:
        status = "uncertain" if r.get("is_uncertain") else r.get("actual", "error")
        nsources = len(r.get("sources", []))
        nquotes = len(r.get("quotes", []))
        match_str = "✓" if r.get("match") else "✗"
        conf_val = f"{r.get('confidence', '-'):.2f}" if r.get('confidence') is not None else "N/A"
        print(
            f"  {r['id']:>2} {status:>10} {r['expected']:>10} {match_str:>6} {conf_val:>6} {nsources:>8} {nquotes:>7} {r.get('elapsed_ms', '-'):>9} {r['question'][:40]}"
        )

    return {
        "total": total,
        "confident": confident,
        "uncertain": uncertain,
        "errors": errors,
        "matches": matches,
        "match_rate": matches / total if total else 0,
        "avg_time_ms": avg_time,
        "avg_confidence": avg_confidence,
        "avg_sources_per_question": sum(sources_count) / max(total, 1),
        "avg_quotes_per_question": sum(quotes_count) / max(total, 1),
    }


def save_results(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    config: Dict[str, Any],
) -> Path:
    """Сохранить результаты в JSON-файл в logs/."""
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_citations_{timestamp}.json"
    filepath = logs_dir / filename

    payload = {
        "benchmark": "RAG Citations (TASK24)",
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

    p = argparse.ArgumentParser(description="Benchmark RAG Citations")
    p.add_argument("--verbose", action="store_true", help="Show raw LLM outputs")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--threshold", type=float, default=0.30, help="Similarity threshold"
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold",
    )
    args = p.parse_args()

    print_header(f"Benchmark RAG с цитатами — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    agent = RagAgent(
        strategy="structure",
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        confidence_threshold=args.confidence,
        enable_query_rewrite=True,
        enable_rerank=True,
        verbose=args.verbose,
    )

    config = {
        "strategy": "structure",
        "top_k": args.top_k,
        "similarity_threshold": args.threshold,
        "confidence_threshold": args.confidence,
        "enable_query_rewrite": True,
        "enable_rerank": True,
        "question_count": len(QUESTIONS),
    }

    results = run_benchmark(agent, QUESTIONS, verbose=args.verbose)
    summary = print_summary(results)
    save_results(results, summary, config)

    print(f"\n  Benchmark завершён. Результаты в {Path(__file__).parent.parent / 'logs' / '*.json'}")


if __name__ == "__main__":
    main()
