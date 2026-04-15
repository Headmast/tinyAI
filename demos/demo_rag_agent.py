"""
Демо RagAgent: сравнение ответов с RAG и без RAG.

По умолчанию прогоняет 3 показательных вопроса (Q1, Q3, Q10).
С флагом --full запускает полный бенчмарк из 10 вопросов.

Запуск:
    python demos/demo_rag_agent.py
    python demos/demo_rag_agent.py --full
    python demos/demo_rag_agent.py --strategy fixed_size --top-k 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_agent import RagAgent
from rag.benchmark import CONTROL_QUESTIONS, run_benchmark
from rag.answer_comparison import build_comparison, print_comparison_report


# Три вопроса для быстрого демо: архитектура, память, RAG-поиск
DEMO_QUESTION_IDS = {1, 3, 10}


def run_demo(agent: RagAgent) -> None:
    demo_specs = [q for q in CONTROL_QUESTIONS if q["id"] in DEMO_QUESTION_IDS]

    print(f"\n  Запуск демо: {len(demo_specs)} вопроса…\n")
    comparisons = []
    for spec in demo_specs:
        print(f"  Q{spec['id']:02d}: {spec['question'][:70]}…")
        compare_result = agent.compare(spec["question"])
        comparison = build_comparison(compare_result, spec)
        comparisons.append(comparison)

    print_comparison_report(comparisons)


def run_full(agent: RagAgent) -> None:
    print(f"\n  Запуск полного бенчмарка: {len(CONTROL_QUESTIONS)} вопросов…\n")

    benchmark_results = run_benchmark(agent)

    # Конвертируем в формат answer_comparison
    comparisons = [
        build_comparison(r["compare"], r["spec"])
        for r in benchmark_results
    ]

    print_comparison_report(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Демо: сравнение ответов RagAgent с RAG и без RAG"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Запустить полный бенчмарк из 10 вопросов вместо 3",
    )
    parser.add_argument(
        "--strategy",
        default="structure",
        choices=["fixed_size", "structure"],
        help="Стратегия RAG-поиска (default: structure)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Количество чанков для контекста (default: 5)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  RAG AGENT DEMO — сравнение ответов с RAG и без RAG")
    print(f"  Стратегия: {args.strategy}  |  top_k: {args.top_k}")
    print("=" * 72)

    agent = RagAgent(
        strategy=args.strategy,
        top_k=args.top_k,
        verbose=True,
    )

    if args.full:
        run_full(agent)
    else:
        run_demo(agent)


if __name__ == "__main__":
    main()
