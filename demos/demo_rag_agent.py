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
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_agent import RagAgent
from rag.benchmark import CONTROL_QUESTIONS
from rag.answer_comparison import build_comparison, print_final_summary


# Три вопроса для быстрого демо: архитектура, память, RAG-поиск
DEMO_QUESTION_IDS = {1, 3, 10}

_SEP = "═" * 72
_SUB = "─" * 72


def _run_questions_streaming(agent: RagAgent, specs: List[Dict[str, Any]]) -> None:
    """Запускает вопросы со стримингом: полный текст вопроса, чанки, рассуждения, резюме."""
    comparisons = []

    for spec in specs:
        q_id = spec["id"]
        question = spec["question"]

        # ── Заголовок вопроса целиком ──
        print(f"\n{_SEP}")
        print(f"  Q{q_id:02d}.")
        # Вопрос может быть длинным — печатаем с отступом
        for line in question.splitlines():
            print(f"  {line}")
        print(_SEP)

        # ── БЕЗ RAG ──
        print(f"\n  ▶ БЕЗ RAG  (генерирую...)")
        print(f"  {_SUB}")
        no_rag_result = agent.ask_without_rag(question, stream=True)
        elapsed_nr = no_rag_result["elapsed_ms"]
        print(f"\n  ⏱  {elapsed_nr:.0f}ms")

        # ── С RAG ──
        print(f"\n  ▶ С RAG  (поиск + генерирую...)")
        print(f"  {_SUB}")
        rag_result = agent.ask_with_rag(question, stream=True)
        elapsed_r = rag_result["elapsed_ms"]
        chunks = rag_result["chunks_used"]
        sources = ", ".join(rag_result["sources"]) if rag_result["sources"] else "—"
        print(f"\n  ⏱  {elapsed_r:.0f}ms  │  chunks: {chunks}")
        print(f"  Источники: {sources}")

        # ── Строим метрики ──
        compare_result = {
            "question": question,
            "no_rag": no_rag_result,
            "rag": rag_result,
        }
        comparison = build_comparison(compare_result, spec)
        comparisons.append(comparison)

        ev = comparison["evaluation"]
        kw_nr = ev["keyword_hits_no_rag"]
        kw_r  = ev["keyword_hits_rag"]
        kw_t  = ev["total_keywords"]
        src   = f"{ev['source_hits']}/{ev['total_sources']}"
        delta = kw_r - kw_nr
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        result_mark = "✅ RAG лучше" if ev["rag_wins"] else "─  ничья / no_rag не уступил"

        # ── Короткое резюме ──
        print(f"\n  📊 Резюме: {result_mark}")
        print(f"     keywords:  БЕЗ RAG = {kw_nr}/{kw_t}  →  С RAG = {kw_r}/{kw_t}"
              f"  ({delta_str})")
        print(f"     источники: {src} совпали")
        print(f"     latency:   БЕЗ RAG = {elapsed_nr:.0f}ms  │  С RAG = {elapsed_r:.0f}ms")

    # ── Итоговая сводка по всем вопросам ──
    print_final_summary(comparisons)


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
        verbose=False,  # демо сам управляет выводом
    )

    if args.full:
        specs = CONTROL_QUESTIONS
        print(f"\n  Запуск полного бенчмарка: {len(specs)} вопросов…")
    else:
        specs = [q for q in CONTROL_QUESTIONS if q["id"] in DEMO_QUESTION_IDS]
        print(f"\n  Запуск демо: {len(specs)} вопроса…")

    _run_questions_streaming(agent, specs)


if __name__ == "__main__":
    main()

