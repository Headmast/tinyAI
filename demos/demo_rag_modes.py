"""
Наглядное консольное демо улучшенного RAG.

Показывает:
1. baseline retrieval без фильтрации/rewrite
2. retrieval с rewrite + threshold filter
3. поэтапный streaming-вывод для 4 режимов (baseline/rewrite_only/rerank_only/combined)
4. итоговую сводку в конце

Запуск:
    python demos/demo_rag_modes.py
    python demos/demo_rag_modes.py --question-id 1 --threshold 0.5
    python demos/demo_rag_modes.py --stream-modes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.benchmark import CONTROL_QUESTIONS
from rag.rag_agent import RagAgent
from rag.search import search
from rag.answer_comparison import build_modes_comparison, print_modes_comparison_report


SEP = "=" * 88
SUB = "-" * 88
MODE_SEQUENCE = [
    ("baseline", False, False),
    ("rewrite_only", True, False),
    ("rerank_only", False, True),
    ("combined", True, True),
]


def _pick_question(question_id: int):
    for spec in CONTROL_QUESTIONS:
        if spec["id"] == question_id:
            return spec
    raise SystemExit(f"Вопрос с id={question_id} не найден")


def _select_specs(question_id: int | None, all_questions: bool):
    if all_questions:
        return CONTROL_QUESTIONS
    if question_id is None:
        return [_pick_question(10)]
    return [_pick_question(question_id)]


def _print_results(title: str, results):
    print("\n" + SEP)
    print(title)
    print(SEP)
    if not results:
        print("Нет результатов.")
        return

    for item in results:
        md = item.chunk.metadata
        print(
            f"#{item.rank} score={item.score:.4f} "
            f"source={md.get('source')} section={md.get('section')}"
        )


def _run_stream_modes(agent: RagAgent, question: str):
    """Запускает 4 режима поэтапно со streaming-выводом."""
    modes = {"question": question}

    for mode_name, use_rewrite, use_rerank in MODE_SEQUENCE:
        print("\n" + SEP)
        print(f"MODE: {mode_name}")
        print(
            f"rewrite={'on' if use_rewrite else 'off'} | "
            f"rerank={'on' if use_rerank else 'off'} | "
            f"threshold={agent.similarity_threshold:.2f}"
        )
        print(SUB)
        try:
            result = agent.ask_with_rag(
                question,
                stream=True,
                enable_query_rewrite=use_rewrite,
                enable_rerank=use_rerank,
            )
        except Exception as e:
            err_text = str(e)
            fallback_model = getattr(agent, "fallback_stream_model", None)
            should_fallback = (
                fallback_model
                and agent.model != fallback_model
                and (
                    "Not enough money" in err_text
                    or "insufficient" in err_text.lower()
                    or "quota" in err_text.lower()
                )
            )

            if not should_fallback:
                raise

            print("\n" + SUB)
            print(
                "warning: не удалось продолжить на модели "
                f"{agent.model} ({err_text})."
            )
            print(f"warning: переключаюсь на fallback модель {fallback_model}.")
            print(SUB)
            agent.model = fallback_model
            result = agent.ask_with_rag(
                question,
                stream=True,
                enable_query_rewrite=use_rewrite,
                enable_rerank=use_rerank,
            )
        modes[mode_name] = result

        sources = ", ".join(result.get("sources", [])) if result.get("sources") else "—"
        print("\n" + SUB)
        print(
            f"meta: tokens={result['token_usage']['total_tokens']} | "
            f"latency={result['elapsed_ms']:.0f}ms | "
            f"chunks={result['chunks_used']}"
        )
        print(f"sources: {sources}")

    return modes


def main() -> None:
    parser = argparse.ArgumentParser(description="Демо сравнения режимов улучшенного RAG")
    parser.add_argument("--question-id", type=int, default=None,
                        help="ID вопроса из benchmark.CONTROL_QUESTIONS (если не указан, default: 10)")
    parser.add_argument("--all-questions", action="store_true",
                        help="Запустить demo по всем 10 вопросам benchmark")
    parser.add_argument("--strategy", choices=["fixed_size", "structure"], default="structure")
    parser.add_argument("--top-k-before", type=int, default=10)
    parser.add_argument("--top-k-after", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Для быстрого demo threshold повышен для большей наглядности")
    parser.add_argument("--model", default="gpt-5.4-mini",
                        help="Быстрая модель для пошагового вывода (default: gpt-5.4-mini)")
    parser.add_argument("--stream-modes", action="store_true",
                        help="Пошагово стримить ответы по 4 режимам и вывести итоги в конце")
    parser.add_argument("--fallback-model", default="zai-org/GLM-4.7-Flash",
                        help="Fallback-модель для streaming при ошибке квоты/баланса")
    args = parser.parse_args()

    specs = _select_specs(args.question_id, args.all_questions)

    mode_comparisons = []

    print(SEP)
    print("DEMO: улучшенный RAG — baseline vs filtered/rewrite")
    print(SEP)
    print(f"questions={len(specs)}")

    agent = RagAgent(
        model=args.model,
        strategy=args.strategy,
        top_k=args.top_k_after,
        top_k_before=args.top_k_before,
        similarity_threshold=args.threshold,
        verbose=True,
    )
    agent.fallback_stream_model = args.fallback_model

    for spec in specs:
        question = spec["question"]

        print("\n" + SEP)
        print(f"question_id={spec['id']}")
        print(question)

        rewritten = agent.query_rewriter.rewrite(question)

        print("\n" + SEP)
        print("QUERY REWRITE")
        print(SEP)
        print("original:")
        print(question)
        print("\nrewritten:")
        print(rewritten)

        baseline = search(
            query=question,
            strategy=args.strategy,
            top_k_before=args.top_k_before,
            top_k_after=args.top_k_after,
            similarity_threshold=0.0,
        )
        filtered = search(
            query=rewritten,
            strategy=args.strategy,
            top_k_before=args.top_k_before,
            top_k_after=args.top_k_after,
            similarity_threshold=args.threshold,
        )

        _print_results("BASELINE SEARCH: без фильтра/rewrite", baseline)
        _print_results(
            f"FILTERED SEARCH: rewrite + threshold={args.threshold:.2f}",
            filtered,
        )

        print("\n" + SEP)
        print("SUMMARY")
        print(SEP)
        print(f"baseline_count={len(baseline)}")
        print(f"filtered_count={len(filtered)}")
        print(f"filtered_out={len(baseline) - len(filtered)}")

        if args.stream_modes:
            print("\n" + SEP)
            print("FULL MODES (STREAM): baseline / rewrite_only / rerank_only / combined")
            print(SEP)
            modes = _run_stream_modes(agent, question)
            comparison = build_modes_comparison(modes, spec)
            mode_comparisons.append(comparison)

    if args.stream_modes and mode_comparisons:
        print("\n" + SEP)
        print("FINAL SUMMARY")
        print(SEP)
        print_modes_comparison_report(mode_comparisons)


if __name__ == "__main__":
    main()
