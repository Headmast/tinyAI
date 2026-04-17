"""
Демо RAG с обязательными цитатами и источниками (TASK24).

Показывает:
1. Ответы с inline-цитатами [N], списком источников и верифицированными цитатами
2. Режим "не знаю" — когда релевантность контекста ниже порога
3. Подробный вывод: разбор ответа, источники, цитаты, статус уверенности
4. Сравнение confident vs uncertain ответов на разных вопросах

Запуск:
    python demos/demo_rag_citations.py
    python demos/demo_rag_citations.py --verbose
    python demos/demo_rag_citations.py --threshold 0.30 --confidence 0.45
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import CitedAnswer
from rag.rag_agent import RagAgent, DEFAULT_CONFIDENCE_THRESHOLD

SEP = "═" * 80
SUB = "─" * 80

# Questions selected to exercise confident and uncertain modes
DEMO_QUESTIONS = [
    {
        "id": 1,
        "question": "Как работает RAG в проекте TinyAI?",
        "expected": "confident",
        "note": "Внутренний вопрос по документации — должен найти источники.",
    },
    {
        "id": 2,
        "question": "Какие стратегии поиска поддерживает RagAgent?",
        "expected": "confident",
        "note": "Прямой вопрос по документации.",
    },
    {
        "id": 3,
        "question": "Каков курс доллара к евро сегодня?",
        "expected": "uncertain",
        "note": "Внешний факт — документации TinyAI не содержит такого.",
    },
    {
        "id": 4,
        "question": "Что такое протокол MCP и как он используется в TinyAI?",
        "expected": "confident",
        "note": "Внутренний архитектурный вопрос.",
    },
    {
        "id": 5,
        "question": "Кто выиграл последний матч 'Реал Мадрид' — 'Барселона'?",
        "expected": "uncertain",
        "note": "Спортивный результат — не входит в документацию проекта.",
    },
]


def print_header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_result(result: CitedAnswer, question: str, elapsed_ms: float) -> None:
    """Красивый вывод CitedAnswer с разбивкой на секции."""
    status_icon = "⚠️" if result.is_uncertain else "✅"
    status_text = "НЕ ЗНАЮ (uncertain)" if result.is_uncertain else "Ответ с цитатами"

    print(f"\n  {status_icon} {status_text}")
    print(f"  Время: {elapsed_ms:.0f}ms")

    # Ответ (без блока источников/цитат — он уже в answer)
    print(f"\n{SUB}")
    print("  📝 Ответ:")
    print(f"{SUB}")
    # Indent the answer
    for line in result.answer.split("\n"):
        print(f"  {line}")

    # Источники
    if result.sources:
        print(f"\n{SUB}")
        print("  📚 Источники:")
        print(f"{SUB}")
        for src in result.sources:
            score_str = f" (score={src.score:.4f})" if src.score else ""
            print(f"  [{src.index}] {src.source}")
            if src.section:
                print(f"       Раздел: {src.section}{score_str}")
            if src.chunk_id:
                print(f"       Chunk ID: {src.chunk_id}")
            else:
                print(f"       {score_str.strip()}")

    # Цитаты
    if result.quotes:
        print(f"\n{SUB}")
        print("  💬 Цитаты:")
        print(f"{SUB}")
        for q in result.quotes:
            verified_icon = "✅" if q.verified else "⚠️"
            print(f"  [{q.index}] {verified_icon} «{q.text}»")

    print()


def print_separator() -> None:
    print(f"\n{SEP}\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Демо RAG с обязательными цитатами и источниками (TASK24)"
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="Модель для генерации ответов (default: gpt-5.4-mini)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Пог similarity для отсечения (default: 0.30)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Порог уверенности для 'не знаю' (default: 0.45)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Подробный вывод"
    )
    parser.add_argument(
        "--question-id",
        type=int,
        default=None,
        help="Запустить только один вопрос по ID",
    )
    args = parser.parse_args()

    questions = DEMO_QUESTIONS
    if args.question_id is not None:
        questions = [q for q in questions if q["id"] == args.question_id]
        if not questions:
            sys.exit(f"Вопрос с id={args.question_id} не найден")

    conf = (
        args.confidence
        if args.confidence is not None
        else DEFAULT_CONFIDENCE_THRESHOLD
    )

    agent = RagAgent(
        model=args.model,
        strategy="structure",
        similarity_threshold=args.threshold,
        confidence_threshold=conf,
        verbose=args.verbose,
    )

    # ═══════════════════════════════════════════════════════════════════
    # Вводная
    # ═══════════════════════════════════════════════════════════════════
    print_header("TASK24: RAG с обязательными цитатами и источниками")
    print(f"  Модель:       {agent.model}")
    print(f"  Стратегия:    {agent.strategy}")
    print(f"  Threshold:    {agent.similarity_threshold:.2f}")
    print(f"  Confidence:   {agent.confidence_threshold:.2f}")
    print(f"  Rewrite:      {'on' if agent.enable_query_rewrite else 'off'}")
    print(f"  Rerank:       {'on' if agent.enable_rerank else 'off'}")
    print(f"  Вопросы:      {len(questions)}")

    # ═══════════════════════════════════════════════════════════════════
    # Прогон вопросов
    # ═══════════════════════════════════════════════════════════════════
    stats = {
        "total": 0,
        "confident": 0,
        "uncertain": 0,
        "total_sources": 0,
        "total_quotes": 0,
        "errors": 0,
    }
    timings: list[float] = []

    for spec in questions:
        q_id = spec["id"]
        q_text = spec["question"]
        expected = spec["expected"]
        note = spec["note"]

        print_header(f"Вопрос #{q_id}")
        print(f"  Вопрос:    {q_text}")
        print(f"  Ожидание:  {expected}")
        print(f"  Примечание: {note}")

        t_start = time.monotonic()

        try:
            result: CitedAnswer = agent.ask_with_citations(
                q_text,
                top_k=5,
                stream=True,
            )
        except Exception as exc:
            stats["errors"] += 1
            print(f"\n  ❌ Ошибка: {exc}")
            continue

        elapsed_ms = (time.monotonic() - t_start) * 1000
        timings.append(elapsed_ms)

        stats["total"] += 1
        if result.is_uncertain:
            stats["uncertain"] += 1
        else:
            stats["confident"] += 1
        stats["total_sources"] += len(result.sources)
        stats["total_quotes"] += len(result.quotes)

        # Визуальный результат
        print_result(result, q_text, elapsed_ms)

        # Проверка соответствия ожидания
        actual = "uncertain" if result.is_uncertain else "confident"
        match_icon = "🎯" if actual == expected else "⚡"
        match_text = "совпадение" if actual == expected else "различие"
        print(f"  {match_icon} Ожидание: {expected}, Факт: {actual} — {match_text}")

    # ═══════════════════════════════════════════════════════════════════
    # Итоговая статистика
    # ═══════════════════════════════════════════════════════════════════
    print_header("Итоговая статистика")
    avg_time = sum(timings) / len(timings) if timings else 0.0

    print(f"  Всего ответов:     {stats['total']}")
    print(f"  Уверенных:         {stats['confident']}")
    print(f"  Не уверены:        {stats['uncertain']}")
    print(f"  Ошибок:            {stats['errors']}")
    print(f"  Всего источников:  {stats['total_sources']}")
    print(f"  Всего цитат:       {stats['total_quotes']}")
    print(f"  Среднее время:     {avg_time:.0f}ms")
    print(f"  Среднее источников/ответ: {stats['total_sources'] / max(1, stats['total']):.1f}")
    print(f"  Среднее цитат/ответ:      {stats['total_quotes'] / max(1, stats['total']):.1f}")

    print(f"\n  {'═' * 60}")
    print("  Демо завершено!")
    print(f"  {'═' * 60}\n")


if __name__ == "__main__":
    main()
