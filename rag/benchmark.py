"""
Бенчмарк для RagAgent: 10 контрольных вопросов с оценкой качества ответов.

Запуск:
    python -m rag.benchmark
    python -m rag.benchmark --strategy fixed_size --top-k 3
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── 10 контрольных вопросов ───────────────────────────────────────────────────

CONTROL_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "question": (
            "Опиши общую архитектуру системы TinyAI: какие основные компоненты "
            "в ней есть, как они взаимодействуют между собой и какую роль "
            "выполняет каждый из них?"
        ),
        "expected_keywords": ["pipeline", "agent", "scheduler", "memory", "journalist"],
        "expected_sources": ["docs/ARCHITECTURE.md"],
        "notes": (
            "Ответ должен перечислить news_agent, journalist_agent, memory_agent, "
            "pipeline, scheduler, MCP и описать связи между ними."
        ),
    },
    {
        "id": 2,
        "question": (
            "Что такое MCP Agent в TinyAI? Перечисли инструменты (tools), "
            "которые он регистрирует, объясни, как происходит вызов инструмента "
            "и чем этот подход отличается от обычного ReAct-агента."
        ),
        "expected_keywords": ["MCP", "tool", "инструмент", "tool_call", "register"],
        "expected_sources": ["docs/tasks/TASK17_README.md", "docs/tasks/TASK16_README.md"],
        "notes": (
            "Ответ должен упомянуть протокол MCP, список инструментов (search, "
            "summarise и др.), механизм tool_call и отличие от чисто реактивного агента."
        ),
    },
    {
        "id": 3,
        "question": (
            "Как устроен memory_agent в TinyAI? Опиши три уровня памяти, "
            "которые он поддерживает, как происходит переход между ними "
            "и каким образом состояние сохраняется между сессиями."
        ),
        "expected_keywords": ["краткосрочная", "рабочая", "долгосрочная", "память", "сессия"],
        "expected_sources": ["docs/ARCHITECTURE.md"],
        "notes": (
            "Ответ должен описать short-term / working / long-term memory, "
            "JSON-блок в ответе модели, персистентность через файлы."
        ),
    },
    {
        "id": 4,
        "question": (
            "Как устроен pipeline для генерации новостных постов? "
            "Перечисли все шаги (роли), опиши, что делает каждый шаг, "
            "и объясни, как данные передаются от одного шага к другому."
        ),
        "expected_keywords": ["planner", "researcher", "writer", "editor", "seo", "шаг"],
        "expected_sources": ["docs/ARCHITECTURE.md"],
        "notes": (
            "Ответ должен содержать цепочку planner → researcher → writer → "
            "editor → seo, описание входа/выхода каждой роли и передачу state."
        ),
    },
    {
        "id": 5,
        "question": (
            "Как работает планировщик заданий (scheduler) в TinyAI? "
            "Где хранятся задания, как определяется время их запуска, "
            "что происходит при сбое выполнения задания?"
        ),
        "expected_keywords": ["SQLite", "cron", "задача", "retry", "daemon", "status"],
        "expected_sources": ["docs/tasks/"],
        "notes": (
            "Ответ должен упомянуть SQLite-хранение, cron-выражения или timestamp, "
            "поля retry/status, фоновый daemon-процесс."
        ),
    },
    {
        "id": 6,
        "question": (
            "Как в проекте TinyAI реализован подсчёт токенов? "
            "Какие библиотеки используются, что происходит если библиотека "
            "недоступна, и как количество токенов влияет на сжатие контекста?"
        ),
        "expected_keywords": ["tiktoken", "TokenCounter", "fallback", "контекст", "сжатие"],
        "expected_sources": ["docs/ARCHITECTURE.md"],
        "notes": (
            "Ответ должен упомянуть tiktoken как основной счётчик, "
            "fallback chars÷4, класс TokenCounter и обрезку контекста при превышении лимита."
        ),
    },
    {
        "id": 7,
        "question": (
            "Как реализован конечный автомат (FSM) в journalist_agent? "
            "Какие состояния существуют, как определяются переходы между ними "
            "и каким образом invariants ограничивают поведение агента?"
        ),
        "expected_keywords": ["FSM", "состояние", "переход", "инвариант", "journalist"],
        "expected_sources": ["docs/tasks/"],
        "notes": (
            "Ответ должен перечислить состояния (INIT, PLAN, RESEARCH, DRAFT, EDIT, DONE), "
            "описать условия переходов и роль invariants.json."
        ),
    },
    {
        "id": 8,
        "question": (
            "В чём разница между стратегиями чанкинга fixed_size и structure "
            "в RAG-модуле TinyAI? Какую стратегию лучше выбрать для "
            "технической документации и почему?"
        ),
        "expected_keywords": ["fixed_size", "structure", "overlap", "заголовок", "Markdown"],
        "expected_sources": ["docs/ARCHITECTURE.md"],
        "notes": (
            "Ответ должен объяснить: fixed_size — равные чанки по токенам с overlap, "
            "structure — разбиение по заголовкам Markdown с сохранением семантических границ. "
            "Для техдоков лучше structure."
        ),
    },
    {
        "id": 9,
        "question": (
            "Какие LLM-модели поддерживаются в TinyAI и как настройка температуры "
            "влияет на качество генерации? При каком значении температуры лучше "
            "генерировать новостные посты, а при каком — аналитические тексты?"
        ),
        "expected_keywords": ["temperature", "GLM", "GPT", "модель", "аналитика", "творческий"],
        "expected_sources": ["docs/MODELS_INFO.md", "docs/TEMPERATURE_GUIDE.md"],
        "notes": (
            "Ответ должен упомянуть GLM-4.7, GPT-5.4, диапазон температур, "
            "низкую температуру для аналитики и высокую для творческих задач."
        ),
    },
    {
        "id": 10,
        "question": (
            "Опиши полный цикл RAG-поиска в TinyAI: как документы индексируются, "
            "как выполняется поиск по запросу, что такое IndexFlatIP в FAISS "
            "и почему для косинусного сходства применяется L2-нормализация эмбеддингов?"
        ),
        "expected_keywords": ["FAISS", "embedding", "cosine", "IndexFlatIP", "L2", "нормализация"],
        "expected_sources": ["docs/ARCHITECTURE.md"],
        "notes": (
            "Ответ должен описать цикл: загрузка → чанкинг → embed → FAISS, "
            "embed query → inner product ≡ cosine после L2-нормализации, top-k из SQLite."
        ),
    },
]


# ── Оценка ───────────────────────────────────────────────────────────────────

def evaluate_answer(
    compare_result: Dict[str, Any],
    question_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Оценивает пару ответов (no_rag / rag) по ключевым словам и источникам.

    Args:
        compare_result: возвращаемый RagAgent.compare()
        question_spec:  запись из CONTROL_QUESTIONS

    Returns:
        dict с полями: question_id, keyword_hits_no_rag, keyword_hits_rag,
                       total_keywords, source_hits, total_sources, rag_wins
    """
    keywords = [kw.lower() for kw in question_spec["expected_keywords"]]
    expected_sources = question_spec["expected_sources"]

    no_rag_answer = (compare_result["no_rag"]["answer"] or "").lower()
    rag_answer = (compare_result["rag"]["answer"] or "").lower()
    rag_sources = [s.lower() for s in compare_result["rag"].get("sources", [])]

    kw_hits_no_rag = sum(1 for kw in keywords if kw in no_rag_answer)
    kw_hits_rag = sum(1 for kw in keywords if kw in rag_answer)

    # Источник считается найденным, если хотя бы один rag_source содержит ожидаемое имя
    source_hits = 0
    for expected in expected_sources:
        expected_lower = expected.lower()
        if any(expected_lower in s or s in expected_lower for s in rag_sources):
            source_hits += 1

    return {
        "question_id": question_spec["id"],
        "question": question_spec["question"],
        "keyword_hits_no_rag": kw_hits_no_rag,
        "keyword_hits_rag": kw_hits_rag,
        "total_keywords": len(keywords),
        "source_hits": source_hits,
        "total_sources": len(expected_sources),
        "rag_wins": kw_hits_rag > kw_hits_no_rag,
    }


# ── Запуск бенчмарка ─────────────────────────────────────────────────────────

def run_benchmark(agent: Any) -> List[Dict[str, Any]]:
    """
    Прогоняет все 10 вопросов через RagAgent.compare() и оценивает ответы.

    Args:
        agent: экземпляр RagAgent

    Returns:
        список результатов с ключами compare_result + evaluation
    """
    results = []
    for spec in CONTROL_QUESTIONS:
        print(f"  Q{spec['id']:02d}: {spec['question'][:60]}…")
        compare_result = agent.compare(spec["question"])
        evaluation = evaluate_answer(compare_result, spec)
        results.append({
            "spec": spec,
            "compare": compare_result,
            "evaluation": evaluation,
        })
    return results


def print_report(results: List[Dict[str, Any]]) -> None:
    """Выводит таблицу результатов бенчмарка."""
    print("\n" + "═" * 90)
    print("  ОТЧЁТ БЕНЧМАРКА RAG AGENT — 10 контрольных вопросов")
    print("═" * 90)
    print(f"  {'#':>2}  {'Ключевые слова':^22}  {'Источники':^12}  {'Победитель':^12}")
    print(f"  {'':>2}  {'no_rag / rag':^22}  {'найдено/ожид.':^12}  {'':^12}")
    print("  " + "─" * 86)

    rag_wins = 0
    for r in results:
        ev = r["evaluation"]
        q_id = ev["question_id"]
        kw_nr = ev["keyword_hits_no_rag"]
        kw_r = ev["keyword_hits_rag"]
        kw_t = ev["total_keywords"]
        src = f"{ev['source_hits']}/{ev['total_sources']}"
        winner = "✅ RAG" if ev["rag_wins"] else "  ─"
        if ev["rag_wins"]:
            rag_wins += 1

        print(
            f"  Q{q_id:02d}"
            f"  {kw_nr}/{kw_t} → {kw_r}/{kw_t} "
            f"{'✓' if kw_r > kw_nr else ' ':>4}"
            f"   src {src:^12}"
            f"   {winner}"
        )

    total = len(results)
    rag_win_pct = rag_wins / total * 100 if total else 0

    print("  " + "─" * 86)
    print(f"\n  RAG побеждает в {rag_wins}/{total} вопросах ({rag_win_pct:.0f}%)")

    # Средние keyword hits
    avg_nr = sum(r["evaluation"]["keyword_hits_no_rag"] for r in results) / total
    avg_r = sum(r["evaluation"]["keyword_hits_rag"] for r in results) / total
    avg_kw_t = sum(r["evaluation"]["total_keywords"] for r in results) / total
    print(f"  Avg keyword precision: no_rag={avg_nr:.1f}/{avg_kw_t:.1f}  "
          f"rag={avg_r:.1f}/{avg_kw_t:.1f}")

    # Средняя точность источников
    src_precision = [
        r["evaluation"]["source_hits"] / r["evaluation"]["total_sources"]
        for r in results if r["evaluation"]["total_sources"] > 0
    ]
    avg_src = sum(src_precision) / len(src_precision) * 100 if src_precision else 0
    print(f"  Avg source precision:  {avg_src:.0f}%")
    print("═" * 90)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    from rag.rag_agent import RagAgent

    parser = argparse.ArgumentParser(description="Бенчмарк RagAgent (10 вопросов)")
    parser.add_argument("--strategy", default="structure",
                        choices=["fixed_size", "structure"],
                        help="Стратегия RAG-поиска (default: structure)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Количество чанков для контекста (default: 5)")
    args = parser.parse_args()

    print(f"\n  Инициализация RagAgent (strategy={args.strategy}, top_k={args.top_k})…")
    agent = RagAgent(strategy=args.strategy, top_k=args.top_k, verbose=True)

    print(f"\n  Запуск бенчмарка: {len(CONTROL_QUESTIONS)} вопросов…\n")
    results = run_benchmark(agent)
    print_report(results)


if __name__ == "__main__":
    main()
