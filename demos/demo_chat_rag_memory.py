"""
Демо: Мини-чат с RAG + памятью (БЫСТРЫЙ РЕЖИМ — GPT).

Два сценария по 10 сообщений каждый:

  Сценарий 1: "RAG архитектура TinyAI" (10 сообщений)
  Сценарий 2: "Память и персонализация" (10 сообщений)

Запуск:
    python demos/demo_chat_rag_memory.py
    python demos/demo_chat_rag_memory.py --fast-model gpt-4o-mini
    python demos/demo_chat_rag_memory.py --save-report results.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rag.rag_chat_agent import RagChatAgent

load_dotenv()

# ── Visual formatting ────────────────────────────────────────────────────────
SEP = "═" * 72
SUB = "─" * 72
ARROW = "  ➜ "

# ── Report progressive save ─────────────────────────────────────────────────
_progress_report_path: Optional[str] = None

def _make_report_path(base_path: str) -> str:
    """Создать путь для отчёта с датой/временем, чтобы не перезатирать."""
    path = Path(base_path)
    stem = path.stem
    suffix = path.suffix
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(path.parent / f"{stem}_{ts}{suffix}")

def _save_progress(
    scenario_results: List[Dict[str, Any]],
    current_scenario: int,
    current_turn: int,
    progress_report_path: str,
) -> None:
    """Сохранить промежуточный отчёт после каждого вопроса."""
    if not progress_report_path:
        return

    total_questions = sum(r.get("total_questions", 0) for r in scenario_results)
    total_with_sources = sum(r.get("questions_with_sources", 0) for r in scenario_results)

    # Add in-progress results
    if scenario_results and "results" in scenario_results[-1]:
        total_with_sources += sum(
            1 for r in scenario_results[-1]["results"] if r.get("has_sources")
        )

    progress = {
        "timestamp": datetime.now().isoformat(),
        "status": "IN_PROGRESS",
        "current_scenario": current_scenario,
        "current_turn": current_turn,
        "total_questions_so_far": total_questions,
        "total_with_sources": total_with_sources,
        "scenarios": scenario_results,
    }

    path = Path(progress_report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ── Test scenarios ───────────────────────────────────────────────────────────

scenarios = []
scenario_questions = [
    "Как устроена RAG-система в TinyAI? Опиши основные компоненты",
    "Какие стратегии chunking (разбиения на чанки) поддерживаются?",
    "Как работает векторизация — какой embedder используется?",
    "Расскажи подробнее про FAISS-индекс. Какой тип индекса и почему?",
    "Какие стратегии поиска поддерживает RagAgent?",
    "Как query rewrite улучшает поиск?",
    "Что делает reranker? Как он ранжирует результаты?",
    "Что такое ask_with_citations? Чем отличается от ask_with_rag?",
    "Как анти-галлюцинационный guard работает?",
    "Где хранится индекс — какие файлы в rag_data?",
]

memory_questions = [
    "Опиши архитектуру MemoryAgent. Какие три слоя памяти?",
    "Чем отличается ShortTermMemory от WorkingMemory?",
    "Как WorkingMemory хранит факты, цели и результаты?",
    "Расскажи подробнее про LongTermMemory. Что она сохраняет?",
    "Как агент решает, что сохранить в память?",
    "Как работает extract_memory_block? Какой формат JSON?",
    "Что делает метод set_task в WorkingMemory? Когда она очищается?",
    "Как build_system_prompt включает память в промпт?",
    "Может ли память персистентно сохраняться?",
    "Сравни MemoryAgent и RagChatAgent — ключевые различия?",
]

SCENARIO_1_RAG_ARCHITECTURE = {
    "name": "RAG архитектура TinyAI",
    "description": "Исследование RAG-системы проекта TinyAI",
    "initial_goal": "Изучить RAG-архитектуру TinyAI",
    "questions": scenario_questions,
}

SCENARIO_2_MEMORY_AGENT = {
    "name": "Память и персонализация",
    "description": "Исследование системы MemoryAgent",
    "initial_goal": "Понять MemoryAgent с тремя слоями памяти",
    "initial_constraints": [
        "Отвечай конкретно про MemoryAgent, не уходи в общие темы",
        "Используй только информацию из контекста",
    ],
    "questions": memory_questions,
}


# ── Scenario runner ──────────────────────────────────────────────────────────

def run_scenario(
    scenario: Dict[str, Any],
    agent: RagChatAgent,
    verbose: bool = False,
    scenario_idx: int = 0,
) -> Dict[str, Any]:
    """
    Запустить один сценарий, записать результаты.

    Возвращает dict с метриками и деталями по каждому вопросу.
    """
    global _progress_report_path, scenarios

    name = scenario["name"]
    questions = scenario["questions"]
    initial_goal = scenario.get("initial_goal", "")
    initial_constraints = scenario.get("initial_constraints", [])

    print(f"\n{SEP}")
    print(f"  🎬 СЦЕНАРИЙ: {name}")
    print(f"  📝 Описание: {scenario['description']}")
    print(f"  🎯 Цель: {initial_goal}")
    print(f"{SUB}")

    # Reset agent state
    agent.reset_memory()
    agent.set_goal(initial_goal)

    # Apply initial constraints
    for constraint in initial_constraints:
        agent.memory.add_constraint(constraint)

    if verbose:
        print(f"  Ограничения: {len(initial_constraints)}")
        for c in initial_constraints:
            print(f"    • {c}")

    results: List[Dict[str, Any]] = []
    total_sources = 0
    total_with_sources = 0
    goal_preserved = True

    for i, question in enumerate(questions, start=1):
        print(f"\n  {ARROW} Вопрос {i}/{len(questions)}:")
        print(f"  {question[:80]}{'...' if len(question) > 80 else ''}")

        try:
            start = time.monotonic()
            result = agent.chat_turn(question)
            elapsed = time.monotonic() - start

            # Collect metrics
            has_sources = len(result.get("sources", [])) > 0
            goal_ok = result.get("goal") == initial_goal if initial_goal else True

            if has_sources:
                total_with_sources += 1
                total_sources += len(result["sources"])

            if not goal_ok:
                goal_preserved = False

            # Print brief summary
            sources_str = ", ".join(
                f"[{s['index']}]" for s in result.get("sources", [])
            )
            confidence = result.get("confidence", 0)
            status = "✅" if not result.get("is_uncertain") else "⚠️ "
            print(f"  {status} Ответ получен ({elapsed:.1f}s)")
            print(f"     Источники: {sources_str or '(нет)'}  confidence: {confidence:.2f}")
            print(f"     Цель сохр: {'✅' if goal_ok else '❌'}")

            # Print snippet of answer
            answer = result.get("answer", "")
            preview = answer[:200].replace("\n", " ") if answer else "(нет ответа)"
            print(f"     Ответ: {preview}...")

            results.append({
                "turn": i,
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": question,
                "answer_preview": answer,  # Полный ответ с markdown, не обрезанный
                "has_sources": has_sources,
                "source_count": len(result.get("sources", [])),
                "sources": result.get("sources", []),
                "quotes": result.get("quotes", []),  # Цитаты
                "goal_preserved": goal_ok,
                "current_goal": result.get("goal", ""),
                "is_uncertain": result.get("is_uncertain", False),
                "confidence": confidence,
                "elapsed_seconds": round(elapsed, 2),
            })

            # ── Progressive save after each question ─────────────────────────
            scenario_result = {
                "scenario_name": name,
                "description": scenario["description"],
                "total_questions": i,
                "successful_answers": i,
                "errors": 0,
                "questions_with_sources": total_with_sources,
                "total_sources": total_sources,
                "goal_preserved_count": sum(1 for r in results if r.get("goal_preserved")),
                "results": results[:i],
                "final_memory": {},
            }
            scenarios.append(scenario_result)
            _save_progress(
                scenarios,
                scenario_idx,
                i,
                progress_report_path=_progress_report_path,
            )
            scenarios.pop()

        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            results.append({
                "turn": i,
                "question": question,
                "error": str(e),
                "has_sources": False,
                "goal_preserved": False,
            })

    # ── Scenario summary ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  📊 ИТОГИ СЦЕНАРИЯ: {name}")
    print(f"{SUB}")

    q_count = len(questions)
    error_count = sum(1 for r in results if "error" in r)
    success_count = q_count - error_count
    goal_preserved_count = sum(1 for r in results if r.get("goal_preserved", False))
    avg_confidence = (
        sum(r.get("confidence", 0) for r in results) / (success_count or 1)
    )

    print(f"  Всего вопросов:       {q_count}")
    print(f"  Успешных ответов:     {success_count}")
    print(f"  Ошибок:              {error_count}")
    print(f"  Ответов с источниками: {total_with_sources}/{q_count}")
    print(f"  Всего источников:     {total_sources}")
    print(f"  Цель сохранена:       {goal_preserved_count}/{q_count}")
    print(f"  Средняя уверенность:  {avg_confidence:.2f}")

    # Full memory state
    final_memory = agent.get_memory_state()
    print(f"\n  🧠 Финальное состояние памяти:")
    print(f"     Цель: {final_memory.get('goal', '(пусто)')}")
    print(f"     Уточнения: {len(final_memory.get('clarifications', []))}")
    print(f"     Ограничения: {len(final_memory.get('constraints', []))}")
    for cl in final_memory.get("clarifications", [])[-3:]:
        print(f"       • {cl[:80]}...")
    for c in final_memory.get("constraints", []):
        print(f"       • {c[:80]}")
    print(f"{SEP}")

    return {
        "scenario_name": name,
        "description": scenario["description"],
        "total_questions": q_count,
        "successful_answers": success_count,
        "errors": error_count,
        "questions_with_sources": total_with_sources,
        "total_sources": total_sources,
        "goal_preserved_count": goal_preserved_count,
        "average_confidence": round(avg_confidence, 2),
        "final_memory": final_memory,
        "results": results,
    }


# ── Report generation ────────────────────────────────────────────────────────

def generate_report(
    scenario_results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Сгенерировать и опционально сохранить финальный отчёт."""
    total_questions = sum(r["total_questions"] for r in scenario_results)
    total_with_sources = sum(r["questions_with_sources"] for r in scenario_results)
    total_goals_preserved = sum(r["goal_preserved_count"] for r in scenario_results)
    avg_confidence = (
        sum(r["average_confidence"] for r in scenario_results)
        / len(scenario_results)
    )

    goal_met = (
        total_with_sources == total_questions
        and total_goals_preserved == total_questions
    )

    report = {
        "timestamp": datetime.now().isoformat(),
        "status": "COMPLETE",
        "scenarios": len(scenario_results),
        "total_questions": total_questions,
        "total_with_sources": total_with_sources,
        "total_goals_preserved": total_goals_preserved,
        "average_confidence": round(avg_confidence, 2),
        "overall_result": "PASS" if goal_met else "NEEDS_ATTENTION",
        "scenarios": scenario_results,
    }

    print(f"\n{SEP}")
    print(f"  🏁 ОБЩИЙ ОТЧЁТ")
    print(f"{SEP}")
    print(f"  Сценариев:            {len(scenario_results)}")
    print(f"  Вопросов всего:       {total_questions}")
    print(f"  С источниками:        {total_with_sources}/{total_questions}")
    print(f"  Цель сохранена:       {total_goals_preserved}/{total_questions}")
    print(f"  Средняя уверенность:  {avg_confidence:.2f}")
    print(f"  РЕЗУЛЬТАТ:            {'✅ PASS' if goal_met else '⚠️  NEEDS ATTENTION'}")

    if save_path:
        # Add timestamp to filename so it doesn't overwrite
        final_path = save_path  # Already timestamped by main()
        path = Path(final_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  📄 Отчёт сохранён: {path}")

    print(f"{SEP}")
    return report


# ── Print full demo output ───────────────────────────────────────────────────

def print_full_conversation(
    scenario_results: List[Dict[str, Any]],
) -> None:
    """Напечатать полные диалоги для наглядности."""
    print(f"\n\n{'=' * 80}")
    print(f"  📖 ПОЛНЫЕ ДИАЛОГИ СЦЕНАРИЕВ")
    print(f"{'=' * 80}")

    for scenario_result in scenario_results:
        name = scenario_result["scenario_name"]
        description = scenario_result.get("description", "")
        print(f"\n{'═' * 80}")
        print(f"  💬 СЦЕНАРИЙ: {name}")
        if description:
            print(f"  📝 {description}")
        print(f"{'═' * 80}")

        for result in scenario_result["results"]:
            turn = result["turn"]
            question = result["question"]
            answer_preview = result.get("answer_preview", "(нет ответа)")

            print(f"\n{SUB}")
            print(f"  Ход {turn}:")
            print(f"{SUB}")
            print(f"  👤 Вы: {question}")
            print(f"  🤖 Ассистент: {answer_preview}")

            # Sources
            sources = result.get("sources", [])
            if sources:
                print(f"  📚 Источники ({len(sources)}):")
                for s in sources:
                    section = s.get("section", "")
                    section_str = f" / {section}" if section else ""
                    print(f"     [{s['index']}] {s['source']}{section_str}")
            else:
                print(f"  ⚠️  Источники: (нет)")

            # Memory state after this turn
            goal = result.get("current_goal", "")
            if goal:
                print(f"  🎯 Цель: {goal[:80]}...")

        print(f"\n{'─' * 80}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Демо RAG-чат с памятью задачи"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Подробный вывод"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="LLM модель (по умолчанию gpt-4o-mini)"
    )
    parser.add_argument(
        "--save-report", type=str, default=None,
        dest="save_report",
        help="Путь к файлу отчёта"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: math reranker + heuristic rewrite, один LLM-вызов (~3-4x быстрее)"
    )

    args = parser.parse_args()

    # Set global progress report path — add timestamp so it doesn't overwrite
    global _progress_report_path
    if args.save_report:
        _progress_report_path = _make_report_path(args.save_report)
        print(f"  📄 Отчёт: {_progress_report_path}")

    print(f"\n{SEP}")
    print(f"  RAG Chat Demo — Память + RAG + Источники")
    print(f"  Модель: {args.model}")
    print(f"  Режим: {'⚡ FAST (math reranker, 1 LLM call)' if args.fast else '🔬 FULL (LLM rewrite + rerank + 2 LLM calls)'}")
    print(f"  Сценарии: 2 по 10 вопросов")
    print(f"{'═' * 72}")

    # Initialize agent with fast model
    agent = RagChatAgent(
        model=args.model,
        fast_mode=args.fast,
    )

    scenarios_to_run = [
        SCENARIO_1_RAG_ARCHITECTURE,
        SCENARIO_2_MEMORY_AGENT,
    ]

    scenario_results = []
    for idx, scenario in enumerate(scenarios_to_run, start=1):
        result = run_scenario(scenario, agent, verbose=args.verbose, scenario_idx=idx)
        scenario_results.append(result)

    # Generate final report — use already-timestamped path
    generate_report(scenario_results, save_path=_progress_report_path)

    # Print full conversation
    print_full_conversation(scenario_results)


if __name__ == "__main__":
    main()
