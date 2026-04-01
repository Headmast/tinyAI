"""
Демонстрация персонализации ассистента (День 12).

Два профиля — Системный аналитик и Журналист — отвечают на одни и те же вопросы.
Скрипт показывает:
  1. Как профиль влияет на стиль, формат и глубину ответов
  2. Как агент записывает информацию в память
  3. Сравнение ответов бок о бок

Запуск:
  python demo_personalization.py
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from memory_agent.memory import MemoryManager
from memory_agent.profile import UserProfile, ProfileManager, BUILTIN_PROFILES
from memory_agent.personalized_agent import PersonalizedAgent

load_dotenv()


# ─────────────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "Что такое микросервисная архитектура и когда её стоит применять?",
    "Расскажи о последних трендах в области искусственного интеллекта.",
    "Как правильно организовать работу команды над сложным проектом?",
]

MODEL = os.getenv("DEMO_MODEL", "zai-org/GLM-4.7")
BASE_URL = os.getenv("BASE_URL", "https://api.cloud.ru/v1")
API_KEY = os.getenv("CLOUD_RU_API_KEY", "")

ANALYST_MEMORY_PATH = "memory_data/demo_analyst_long_term.json"
JOURNALIST_MEMORY_PATH = "memory_data/demo_journalist_long_term.json"


def create_client() -> OpenAI:
    """Создаёт OpenAI клиент."""
    if not API_KEY:
        print("❌ Не задан CLOUD_RU_API_KEY в .env")
        sys.exit(1)
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def cleanup_demo_memory() -> None:
    """Удаляет демо-файлы памяти от предыдущих запусков."""
    for path in [ANALYST_MEMORY_PATH, JOURNALIST_MEMORY_PATH]:
        p = Path(path)
        if p.exists():
            p.unlink()


def run_profile_session(
    client: OpenAI,
    profile: UserProfile,
    questions: List[str],
    memory_path: str,
) -> List[dict]:
    """
    Запускает сессию с заданным профилем и списком вопросов.
    Возвращает список {question, answer, memory_state, tokens}.
    """
    memory = MemoryManager(long_term_path=memory_path)
    agent = PersonalizedAgent(
        client=client,
        model=MODEL,
        profile=profile,
        memory_manager=memory,
        verbose=True,
        max_completion_tokens=3000,
        temperature=0.7,
    )

    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n{'▸' * 3} Вопрос {i}/{len(questions)}: {question}")
        print()

        answer = agent.chat(question)
        memory_state = agent.get_memory_state()
        tokens = agent.get_token_usage()

        results.append({
            "question": question,
            "answer": answer,
            "memory_state": {
                "working_facts": list(memory.working.facts),
                "working_goals": list(memory.working.goals),
                "long_term_profile": memory.long_term.get_full_profile(),
                "long_term_decisions": memory.long_term.get_decisions(),
                "long_term_knowledge": memory.long_term.get_all_knowledge(),
            },
            "tokens": dict(tokens),
        })

    return results


def print_comparison(
    analyst_results: List[dict],
    journalist_results: List[dict],
) -> None:
    """Выводит сравнительную таблицу ответов двух профилей."""
    print("\n" + "═" * 80)
    print("  СРАВНЕНИЕ ОТВЕТОВ: Системный аналитик vs Журналист")
    print("═" * 80)

    for i, (a, j) in enumerate(zip(analyst_results, journalist_results)):
        print(f"\n{'─' * 80}")
        print(f"  ВОПРОС {i + 1}: {a['question']}")
        print(f"{'─' * 80}")

        print(f"\n  {'▌ СИСТЕМНЫЙ АНАЛИТИК':}")
        print(f"  {'─' * 38}")
        for line in a["answer"][:1500].split("\n"):
            print(f"  │ {line}")
        if len(a["answer"]) > 1500:
            print(f"  │ ... [{len(a['answer'])} символов всего]")

        print(f"\n  {'▌ ЖУРНАЛИСТ':}")
        print(f"  {'─' * 38}")
        for line in j["answer"][:1500].split("\n"):
            print(f"  │ {line}")
        if len(j["answer"]) > 1500:
            print(f"  │ ... [{len(j['answer'])} символов всего]")

        print(f"\n  📊 Длина ответов: Аналитик={len(a['answer'])} сим. | Журналист={len(j['answer'])} сим.")

    # Итоговая статистика
    print(f"\n{'═' * 80}")
    print("  ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'═' * 80}")

    a_tokens = analyst_results[-1]["tokens"] if analyst_results else {}
    j_tokens = journalist_results[-1]["tokens"] if journalist_results else {}

    print(f"\n  Системный аналитик:")
    print(f"    Токены:   prompt={a_tokens.get('prompt_tokens', 0):,} | "
          f"completion={a_tokens.get('completion_tokens', 0):,} | "
          f"total={a_tokens.get('total_tokens', 0):,}")
    if analyst_results:
        a_mem = analyst_results[-1]["memory_state"]
        print(f"    Память:   факты={len(a_mem['working_facts'])}, "
              f"знания={len(a_mem['long_term_knowledge'])}, "
              f"решения={len(a_mem['long_term_decisions'])}")

    print(f"\n  Журналист:")
    print(f"    Токены:   prompt={j_tokens.get('prompt_tokens', 0):,} | "
          f"completion={j_tokens.get('completion_tokens', 0):,} | "
          f"total={j_tokens.get('total_tokens', 0):,}")
    if journalist_results:
        j_mem = journalist_results[-1]["memory_state"]
        print(f"    Память:   факты={len(j_mem['working_facts'])}, "
              f"знания={len(j_mem['long_term_knowledge'])}, "
              f"решения={len(j_mem['long_term_decisions'])}")

    print(f"\n{'═' * 80}")


def save_results(
    analyst_results: List[dict],
    journalist_results: List[dict],
) -> str:
    """Сохраняет результаты в JSON-файл."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"personalization_comparison_{timestamp}.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "questions": DEMO_QUESTIONS,
        "profiles": {
            "system_analyst": BUILTIN_PROFILES["system_analyst"].to_dict(),
            "journalist": BUILTIN_PROFILES["journalist"].to_dict(),
        },
        "results": {
            "system_analyst": [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "tokens": r["tokens"],
                    "memory_state": r["memory_state"],
                }
                for r in analyst_results
            ],
            "journalist": [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "tokens": r["tokens"],
                    "memory_state": r["memory_state"],
                }
                for r in journalist_results
            ],
        },
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    return filename


def main():
    print("\n" + "═" * 80)
    print("  ДЕНЬ 12: ПЕРСОНАЛИЗАЦИЯ АССИСТЕНТА")
    print("  Сравнение профилей: Системный аналитик vs Журналист")
    print(f"  Модель: {MODEL}")
    print(f"  Вопросов: {len(DEMO_QUESTIONS)}")
    print("═" * 80)

    client = create_client()
    cleanup_demo_memory()

    # ── Сессия 1: Системный аналитик ──
    analyst_profile = BUILTIN_PROFILES["system_analyst"]
    print(f"\n\n{'▓' * 80}")
    print(f"  ПРОФИЛЬ: {analyst_profile.name} ({analyst_profile.role})")
    print(f"  Стиль: {analyst_profile.response_style[:70]}...")
    print(f"  Тон: {analyst_profile.tone}")
    print(f"{'▓' * 80}")

    analyst_results = run_profile_session(
        client=client,
        profile=analyst_profile,
        questions=DEMO_QUESTIONS,
        memory_path=ANALYST_MEMORY_PATH,
    )

    # ── Сессия 2: Журналист ──
    journalist_profile = BUILTIN_PROFILES["journalist"]
    print(f"\n\n{'▓' * 80}")
    print(f"  ПРОФИЛЬ: {journalist_profile.name} ({journalist_profile.role})")
    print(f"  Стиль: {journalist_profile.response_style[:70]}...")
    print(f"  Тон: {journalist_profile.tone}")
    print(f"{'▓' * 80}")

    journalist_results = run_profile_session(
        client=client,
        profile=journalist_profile,
        questions=DEMO_QUESTIONS,
        memory_path=JOURNALIST_MEMORY_PATH,
    )

    # ── Сравнение ──
    print_comparison(analyst_results, journalist_results)

    # ── Сохранение ──
    filename = save_results(analyst_results, journalist_results)
    print(f"\n💾 Результаты сохранены в: {filename}")
    print()


if __name__ == "__main__":
    main()
