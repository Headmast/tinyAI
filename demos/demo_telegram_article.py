"""
Сценарий День 12: Персонализированные агенты пишут статью про Telegram.

Два агента с разными профилями:
  1. Системный аналитик — структурированный технический разбор
  2. Журналист — живая статья для широкой аудитории

Тема: ограничения работы Telegram и наиболее эффективные способы с ними справиться.

Каждый агент проходит 3 диалоговых хода:
  1. Анализ темы и ключевых ограничений
  2. Структурирование и выбор лучших решений
  3. Написание финальной статьи

Запуск:
  python3 demo_telegram_article.py
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from memory_agent.memory import MemoryManager
from memory_agent.profile import BUILTIN_PROFILES
from memory_agent.personalized_agent import PersonalizedAgent

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────────────

MODEL = "zai-org/GLM-4.7"
BASE_URL = "https://foundation-models.api.cloud.ru/v1"
API_KEY = os.getenv("CLOUD_API_KEY", "")

TOPIC = "ограничения работы Telegram и самые эффективные способы с ними справиться"

STEPS = [
    (
        "Анализ ограничений",
        "Проанализируй основные ограничения и технические лимиты Telegram: "
        "размер файлов, количество участников в группах/каналах, ограничения API, "
        "ограничения ботов, блокировки и геоограничения, лимиты на рассылку. "
        "Выдели самые болезненные для пользователей и бизнеса."
    ),
    (
        "Эффективные решения",
        "На основе проанализированных ограничений определи 5-7 наиболее эффективных "
        "способов их преодоления. Для каждого способа оцени: сложность реализации, "
        "эффективность, риски. Что реально работает на практике?"
    ),
    (
        "Итоговая статья",
        "Напиши законченную статью на тему: «Ограничения Telegram: о чём молчат и "
        "как с этим работать». Статья должна быть готова к публикации. "
        "Включи конкретные советы, цифры и практические примеры."
    ),
]

MEMORY_DIR = Path("memory_data")


def create_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def cleanup_session_memory(profile_key: str) -> None:
    path = MEMORY_DIR / f"telegram_{profile_key}_long_term.json"
    if path.exists():
        path.unlink()


def print_header(title: str, char: str = "═", width: int = 72) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_step_header(step_num: int, total: int, title: str) -> None:
    print(f"\n{'─' * 72}")
    print(f"  ШАГ {step_num}/{total}: {title}")
    print(f"{'─' * 72}\n")


def run_agent_session(
    client: OpenAI,
    profile_key: str,
    profile_name: str,
) -> dict:
    """
    Запускает диалоговую сессию агента с заданным профилем.
    Возвращает словарь с ответами на каждом шаге и финальной статьёй.
    """
    profile = BUILTIN_PROFILES[profile_key]
    memory_path = str(MEMORY_DIR / f"telegram_{profile_key}_long_term.json")

    memory = MemoryManager(long_term_path=memory_path)
    agent = PersonalizedAgent(
        client=client,
        model=MODEL,
        profile=profile,
        memory_manager=memory,
        verbose=True,
        max_completion_tokens=4000,
        temperature=0.7,
    )

    print_header(
        f"ПРОФИЛЬ: {profile_name}  |  Модель: {MODEL}",
        char="▓",
    )
    print(f"  Стиль: {profile.response_style[:80]}...")
    print(f"  Тон:   {profile.tone}")
    print(f"  Формат: {profile.preferred_format[:80]}...")

    step_results = []

    for i, (step_title, prompt) in enumerate(STEPS, 1):
        print_step_header(i, len(STEPS), step_title)
        answer = agent.chat(prompt)
        step_results.append({
            "step": i,
            "title": step_title,
            "prompt": prompt,
            "answer": answer,
        })

    final_article = step_results[-1]["answer"]

    mem_state = agent.get_memory_state()
    tokens = agent.get_token_usage()

    print(f"\n{'─' * 72}")
    print(f"  ИТОГ [{profile_name}]")
    print(f"{'─' * 72}")
    lt = mem_state.get("long_term", {})
    print(f"  Знания записано в память:  {lt.get('knowledge_count', 0)}")
    print(f"  Решений записано:          {lt.get('decisions_count', 0)}")
    print(f"  Ключей в профиле:          {len(lt.get('profile_keys', []))}")
    print(f"  Токены: prompt={tokens['prompt_tokens']:,} | "
          f"completion={tokens['completion_tokens']:,} | "
          f"total={tokens['total_tokens']:,}")

    return {
        "profile_key": profile_key,
        "profile_name": profile_name,
        "steps": step_results,
        "final_article": final_article,
        "memory": {
            "knowledge_count": lt.get("knowledge_count", 0),
            "decisions_count": lt.get("decisions_count", 0),
            "profile_keys": lt.get("profile_keys", []),
            "working_facts": list(memory.working.facts),
        },
        "tokens": tokens,
    }


def print_side_by_side_articles(analyst: dict, journalist: dict) -> None:
    """Показывает финальные статьи двух агентов бок о бок."""
    print_header("СРАВНЕНИЕ ФИНАЛЬНЫХ СТАТЕЙ", char="═", width=72)

    print(f"\n{'▌' * 3} СИСТЕМНЫЙ АНАЛИТИК {'▌' * 3}")
    print(f"{'─' * 72}")
    for line in analyst["final_article"].split("\n"):
        print(f"  {line}")

    print(f"\n\n{'▌' * 3} ЖУРНАЛИСТ {'▌' * 3}")
    print(f"{'─' * 72}")
    for line in journalist["final_article"].split("\n"):
        print(f"  {line}")


def save_results(analyst: dict, journalist: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"telegram_articles_{timestamp}.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "topic": TOPIC,
        "results": {
            "system_analyst": analyst,
            "journalist": journalist,
        },
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    return filename


def main():
    print_header(
        f"ДЕНЬ 12: ПЕРСОНАЛИЗИРОВАННЫЕ АГЕНТЫ — СТАТЬЯ ПРО TELEGRAM",
        char="═", width=72,
    )
    print(f"\n  Тема:  {TOPIC}")
    print(f"  Шагов: {len(STEPS)}")
    print(f"  Модель: {MODEL}")

    client = create_client()

    # Очищаем память от предыдущих запусков
    for key in ("system_analyst", "journalist"):
        cleanup_session_memory(key)

    # ── Сессия 1: Системный аналитик ──────────────────────────
    analyst_result = run_agent_session(
        client=client,
        profile_key="system_analyst",
        profile_name="Системный аналитик",
    )

    # ── Сессия 2: Журналист ───────────────────────────────────
    journalist_result = run_agent_session(
        client=client,
        profile_key="journalist",
        profile_name="Журналист",
    )

    # ── Сравнение ──────────────────────────────────────────────
    print_side_by_side_articles(analyst_result, journalist_result)

    # ── Итоговая статистика ────────────────────────────────────
    print_header("ИТОГОВАЯ СТАТИСТИКА", char="─", width=72)
    a_tok = analyst_result["tokens"]
    j_tok = journalist_result["tokens"]
    print(f"  Системный аналитик: {a_tok['total_tokens']:,} токенов")
    print(f"  Журналист:          {j_tok['total_tokens']:,} токенов")
    print(f"  Суммарно:           {a_tok['total_tokens'] + j_tok['total_tokens']:,} токенов")

    a_chars = len(analyst_result["final_article"])
    j_chars = len(journalist_result["final_article"])
    print(f"\n  Объём финальной статьи:")
    print(f"  Системный аналитик: {a_chars:,} символов")
    print(f"  Журналист:          {j_chars:,} символов")

    # ── Сохранение ─────────────────────────────────────────────
    filename = save_results(analyst_result, journalist_result)
    print(f"\n  💾 Результаты: {filename}")
    print()


if __name__ == "__main__":
    main()
