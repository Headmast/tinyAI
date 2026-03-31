"""
Тестовая сессия агента с трёхслойной моделью памяти.

Демонстрирует:
  1. Краткосрочная память — диалог хранится и передаётся в API
  2. Рабочая память — факты, цели, контекст текущей задачи
  3. Долговременная память — профиль, решения, знания (сохраняются на диск)
  4. Агент сам решает, что и куда сохранять

Результаты сохраняются в memory_data/test_session_log.json
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from memory_agent.memory import MemoryManager
from memory_agent.agent import MemoryAgent

load_dotenv()


def create_client() -> OpenAI:
    api_key = os.getenv("CLOUD_API_KEY")
    if not api_key:
        print("❌ CLOUD_API_KEY не найден в .env")
        sys.exit(1)
    return OpenAI(
        api_key=api_key,
        base_url="https://foundation-models.api.cloud.ru/v1",
        timeout=120.0,
    )


TEST_CONVERSATION = [
    # 1. Знакомство — агент должен сохранить профиль в долговременную память
    "Привет! Меня зовут Кирилл, я студент-программист. Изучаю AI и машинное обучение.",

    # 2. Задача — агент должен создать рабочую память
    "Помоги мне разобраться в архитектуре Transformer. Мне нужно подготовить доклад.",

    # 3. Углубление — агент добавляет факты в рабочую и знания в долговременную
    "Расскажи подробнее про механизм self-attention. Какие у него ключевые формулы?",

    # 4. Смена подзадачи — проверяем что рабочая память обновляется
    "А чем отличается encoder от decoder в оригинальном Transformer?",

    # 5. Проверка долговременной памяти — агент должен помнить имя и контекст
    "Напомни, как меня зовут и о чём мы разговариваем?",

    # 6. Новое знание для долговременной памяти
    "Запомни: мой любимый фреймворк — PyTorch, и я предпочитаю объяснения с примерами кода.",

    # 7. Проверка сохранения предпочтений
    "Объясни positional encoding, учитывая мои предпочтения.",
]


def run_test_session():
    print("=" * 65)
    print("  ТЕСТОВАЯ СЕССИЯ: Агент с моделью памяти")
    print("  Модель: GLM-4.7 (Cloud.ru)")
    print("=" * 65)

    client = create_client()

    memory_mgr = MemoryManager(
        short_term_limit=40,
        long_term_path="memory_data/long_term.json",
    )

    agent = MemoryAgent(
        client=client,
        model="zai-org/GLM-4.7",
        memory_manager=memory_mgr,
        verbose=True,
        max_completion_tokens=4000,
        temperature=0.7,
    )

    session_log = {
        "session_start": datetime.now().isoformat(),
        "model": "zai-org/GLM-4.7",
        "turns": [],
        "memory_snapshots": [],
    }

    for i, user_msg in enumerate(TEST_CONVERSATION, 1):
        print(f"\n{'═' * 65}")
        print(f"  👤 ПОЛЬЗОВАТЕЛЬ (ход {i}/{len(TEST_CONVERSATION)}):")
        print(f"  {user_msg}")
        print(f"{'═' * 65}")

        try:
            response = agent.chat(user_msg)
        except Exception as e:
            response = f"[ОШИБКА: {e}]"
            print(f"\n  ❌ Ошибка: {e}")

        print(f"\n  🤖 АССИСТЕНТ:")
        print(f"  {response[:500]}")
        if len(response) > 500:
            print(f"  ... (ещё {len(response) - 500} символов)")

        memory_state = agent.get_memory_state()
        session_log["turns"].append({
            "turn": i,
            "user": user_msg,
            "assistant": response,
            "memory_after": memory_state,
        })

        memory_state_snapshot = {
            "after_turn": i,
            "short_term_messages": memory_state["short_term"]["message_count"],
            "working_facts": list(agent.memory.working.facts),
            "working_goals": list(agent.memory.working.goals),
            "long_term_profile": agent.memory.long_term.get_full_profile(),
            "long_term_decisions_count": memory_state["long_term"]["decisions_count"],
            "long_term_knowledge_count": memory_state["long_term"]["knowledge_count"],
        }
        session_log["memory_snapshots"].append(memory_state_snapshot)

    session_log["session_end"] = datetime.now().isoformat()
    session_log["total_tokens"] = agent.get_token_usage()
    session_log["final_memory_state"] = agent.get_memory_state()

    # Детальная инспекция долговременной памяти
    session_log["final_long_term_detail"] = {
        "profile": agent.memory.long_term.get_full_profile(),
        "decisions": agent.memory.long_term.get_decisions(),
        "knowledge": agent.memory.long_term.get_all_knowledge(),
    }

    log_path = "memory_data/test_session_log.json"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(session_log, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'═' * 65}")
    print(f"  📁 Лог сессии сохранён: {log_path}")
    print(f"{'═' * 65}")

    _print_final_report(agent, session_log)

    agent.memory.save_state_to_file("memory_data/final_state.json")
    print(f"  📁 Состояние памяти сохранено: memory_data/final_state.json")


def _print_final_report(agent: MemoryAgent, log: dict):
    print(f"\n{'═' * 65}")
    print("  ИТОГОВЫЙ ОТЧЁТ")
    print(f"{'═' * 65}")

    tokens = agent.get_token_usage()
    print(f"\n  🔢 Токены:")
    print(f"     Prompt:     {tokens['prompt_tokens']:,}")
    print(f"     Completion: {tokens['completion_tokens']:,}")
    print(f"     Total:      {tokens['total_tokens']:,}")

    print(f"\n  📊 Память после сессии:")

    stm = agent.memory.short_term
    print(f"\n  1️⃣  КРАТКОСРОЧНАЯ ({stm.message_count} сообщений):")
    for msg in stm.get_last_n(4):
        role = msg['role']
        text = msg['content'][:80]
        print(f"       [{role}] {text}...")

    wm = agent.memory.working
    print(f"\n  2️⃣  РАБОЧАЯ:")
    print(f"       Задача: {wm.current_task or '(нет)'}")
    print(f"       Цели:  {wm.goals or '(нет)'}")
    print(f"       Факты: {len(wm.facts)} шт.")
    for f in wm.facts[:5]:
        print(f"         - {f[:100]}")

    ltm = agent.memory.long_term
    print(f"\n  3️⃣  ДОЛГОВРЕМЕННАЯ:")
    profile = ltm.get_full_profile()
    print(f"       Профиль: {json.dumps(profile, ensure_ascii=False, default=str)[:200]}")
    decisions = ltm.get_decisions()
    print(f"       Решения: {len(decisions)} шт.")
    for d in decisions[-3:]:
        print(f"         - {d['decision'][:100]}")
    knowledge = ltm.get_all_knowledge()
    print(f"       Знания:  {len(knowledge)} шт.")
    for k in knowledge[-3:]:
        print(f"         - [{k['topic']}]: {k['content'][:80]}")

    print(f"\n  📈 Динамика памяти по ходам:")
    for snap in log["memory_snapshots"]:
        t = snap["after_turn"]
        print(f"       Ход {t}: STM={snap['short_term_messages']} msg | "
              f"WM facts={len(snap['working_facts'])} | "
              f"LTM profile={len(snap['long_term_profile'])} keys, "
              f"decisions={snap['long_term_decisions_count']}, "
              f"knowledge={snap['long_term_knowledge_count']}")

    print(f"\n{'═' * 65}")


if __name__ == "__main__":
    run_test_session()
