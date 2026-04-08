"""
run_strategy_benchmark — автоматический прогон 20 промптов по всем стратегиям.

Сценарий: «Сбор ТЗ на мобильное приложение для фитнеса» (10–15+ сообщений).
Прогоняет один и тот же набор промптов через:
  1. baseline (полная история, без стратегии)
  2. sliding_window (window=10)
  3. sticky_facts (window=6)
  4. branching (checkpoint на промпте 10, 2 ветки по 5 промптов)

Для каждой стратегии пишет JSONL-лог и summary.
В конце выводит сравнительную таблицу.

Использование:
    python3 run_strategy_benchmark.py [--model gpt-5.4-mini]
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from news_agent.session_manager import ConversationSession, SessionStorage
from news_agent.token_counter import TokenCounter
from news_agent.context_strategies import (
    SlidingWindowStrategy,
    StickyFactsStrategy,
    BranchingStrategy,
    create_strategy,
)
from news_agent.strategy_logger import StrategyLogger, compare_logs


# ─────────────────────────────────────────────────────────────
# 20 промптов — сценарий «Сбор ТЗ на фитнес-приложение»
# ─────────────────────────────────────────────────────────────

SCENARIO_PROMPTS: List[Dict[str, str]] = [
    {
        "prompt": "Привет! Мне нужно разработать мобильное приложение для фитнеса. Можешь помочь составить ТЗ?",
        "expected_facts": "цель: мобильное приложение для фитнеса",
    },
    {
        "prompt": "Приложение должно работать на iOS и Android одновременно.",
        "expected_facts": "платформы: iOS, Android",
    },
    {
        "prompt": "Бюджет проекта — 2 миллиона рублей, не больше.",
        "expected_facts": "бюджет: 2 млн руб",
    },
    {
        "prompt": "Срок разработки — 6 месяцев до первого релиза.",
        "expected_facts": "срок: 6 месяцев",
    },
    {
        "prompt": "Целевая аудитория — люди 25-45 лет, которые регулярно занимаются спортом.",
        "expected_facts": "ЦА: 25-45, спортсмены",
    },
    {
        "prompt": "Основная функция — трекер тренировок с таймером, подсчётом подходов и повторений.",
        "expected_facts": "функция: трекер тренировок",
    },
    {
        "prompt": "Обязательно нужна интеграция с Apple Health и Google Fit для синхронизации данных.",
        "expected_facts": "интеграция: Apple Health, Google Fit",
    },
    {
        "prompt": "Хочу систему достижений и геймификацию — бейджики, уровни, челленджи.",
        "expected_facts": "функция: геймификация, достижения",
    },
    {
        "prompt": "Дизайн должен быть минималистичным, в тёмных тонах, без лишних элементов.",
        "expected_facts": "дизайн: минималистичный, тёмная тема",
    },
    {
        "prompt": "Монетизация — через подписку. Базовый функционал бесплатный, премиум за 299 рублей в месяц.",
        "expected_facts": "монетизация: freemium, подписка 299р/мес",
    },
    {
        "prompt": "Какой стек технологий ты бы посоветовал для нашего проекта с учётом бюджета и сроков?",
        "expected_facts": "(тест: помнит контекст проекта)",
    },
    {
        "prompt": "Очень важно чтобы приложение работало оффлайн — тренировки можно делать без интернета.",
        "expected_facts": "требование: оффлайн-режим",
    },
    {
        "prompt": "Добавим социальную функцию — пользователи должны уметь делиться результатами с друзьями.",
        "expected_facts": "функция: социальная, шеринг",
    },
    {
        "prompt": "Напомни мне, пожалуйста, какой у нас бюджет и сроки на проект?",
        "expected_facts": "(прямой тест памяти: должен ответить 2 млн и 6 мес)",
    },
    {
        "prompt": "Ещё нужны push-уведомления для напоминания о тренировках и мотивационные сообщения.",
        "expected_facts": "функция: push-уведомления",
    },
    {
        "prompt": "Бэкенд решили делать на Python с FastAPI, база данных — PostgreSQL.",
        "expected_facts": "стек: FastAPI, PostgreSQL",
    },
    {
        "prompt": "Мобильное приложение будем делать на Flutter — для кроссплатформенности.",
        "expected_facts": "стек: Flutter",
    },
    {
        "prompt": "Составь, пожалуйста, полное ТЗ на основе всего, что мы обсудили. Включи все требования и детали.",
        "expected_facts": "(итоговый тест: все факты должны присутствовать в ответе)",
    },
    {
        "prompt": "Какие основные риски ты видишь в этом проекте? Что может пойти не так?",
        "expected_facts": "(тест: использует весь контекст для анализа рисков)",
    },
    {
        "prompt": "Спасибо за работу! Подведи итог нашей встречи — что мы решили, какие ключевые моменты.",
        "expected_facts": "(финальный тест: полнота контекста в итоговом резюме)",
    },
]

SYSTEM_PROMPT = (
    "Ты — опытный бизнес-аналитик и IT-консультант. "
    "Помогаешь клиенту составить техническое задание на разработку приложения. "
    "Будь конкретным, задавай уточняющие вопросы если нужно, "
    "запоминай все детали которые сообщает клиент."
)

# Промпты для веток branching (после checkpoint на промпте 10)
BRANCH_A_PROMPTS = [
    "Давай рассмотрим вариант с нативной разработкой вместо Flutter. Swift для iOS и Kotlin для Android.",
    "При нативной разработке какой будет бюджет? Уложимся ли в 2 миллиона?",
    "Какие преимущества нативного подхода для нашего фитнес-приложения?",
    "Сколько разработчиков нужно для нативной версии?",
    "Подведи итог: стоит ли переходить на натив?",
]

BRANCH_B_PROMPTS = [
    "А что если вместо подписки сделать разовую покупку приложения за 990 рублей?",
    "Какие плюсы и минусы разовой покупки vs подписки для фитнес-приложения?",
    "Как разовая покупка повлияет на наш бюджет и окупаемость?",
    "Можно ли комбинировать: базовая версия бесплатная, расширенная — разовая покупка?",
    "Подведи итог: какая модель монетизации лучше для нас?",
]


# ─────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────

def get_client(model: str) -> OpenAI:
    """Создаёт OpenAI-клиент подходящий для модели."""
    if model.startswith("zai-org/"):
        api_key = os.getenv("CLOUD_API_KEY")
        return OpenAI(
            api_key=api_key,
            base_url="https://foundation-models.api.cloud.ru/v1",
            timeout=120.0,
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAI(api_key=api_key, timeout=120.0)


def send_message(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 2000,
) -> Tuple[str, int, int, float]:
    """
    Отправляет messages в API и возвращает (response_text, prompt_tokens, completion_tokens, elapsed_ms).
    """
    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0.3,
    }
    if "gpt-5-nano" in model:
        del params["temperature"]

    start = time.monotonic()
    response = client.chat.completions.create(**params)
    elapsed_ms = (time.monotonic() - start) * 1000

    text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) if usage else 0
    ct = getattr(usage, "completion_tokens", 0) if usage else 0
    return text, pt, ct, elapsed_ms


def run_strategy(
    client: OpenAI,
    model: str,
    strategy_name: str,
    strategy_kwargs: Dict[str, Any],
    prompts: List[Dict[str, str]],
    logger: StrategyLogger,
) -> List[str]:
    """Прогоняет список промптов через одну стратегию. Возвращает список ответов."""
    session = ConversationSession(
        name=f"benchmark_{strategy_name}",
        model=model,
        system_prompt=SYSTEM_PROMPT,
    )

    if strategy_name != "baseline":
        session.set_strategy(strategy_name, **strategy_kwargs)

    responses: List[str] = []

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        session.add_user_message(prompt)

        # Хук стратегии
        if session.context_strategy is not None:
            session.context_strategy.on_user_message(
                user_content=prompt,
                all_messages=session.messages,
                client=client,
                model=model,
            )

        api_msgs = session.get_messages_for_api()
        response_text, pt, ct, elapsed = send_message(client, model, api_msgs)

        session.add_assistant_message(response_text)

        # Хук branching
        if session.context_strategy is not None and hasattr(session.context_strategy, "on_assistant_message"):
            session.context_strategy.on_assistant_message(response_text)

        strategy_stats = {}
        if session.context_strategy is not None:
            strategy_stats = session.context_strategy.get_stats()

        logger.log_turn(
            user_message=prompt,
            assistant_response=response_text,
            prompt_tokens=pt,
            completion_tokens=ct,
            messages_sent_to_api=len(api_msgs),
            messages_total_in_session=len(session.messages),
            response_time_ms=elapsed,
            strategy_stats=strategy_stats,
            extra={"expected_facts": item.get("expected_facts", "")},
        )

        responses.append(response_text)
        print(f"    [{i+1}/{len(prompts)}] prompt_tokens={pt:,} completion_tokens={ct:,} elapsed={elapsed:.0f}ms")

    logger.save_summary()
    return responses


def run_branching(
    client: OpenAI,
    model: str,
    logger: StrategyLogger,
) -> Dict[str, List[str]]:
    """Прогоняет branching-сценарий: 10 общих → checkpoint → 2 ветки по 5."""
    session = ConversationSession(
        name="benchmark_branching",
        model=model,
        system_prompt=SYSTEM_PROMPT,
    )
    session.set_strategy("branching")
    strategy: BranchingStrategy = session.context_strategy  # type: ignore

    responses_main: List[str] = []
    responses_a: List[str] = []
    responses_b: List[str] = []

    # Фаза 1: первые 10 промптов в основном диалоге
    print("  Фаза 1: основной диалог (10 промптов)")
    for i, item in enumerate(SCENARIO_PROMPTS[:10]):
        prompt = item["prompt"]
        session.add_user_message(prompt)
        strategy.on_user_message(prompt, session.messages)

        api_msgs = session.get_messages_for_api()
        response_text, pt, ct, elapsed = send_message(client, model, api_msgs)

        session.add_assistant_message(response_text)
        strategy.on_assistant_message(response_text)

        strategy_stats = strategy.get_stats()
        logger.log_turn(
            user_message=prompt,
            assistant_response=response_text,
            prompt_tokens=pt,
            completion_tokens=ct,
            messages_sent_to_api=len(api_msgs),
            messages_total_in_session=len(session.messages),
            response_time_ms=elapsed,
            strategy_stats=strategy_stats,
            extra={"phase": "main", "expected_facts": item.get("expected_facts", "")},
        )
        responses_main.append(response_text)
        print(f"    [main {i+1}/10] pt={pt:,} ct={ct:,} {elapsed:.0f}ms")

    # Checkpoint
    strategy.save_checkpoint("after_10", session.messages)
    print("  Checkpoint 'after_10' сохранён")

    # Фаза 2а: ветка A (нативная разработка)
    strategy.create_branch("native_dev", "after_10")
    strategy.switch_branch("native_dev")
    print("  Фаза 2а: ветка 'native_dev' (5 промптов)")
    for i, prompt in enumerate(BRANCH_A_PROMPTS):
        strategy.on_user_message(prompt, session.messages)
        api_msgs = strategy.get_messages_for_api(session.messages)
        response_text, pt, ct, elapsed = send_message(client, model, api_msgs)
        strategy.on_assistant_message(response_text)

        logger.log_turn(
            user_message=prompt,
            assistant_response=response_text,
            prompt_tokens=pt,
            completion_tokens=ct,
            messages_sent_to_api=len(api_msgs),
            messages_total_in_session=len(session.messages),
            response_time_ms=elapsed,
            strategy_stats=strategy.get_stats(),
            extra={"phase": "branch_a", "branch": "native_dev"},
        )
        responses_a.append(response_text)
        print(f"    [branch_a {i+1}/5] pt={pt:,} ct={ct:,} {elapsed:.0f}ms")

    # Фаза 2б: ветка B (монетизация)
    strategy.create_branch("monetization", "after_10")
    strategy.switch_branch("monetization")
    print("  Фаза 2б: ветка 'monetization' (5 промптов)")
    for i, prompt in enumerate(BRANCH_B_PROMPTS):
        strategy.on_user_message(prompt, session.messages)
        api_msgs = strategy.get_messages_for_api(session.messages)
        response_text, pt, ct, elapsed = send_message(client, model, api_msgs)
        strategy.on_assistant_message(response_text)

        logger.log_turn(
            user_message=prompt,
            assistant_response=response_text,
            prompt_tokens=pt,
            completion_tokens=ct,
            messages_sent_to_api=len(api_msgs),
            messages_total_in_session=len(session.messages),
            response_time_ms=elapsed,
            strategy_stats=strategy.get_stats(),
            extra={"phase": "branch_b", "branch": "monetization"},
        )
        responses_b.append(response_text)
        print(f"    [branch_b {i+1}/5] pt={pt:,} ct={ct:,} {elapsed:.0f}ms")

    logger.save_summary()
    return {"main": responses_main, "branch_a": responses_a, "branch_b": responses_b}


# ─────────────────────────────────────────────────────────────
# Анализ качества
# ─────────────────────────────────────────────────────────────

MEMORY_CHECK_KEYWORDS = [
    "2 миллион", "2 млн", "2000000", "два миллион",
    "6 месяц", "полгод", "6 мес",
    "ios", "android",
    "flutter",
    "fastapi", "postgresql",
    "299", "подписк",
    "apple health", "google fit",
    "оффлайн", "офлайн", "без интернет",
    "геймифик", "достижен", "бейдж",
    "push", "уведомлен",
    "25-45", "25–45",
    "минималист", "тёмн",
]


def check_memory_in_response(response: str, keywords: List[str]) -> Tuple[int, List[str]]:
    """Проверяет сколько ключевых фактов упомянуто в ответе."""
    response_lower = response.lower()
    found = [kw for kw in keywords if kw.lower() in response_lower]
    return len(found), found


def analyze_responses(
    strategy_name: str,
    responses: List[str],
    prompts: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Анализирует качество ответов для стратегии."""
    # Промпт #14 — прямой тест памяти (бюджет + сроки)
    memory_keywords_14 = ["2 миллион", "2 млн", "6 месяц", "6 мес", "полгод"]
    if len(responses) > 13:
        mem_14_count, mem_14_found = check_memory_in_response(responses[13], memory_keywords_14)
        remembers_budget = any(k in responses[13].lower() for k in ["2 миллион", "2 млн", "2000000"])
        remembers_deadline = any(k in responses[13].lower() for k in ["6 месяц", "6 мес", "полгод"])
    else:
        mem_14_count, mem_14_found = 0, []
        remembers_budget = False
        remembers_deadline = False

    # Промпт #18 — ТЗ (полнота)
    if len(responses) > 17:
        tz_count, tz_found = check_memory_in_response(responses[17], MEMORY_CHECK_KEYWORDS)
    else:
        tz_count, tz_found = 0, []

    # Промпт #20 — итог
    if len(responses) > 19:
        summary_count, summary_found = check_memory_in_response(responses[19], MEMORY_CHECK_KEYWORDS)
    else:
        summary_count, summary_found = 0, []

    return {
        "strategy": strategy_name,
        "total_responses": len(responses),
        "memory_test_14": {
            "remembers_budget": remembers_budget,
            "remembers_deadline": remembers_deadline,
            "keywords_found": mem_14_found,
        },
        "tz_completeness_18": {
            "facts_found": tz_count,
            "facts_total": len(MEMORY_CHECK_KEYWORDS),
            "percentage": round(tz_count / len(MEMORY_CHECK_KEYWORDS) * 100, 1),
            "keywords_found": tz_found,
        },
        "summary_completeness_20": {
            "facts_found": summary_count,
            "facts_total": len(MEMORY_CHECK_KEYWORDS),
            "percentage": round(summary_count / len(MEMORY_CHECK_KEYWORDS) * 100, 1),
            "keywords_found": summary_found,
        },
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    model = "gpt-5.4-mini"
    if len(sys.argv) > 2 and sys.argv[1] == "--model":
        model = sys.argv[2]

    print(f"\n{'═' * 70}")
    print(f"  BENCHMARK: Стратегии управления контекстом")
    print(f"  Модель: {model}")
    print(f"  Промптов: {len(SCENARIO_PROMPTS)}")
    print(f"  Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 70}\n")

    client = get_client(model)
    log_files: List[str] = []
    all_analyses: Dict[str, Any] = {}

    # ── 1. Baseline (полная история) ──────────────────────────
    print("━" * 50)
    print("  1/4  BASELINE (полная история)")
    print("━" * 50)
    logger_bl = StrategyLogger("baseline", model, log_dir="logs")
    responses_bl = run_strategy(client, model, "baseline", {}, SCENARIO_PROMPTS, logger_bl)
    log_files.append(str(logger_bl.log_file))
    all_analyses["baseline"] = analyze_responses("baseline", responses_bl, SCENARIO_PROMPTS)
    summary_bl = logger_bl.get_summary()
    print(f"  Итого: {summary_bl['total_prompt_tokens']:,} prompt + {summary_bl['total_completion_tokens']:,} compl = {summary_bl['total_all_tokens']:,} всего\n")

    # ── 2. Sliding Window ─────────────────────────────────────
    print("━" * 50)
    print("  2/4  SLIDING WINDOW (window=10)")
    print("━" * 50)
    logger_sw = StrategyLogger("sliding_window", model, log_dir="logs")
    responses_sw = run_strategy(client, model, "sliding_window", {"window_size": 10}, SCENARIO_PROMPTS, logger_sw)
    log_files.append(str(logger_sw.log_file))
    all_analyses["sliding_window"] = analyze_responses("sliding_window", responses_sw, SCENARIO_PROMPTS)
    summary_sw = logger_sw.get_summary()
    print(f"  Итого: {summary_sw['total_prompt_tokens']:,} prompt + {summary_sw['total_completion_tokens']:,} compl = {summary_sw['total_all_tokens']:,} всего\n")

    # ── 3. Sticky Facts ───────────────────────────────────────
    print("━" * 50)
    print("  3/4  STICKY FACTS (window=6)")
    print("━" * 50)
    logger_sf = StrategyLogger("sticky_facts", model, log_dir="logs")
    responses_sf = run_strategy(client, model, "sticky_facts", {"window_size": 6}, SCENARIO_PROMPTS, logger_sf)
    log_files.append(str(logger_sf.log_file))
    all_analyses["sticky_facts"] = analyze_responses("sticky_facts", responses_sf, SCENARIO_PROMPTS)
    summary_sf = logger_sf.get_summary()
    print(f"  Итого: {summary_sf['total_prompt_tokens']:,} prompt + {summary_sf['total_completion_tokens']:,} compl = {summary_sf['total_all_tokens']:,} всего\n")

    # ── 4. Branching ──────────────────────────────────────────
    print("━" * 50)
    print("  4/4  BRANCHING (checkpoint + 2 ветки)")
    print("━" * 50)
    logger_br = StrategyLogger("branching", model, log_dir="logs")
    branching_responses = run_branching(client, model, logger_br)
    log_files.append(str(logger_br.log_file))
    # Для branching анализируем main-часть (первые 10 промптов)
    all_analyses["branching"] = {
        "strategy": "branching",
        "main_responses": len(branching_responses["main"]),
        "branch_a_responses": len(branching_responses["branch_a"]),
        "branch_b_responses": len(branching_responses["branch_b"]),
    }
    summary_br = logger_br.get_summary()
    print(f"  Итого: {summary_br['total_prompt_tokens']:,} prompt + {summary_br['total_completion_tokens']:,} compl = {summary_br['total_all_tokens']:,} всего\n")

    # ── Сравнительная таблица ─────────────────────────────────
    print("\n" + "═" * 70)
    print("  СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("═" * 70)

    summaries = {
        "baseline": summary_bl,
        "sliding_window": summary_sw,
        "sticky_facts": summary_sf,
        "branching": summary_br,
    }

    print(f"\n{'Метрика':<30} {'Baseline':>12} {'SlidWin':>12} {'Facts':>12} {'Branch':>12}")
    print("─" * 78)

    print(f"{'Prompt tokens':<30} {summary_bl['total_prompt_tokens']:>12,} {summary_sw['total_prompt_tokens']:>12,} {summary_sf['total_prompt_tokens']:>12,} {summary_br['total_prompt_tokens']:>12,}")
    print(f"{'Completion tokens':<30} {summary_bl['total_completion_tokens']:>12,} {summary_sw['total_completion_tokens']:>12,} {summary_sf['total_completion_tokens']:>12,} {summary_br['total_completion_tokens']:>12,}")
    print(f"{'Extraction tokens':<30} {summary_bl['total_extraction_tokens']:>12,} {summary_sw['total_extraction_tokens']:>12,} {summary_sf['total_extraction_tokens']:>12,} {summary_br['total_extraction_tokens']:>12,}")
    print(f"{'ВСЕГО токенов':<30} {summary_bl['total_all_tokens']:>12,} {summary_sw['total_all_tokens']:>12,} {summary_sf['total_all_tokens']:>12,} {summary_br['total_all_tokens']:>12,}")
    print(f"{'Среднее время ответа (мс)':<30} {summary_bl['avg_response_time_ms']:>12.0f} {summary_sw['avg_response_time_ms']:>12.0f} {summary_sf['avg_response_time_ms']:>12.0f} {summary_br['avg_response_time_ms']:>12.0f}")
    print("─" * 78)

    # Экономия относительно baseline
    bl_tokens = summary_bl["total_all_tokens"]
    for name, s in summaries.items():
        if name == "baseline":
            continue
        saving = bl_tokens - s["total_all_tokens"]
        pct = (saving / bl_tokens * 100) if bl_tokens > 0 else 0
        sign = "+" if saving < 0 else ""
        print(f"{'Экономия vs baseline (' + name + ')':<40} {sign}{saving:>8,} токенов ({sign}{pct:.1f}%)")

    # Качество памяти
    print(f"\n{'═' * 70}")
    print("  КАЧЕСТВО ПАМЯТИ")
    print("═" * 70)

    for name in ["baseline", "sliding_window", "sticky_facts"]:
        a = all_analyses[name]
        mt = a["memory_test_14"]
        tz = a["tz_completeness_18"]
        sm = a["summary_completeness_20"]
        print(f"\n  {name}:")
        print(f"    Помнит бюджет (#14):      {'✅' if mt['remembers_budget'] else '❌'}")
        print(f"    Помнит сроки (#14):        {'✅' if mt['remembers_deadline'] else '❌'}")
        print(f"    Полнота ТЗ (#18):          {tz['facts_found']}/{tz['facts_total']} ({tz['percentage']}%)")
        print(f"    Полнота итога (#20):       {sm['facts_found']}/{sm['facts_total']} ({sm['percentage']}%)")

    print(f"\n  branching:")
    print(f"    Основной диалог: {all_analyses['branching']['main_responses']} ответов")
    print(f"    Ветка A (натив): {all_analyses['branching']['branch_a_responses']} ответов")
    print(f"    Ветка B (монетизация): {all_analyses['branching']['branch_b_responses']} ответов")

    # ── Сохраняем полный отчёт ────────────────────────────────
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "prompts_count": len(SCENARIO_PROMPTS),
        "summaries": summaries,
        "quality_analysis": all_analyses,
        "log_files": log_files,
    }

    report_file = Path("logs") / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📄 Полный отчёт: {report_file}")
    print(f"📂 Логи: {', '.join(log_files)}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
