#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  TinyAI — Полная демонстрация Orchestration MCP (День 20)
  
  Реальные API-вызовы, реальные MCP-серверы, полное логирование
═══════════════════════════════════════════════════════════════════

Демонстрация включает:
  Часть 1: Health Check — проверка всех MCP-серверов
  Часть 2: Dynamic Tool Discovery — обнаружение 18+ инструментов
  Часть 3: Intent Router — классификация запросов пользователя
  Часть 4: Orchestrator Agent — реальный мульти-серверный флоу с LLM
  Часть 5: Pipeline Templates — готовые цепочки инструментов
  Часть 6: Итоговая сводка
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_LINES = []


def log(msg: str = ""):
    print(msg)
    LOG_LINES.append(msg)


def section(title: str):
    log("")
    log("═" * 70)
    log(f"  {title}")
    log("═" * 70)


def subsection(title: str):
    log("")
    log("─" * 60)
    log(f"  {title}")
    log("─" * 60)


def user_says(text: str):
    log(f"\n  👤 Пользователь: {text}")


def agent_says(text: str):
    lines = text.strip().split("\n")
    log(f"\n  🤖 Агент:")
    for line in lines[:20]:  # ограничиваем для читаемости
        log(f"     {line}")
    if len(lines) > 20:
        log(f"     ... (ещё {len(lines) - 20} строк)")


def create_client() -> OpenAI:
    api_key = os.getenv("CLOUD_API_KEY")
    if not api_key:
        log("❌ CLOUD_API_KEY не найден в .env")
        sys.exit(1)
    return OpenAI(
        api_key=api_key,
        base_url="https://foundation-models.api.cloud.ru/v1",
        timeout=120.0,
    )


def main():
    start_time = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log("")
    log("█" * 70)
    log("█" + " " * 68 + "█")
    log("█   TinyAI Orchestration MCP — Полная демонстрация              █")
    log("█   Реальные серверы • Реальные API-вызовы • Полное логирование  █")
    log("█" + " " * 68 + "█")
    log("█" * 70)
    log(f"\n  Время запуска: {ts}")
    log(f"  Рабочая директория: {BASE_DIR}")

    client = create_client()
    log("  API клиент: создан ✅")

    # ══════════════════════════════════════════════════════════════════════
    # ЧАСТЬ 1: Health Check
    # ══════════════════════════════════════════════════════════════════════

    section("ЧАСТЬ 1: Health Check — Проверка MCP-серверов")

    from mcp_registry import MCPRegistry

    registry = MCPRegistry(verbose=False)
    registry.register("logs", os.path.join(BASE_DIR, "mcp_server.py"),
                       tags=["logs", "memory", "stats"])
    registry.register("scheduler", os.path.join(BASE_DIR, "mcp_scheduler_server.py"),
                       tags=["scheduler", "tasks", "backups"])
    registry.register("search", os.path.join(BASE_DIR, "pipeline", "servers", "search_server.py"),
                       tags=["content", "search"])
    registry.register("format", os.path.join(BASE_DIR, "pipeline", "servers", "format_server.py"),
                       tags=["content", "format"])
    registry.register("store", os.path.join(BASE_DIR, "pipeline", "servers", "store_server.py"),
                       tags=["storage", "file", "db"])

    log("\n  🏥 Проверяем доступность 5 MCP-серверов...")
    health = registry.health_check()
    log("")
    log(f"  {'Сервер':<15} {'Статус':<12} {'Latency':<12} {'Tools':<8}")
    log(f"  {'─'*15} {'─'*12} {'─'*12} {'─'*8}")
    for h in health:
        status_icon = "🟢" if h["status"] == "healthy" else "🔴"
        log(f"  {h['name']:<15} {status_icon} {h['status']:<10} {h['latency_ms']:>6.1f}ms   {h['tool_count']}")

    healthy_count = sum(1 for h in health if h["status"] == "healthy")
    log(f"\n  Итого: {healthy_count}/{len(health)} серверов healthy ✅")

    # ══════════════════════════════════════════════════════════════════════
    # ЧАСТЬ 2: Dynamic Tool Discovery
    # ══════════════════════════════════════════════════════════════════════

    section("ЧАСТЬ 2: Dynamic Tool Discovery — Обнаружение инструментов")

    log("\n  🔍 Вызываем tools/list на каждом сервере...")
    registry.discover_tools()

    servers = registry.list_servers()
    total_tools = 0
    for s in servers:
        log(f"\n  📦 Сервер «{s['name']}» (теги: {', '.join(s['tags'])})")
        log(f"     Инструментов: {s['tool_count']}")
        for tool_name in s["tools"]:
            log(f"       • {tool_name}")
        total_tools += s["tool_count"]

    all_tools = sorted(registry.get_all_tool_names())
    log(f"\n  📊 Всего обнаружено уникальных инструментов: {len(all_tools)}")
    log(f"  📋 Полный список: {', '.join(all_tools)}")

    # Показываем маппинг tool → server
    subsection("Маппинг tool → server")
    for tool_name in all_tools:
        server = registry.get_server_for_tool(tool_name)
        log(f"    {tool_name:<25} → {server}")

    # ══════════════════════════════════════════════════════════════════════
    # ЧАСТЬ 3: Intent Router — Классификация запросов
    # ══════════════════════════════════════════════════════════════════════

    section("ЧАСТЬ 3: Intent Router — Классификация запросов пользователя")

    from intent_router import IntentRouter, Intent

    intent_router = IntentRouter(client=client, verbose=False)

    test_queries = [
        "Напиши статью про искусственный интеллект",
        "Покажи логи последних разговоров",
        "Добавь задачу-напоминание в планировщик",
        "Что ты помнишь обо мне?",
        "Проверь текст как журналист-редактор",
        "Запусти pipeline дайджест",
        "Сколько будет 2+2?",
    ]

    log(f"\n  🏷️  Классифицируем {len(test_queries)} запросов:")
    log("")
    log(f"  {'Запрос':<45} {'Intent':<20} {'Агент'}")
    log(f"  {'─'*45} {'─'*20} {'─'*25}")

    from intent_router import _keyword_classify, _INTENT_AGENT_NAMES

    for q in test_queries:
        intent = intent_router.classify(q)
        agent = _INTENT_AGENT_NAMES.get(intent, "?")
        q_short = q[:43] + ".." if len(q) > 43 else q
        log(f"  {q_short:<45} {intent.value:<20} {agent}")

    # ══════════════════════════════════════════════════════════════════════
    # ЧАСТЬ 4: Orchestrator Agent — Реальный мульти-серверный флоу
    # ══════════════════════════════════════════════════════════════════════

    section("ЧАСТЬ 4: Orchestrator Agent — Реальные API-вызовы")
    log("\n  🚀 Создаём MCPOrchestratorAgent с 5 серверами и LLM...")

    from mcp_orchestrator_agent import MCPOrchestratorAgent

    agent = MCPOrchestratorAgent(
        client=client,
        registry=registry,
        verbose=True,
        max_completion_tokens=3000,
    )

    tool_defs = registry.get_tool_definitions()
    log(f"  📐 Tool definitions для LLM: {len(tool_defs)} инструментов")
    log(f"  🧠 Модель: zai-org/GLM-4.7")
    log(f"  🔄 Max iterations: 15")

    # ── Запрос 1: Работа с логами (сервер: logs) ──
    subsection("Запрос 1: Работа с логами и статистикой")
    user_says("Покажи список доступных логов разговоров и статистику использования токенов.")

    log("\n  ⏳ Отправляем запрос LLM → function calling → MCP серверы...\n")
    t0 = time.time()
    answer1 = agent.chat(
        "Покажи список доступных логов разговоров и статистику использования токенов."
    )
    t1 = time.time()
    agent_says(answer1)
    log(f"\n  ⏱️  Время ответа: {t1 - t0:.1f}с")

    # ── Запрос 2: Работа с планировщиком (сервер: scheduler) ──
    subsection("Запрос 2: Планировщик задач")
    user_says("Какие задачи сейчас есть в планировщике? Покажи полный список.")

    log("\n  ⏳ LLM → scheduler_list_tasks → MCP scheduler server...\n")
    t0 = time.time()
    answer2 = agent.chat(
        "Какие задачи сейчас есть в планировщике? Покажи полный список."
    )
    t1 = time.time()
    agent_says(answer2)
    log(f"\n  ⏱️  Время ответа: {t1 - t0:.1f}с")

    # ── Запрос 3: Поиск (сервер: search) ──
    subsection("Запрос 3: Поиск по данным проекта")
    user_says("Поищи в данных проекта информацию о 'pipeline' и расскажи что нашёл.")

    log("\n  ⏳ LLM → search → MCP search server...\n")
    t0 = time.time()
    answer3 = agent.chat(
        "Поищи в данных проекта информацию о 'pipeline' и расскажи что нашёл."
    )
    t1 = time.time()
    agent_says(answer3)
    log(f"\n  ⏱️  Время ответа: {t1 - t0:.1f}с")

    # ── Запрос 4: Кросс-серверная задача (logs + search + memory) ──
    subsection("Запрос 4: Кросс-серверная задача (3 сервера)")
    user_says(
        "Сделай комплексный анализ: "
        "1) Посмотри сколько логов есть в системе, "
        "2) Поищи в данных слово 'agent', "
        "3) Запиши краткий итог в память (файл orchestrator_demo.json, ключ 'demo_analysis')."
    )

    log("\n  ⏳ LLM → list_logs + search + save_memory → 3 MCP сервера...\n")
    t0 = time.time()
    answer4 = agent.chat(
        "Сделай комплексный анализ: "
        "1) Посмотри сколько логов есть в системе, "
        "2) Поищи в данных слово 'agent', "
        "3) Запиши краткий итог в память (файл orchestrator_demo.json, ключ 'demo_analysis')."
    )
    t1 = time.time()
    agent_says(answer4)
    log(f"\n  ⏱️  Время ответа: {t1 - t0:.1f}с")

    # ── Запрос 5: Форматирование контента (сервер: format) ──
    subsection("Запрос 5: Форматирование контента для Telegram")
    user_says("Отформатируй текст 'TinyAI — мульти-агентная система с оркестрацией MCP-серверов. Поддерживает 5 серверов и 18 инструментов.' для платформы telegram с заголовком 'TinyAI Demo'.")

    log("\n  ⏳ LLM → format_content → MCP format server...\n")
    t0 = time.time()
    answer5 = agent.chat(
        "Отформатируй текст 'TinyAI — мульти-агентная система с оркестрацией MCP-серверов. "
        "Поддерживает 5 серверов и 18 инструментов.' для платформы telegram с заголовком 'TinyAI Demo'."
    )
    t1 = time.time()
    agent_says(answer5)
    log(f"\n  ⏱️  Время ответа: {t1 - t0:.1f}с")

    # ── Сводка маршрутизации ──
    subsection("Сводка маршрутизации MCPRouter")
    summary = agent.routing_summary
    log(f"\n  📊 Всего вызовов инструментов: {summary['total_calls']}")
    log(f"  ✅ Успешных: {summary['successful']}")
    log(f"  ❌ Ошибок: {summary['errors']}")
    log(f"  🖥️  Серверы задействованы: {', '.join(summary['servers_used'])}")
    log(f"  ⏱️  Общее время MCP: {summary['total_elapsed_ms']:.0f}мс")

    log(f"\n  📝 Детализация вызовов:")
    log(f"  {'#':<4} {'Tool':<28} {'Server':<12} {'Time':<10} {'Status'}")
    log(f"  {'─'*4} {'─'*28} {'─'*12} {'─'*10} {'─'*8}")
    for i, call in enumerate(summary["calls"], 1):
        status = "✅" if call["success"] else "❌"
        log(f"  {i:<4} {call['tool']:<28} {call['server']:<12} {call['elapsed_ms']:>6.0f}ms   {status}")

    usage = agent.token_usage
    log(f"\n  🔤 Токены LLM:")
    log(f"     Prompt:     {usage['prompt_tokens']:>8}")
    log(f"     Completion: {usage['completion_tokens']:>8}")
    log(f"     Total:      {usage['total_tokens']:>8}")

    # ══════════════════════════════════════════════════════════════════════
    # ЧАСТЬ 5: Pipeline Templates
    # ══════════════════════════════════════════════════════════════════════

    section("ЧАСТЬ 5: Pipeline Templates — Готовые цепочки")

    from pipeline_templates import list_templates, build_steps

    templates = list_templates()
    log(f"\n  📋 Доступные шаблоны ({len(templates)}):")
    for t in templates:
        log(f"     • {t['name']:<22} — {t['description']}")
        log(f"       Теги: {', '.join(t['tags'])}")
        log(f"       Параметры по умолчанию: {t['default_params']}")

    # Показываем построение шагов для social_media_post
    subsection("Пример: build_steps('social_media_post', topic='AI agents')")
    steps = build_steps("social_media_post", topic="AI agents")
    log(f"\n  Сгенерировано {len(steps)} шагов:")
    for i, step in enumerate(steps):
        args_str = json.dumps(step.args, ensure_ascii=False)
        log(f"    [{i}] {step.name or step.tool}: {step.tool}({args_str})")

    subsection("Пример: build_steps('research_report', topic='Neural Networks')")
    steps2 = build_steps("research_report", topic="Neural Networks")
    log(f"\n  Сгенерировано {len(steps2)} шагов:")
    for i, step in enumerate(steps2):
        args_str = json.dumps(step.args, ensure_ascii=False)
        log(f"    [{i}] {step.name or step.tool}: {step.tool}({args_str})")

    # ══════════════════════════════════════════════════════════════════════
    # ЧАСТЬ 6: Итоговая сводка
    # ══════════════════════════════════════════════════════════════════════

    total_time = time.time() - start_time

    section("ЧАСТЬ 6: Итоговая сводка сессии")

    log(f"""
  ┌────────────────────────────────────────────────────────┐
  │  MCP-серверы зарегистрировано:    {len(servers):<22} │
  │  MCP-серверы healthy:             {healthy_count}/{len(servers):<20} │
  │  Инструментов обнаружено:         {len(all_tools):<22} │
  │  Запросов к агенту:               {agent.tool_calls_total:>3} tool calls         │
  │  Серверы задействованы:           {', '.join(summary['servers_used']):<22} │
  │  Токены LLM:                      {usage['total_tokens']:<22} │
  │  Время MCP-вызовов:              {summary['total_elapsed_ms']:>7.0f}мс              │
  │  Общее время сессии:             {total_time:>7.1f}с               │
  └────────────────────────────────────────────────────────┘
""")

    # Закрываем все соединения
    agent.close()
    log("  🔒 Все MCP-соединения закрыты")

    # Сохраняем лог сессии
    log_path = os.path.join(BASE_DIR, "logs", f"orchestration_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
    log(f"  📄 Лог сессии сохранён: {log_path}")

    log("")
    log("█" * 70)
    log("█   Демонстрация завершена                                       █")
    log("█" * 70)
    log("")


if __name__ == "__main__":
    main()
