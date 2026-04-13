"""
Демонстрация MCPOrchestratorAgent — оркестрация нескольких MCP-серверов (День 20).

Агент динамически обнаруживает инструменты с 3 серверов и выполняет
длинный мульти-серверный флоу через function calling.

Серверы:
  1. logs    → mcp_server.py             (логи, память, статистика)
  2. scheduler → mcp_scheduler_server.py  (задачи, бэкапы, сводки)
  3. search  → pipeline/servers/search_server.py (поиск по данным)

Запуск:
  python demos/demo_mcp_orchestrator.py
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from mcp_registry import MCPRegistry
from mcp_orchestrator_agent import MCPOrchestratorAgent

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


# ── Сценарий: длинный мульти-серверный флоу ──────────────────────────────────

DEMO_TASKS = [
    # Шаг 1: Инструменты с сервера logs
    "Покажи список всех логов разговоров и статистику использования.",

    # Шаг 2: Инструменты с сервера scheduler
    "Какие задачи сейчас есть в планировщике? Покажи список.",

    # Шаг 3: Инструменты с сервера search (pipeline)
    "Поищи в данных проекта информацию об 'AI agents'.",

    # Шаг 4: Кросс-серверная задача — анализ + запись в память
    (
        "На основе найденной информации:\n"
        "1. Посмотри последние логи разговоров\n"
        "2. Поищи в данных слово 'pipeline'\n"
        "3. Запиши краткий итог анализа в память агента "
        "(файл orchestrator_notes.json, ключ 'analysis_summary')"
    ),

    # Шаг 5: Сводка — использует данные из нескольких серверов
    (
        "Сделай финальный отчёт:\n"
        "- Сколько логов найдено?\n"
        "- Сколько задач в планировщике?\n"
        "- Что удалось найти по запросу 'pipeline'?\n"
        "- Что записано в память?"
    ),
]


def main() -> None:
    print("\n" + "█" * 60)
    print("  TinyAI Orchestrator Agent — День 20")
    print("  Мульти-серверная оркестрация MCP")
    print("█" * 60)

    client = create_client()

    # ── Регистрируем 3 MCP-сервера ──
    registry = MCPRegistry(verbose=True)
    registry.register(
        "logs",
        os.path.join(BASE_DIR, "mcp_server.py"),
        tags=["logs", "memory", "stats"],
    )
    registry.register(
        "scheduler",
        os.path.join(BASE_DIR, "mcp_scheduler_server.py"),
        tags=["scheduler", "tasks", "backups"],
    )
    registry.register(
        "search",
        os.path.join(BASE_DIR, "pipeline", "servers", "search_server.py"),
        tags=["content", "search"],
    )

    print("\n🔍 Обнаружение инструментов...")
    registry.discover_tools()

    # Показываем обнаруженные серверы
    print("\n📋 Зарегистрированные серверы:")
    for s in registry.list_servers():
        print(f"  • {s['name']} ({s['tool_count']} инструментов): {', '.join(s['tools'])}")

    all_tools = registry.get_all_tool_names()
    print(f"\n  Всего инструментов: {len(all_tools)}")

    # ── Создаём агента ──
    agent = MCPOrchestratorAgent(
        client=client,
        registry=registry,
        verbose=True,
    )

    print("\n" + "─" * 60)
    print("  Запуск мульти-серверного флоу")
    print("─" * 60)

    for i, task in enumerate(DEMO_TASKS, 1):
        print(f"\n{'━' * 60}")
        print(f"  Шаг {i}/{len(DEMO_TASKS)}")
        print(f"{'━' * 60}")
        print(f"\n👤 Задача: {task[:100]}{'...' if len(task) > 100 else ''}\n")

        try:
            answer = agent.chat(task)
            print(f"\n🤖 Ответ:\n{answer}")
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")

    # ── Итоги ──
    print(f"\n{'═' * 60}")
    print("  📊 Итоги оркестрации")
    print(f"{'═' * 60}")

    summary = agent.routing_summary
    print(f"  Всего вызовов инструментов: {summary['total_calls']}")
    print(f"  Успешных: {summary['successful']}")
    print(f"  Ошибок: {summary['errors']}")
    print(f"  Серверы задействованы: {', '.join(summary['servers_used'])}")
    print(f"  Общее время MCP: {summary['total_elapsed_ms']:.0f} мс")

    print("\n  📝 Детали вызовов:")
    for call in summary["calls"]:
        status = "✅" if call["success"] else "❌"
        print(f"    {status} {call['tool']} → [{call['server']}] ({call['elapsed_ms']:.0f}ms)")

    usage = agent.token_usage
    print(f"\n  🔤 Токены: {usage['total_tokens']} "
          f"(prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']})")

    agent.close()
    print(f"\n{'█' * 60}")
    print("  Демонстрация завершена")
    print(f"{'█' * 60}\n")


if __name__ == "__main__":
    main()
