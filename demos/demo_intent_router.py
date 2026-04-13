"""
Демонстрация IntentRouter — единый entry-point для всех агентов (День 20+).

Пользователь пишет запрос в свободной форме, IntentRouter:
  1. Классифицирует намерение (keywords → LLM fallback)
  2. Маршрутизирует к подходящему агенту
  3. Возвращает ответ с информацией о маршрутизации

Запуск:
  python demos/demo_intent_router.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from intent_router import IntentRouter, Intent
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


DEMO_REQUESTS = [
    # Должен пойти в generate_content
    "Напиши короткий пост про нейросети для телеграма",

    # Должен пойти в search_logs
    "Покажи мне логи последних разговоров",

    # Должен пойти в schedule_task
    "Какие задачи сейчас в планировщике?",

    # Должен пойти в memory
    "Запомни, пожалуйста: я предпочитаю ответы в формате bullet points",

    # Должен пойти в pipeline
    "Запусти pipeline дайджест по теме AI",

    # Должен пойти в general (fallback LLM)
    "Объясни в двух словах, что такое квантовые вычисления",
]


def main() -> None:
    print("\n" + "█" * 60)
    print("  TinyAI Intent Router — Единый интерфейс")
    print("█" * 60)

    client = create_client()

    # ── Настраиваем IntentRouter ──
    router = IntentRouter(client=client, verbose=True)

    # Регистрируем обработчики
    # Для MCP-задач используем orchestrator
    registry = MCPRegistry(verbose=False)
    registry.register("logs", os.path.join(BASE_DIR, "mcp_server.py"), tags=["logs"])
    registry.register("scheduler", os.path.join(BASE_DIR, "mcp_scheduler_server.py"),
                       tags=["scheduler"])
    registry.discover_tools()

    orchestrator = MCPOrchestratorAgent(client=client, registry=registry, verbose=False)

    router.register_handler(Intent.SEARCH_LOGS, orchestrator.chat)
    router.register_handler(Intent.SCHEDULE_TASK, orchestrator.chat)
    router.register_handler(Intent.MEMORY, orchestrator.chat)

    # ── Демонстрация ──
    for i, request in enumerate(DEMO_REQUESTS, 1):
        print(f"\n{'━' * 60}")
        print(f"  Запрос {i}/{len(DEMO_REQUESTS)}")
        print(f"{'━' * 60}")
        print(f"\n👤 {request}\n")

        result = router.handle(request)

        print(f"\n  📌 Intent: {result.intent.value}")
        print(f"  🤖 Agent: {result.agent_used}")
        print(f"\n💬 Ответ:\n{result.response[:300]}")
        if len(result.response) > 300:
            print("...")

    # ── Статистика ──
    print(f"\n{'═' * 60}")
    print("  📊 Статистика маршрутизации")
    print(f"{'═' * 60}")

    stats = router.get_stats()
    print(f"  Всего запросов: {stats['total_requests']}")
    print(f"  По типам:")
    for intent, count in stats["by_intent"].items():
        print(f"    • {intent}: {count}")
    print(f"  Агенты: {', '.join(stats['agents_used'])}")

    orchestrator.close()
    print(f"\n{'█' * 60}")
    print("  Демонстрация завершена")
    print(f"{'█' * 60}\n")


if __name__ == "__main__":
    main()
