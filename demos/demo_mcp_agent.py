"""
Демонстрация MCPAgent — агент с MCP-инструментами (День 17).

Агент автоматически вызывает MCP-инструменты через function calling
для работы с историей разговоров, памятью и статистикой.

Запуск:
  python demo_mcp_agent.py
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from mcp_agent import MCPAgent

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


# ── Сценарий демонстрации ────────────────────────────────────────────────────

DEMO_TASKS = [
    # 1. Агент должен вызвать list_logs
    "Покажи мне список всех доступных логов разговоров.",

    # 2. Агент должен вызвать list_memory + read_memory
    "Какие файлы памяти сейчас есть в системе? Прочитай первый из них.",

    # 3. Агент должен вызвать save_memory (новый инструмент Дня 17)
    "Запомни, пожалуйста: мой любимый формат ответов — краткий и структурированный, с bullet points.",

    # 4. Агент должен вызвать get_usage_stats
    "Покажи статистику использования — сколько всего токенов потрачено?",

    # 5. Агент должен вызвать search_logs
    "Поищи в истории разговоров упоминания слова 'Transformer'.",
]


def main() -> None:
    print("\n" + "█" * 60)
    print("  TinyAI MCP Agent — День 17")
    print("  Агент с MCP-инструментами (function calling)")
    print("█" * 60)

    client = create_client()

    with MCPAgent(client=client, verbose=True) as agent:
        for i, task in enumerate(DEMO_TASKS, 1):
            print(f"\n{'='*60}")
            print(f"  Задача {i}/{len(DEMO_TASKS)}")
            print(f"{'='*60}")
            print(f"\n👤 User: {task}\n")

            response = agent.chat(task)

            print(f"\n🤖 Agent: {response}\n")
            print(f"  [Токены: {agent.token_usage}]")
            print(f"  [MCP-вызовов всего: {agent.tool_calls_total}]")

        # ── Итоговая статистика ─────────────────────────────────────────────────
        print(f"\n{'█'*60}")
        print(f"  ИТОГО")
        print(f"{'█'*60}")
        usage = agent.token_usage
        print(f"  Prompt tokens:     {usage['prompt_tokens']:,}")
        print(f"  Completion tokens: {usage['completion_tokens']:,}")
        print(f"  Total tokens:      {usage['total_tokens']:,}")
        print(f"  MCP-вызовов:       {agent.tool_calls_total}")
        print(f"{'█'*60}\n")


if __name__ == "__main__":
    main()
