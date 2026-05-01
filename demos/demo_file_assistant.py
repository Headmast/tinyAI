#!/usr/bin/env python3
"""
Демо File Assistant — AI-ассистент для автономной работы с файлами проекта.

Запускает 3 сценария:
  1. Найти все использования MCPRegistry в проекте
  2. Сгенерировать CHANGELOG.md на основе анализа проекта
  3. Проверить MCP-серверы на наличие защиты от path traversal

Каждый сценарий — самостоятельная задача, где ассистент сам решает
какие файлы открыть, прочитать и что с ними делать.

Запуск:
    python demos/demo_file_assistant.py
    python demos/demo_file_assistant.py --scenario 1   # только первый сценарий
    python demos/demo_file_assistant.py --scenario 2   # только второй
    python demos/demo_file_assistant.py --scenario 3   # только третий
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from file_assistant import FileAssistant


# ── Сценарии ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id": 1,
        "title": "Поиск использований MCPRegistry",
        "task": (
            "Найди все файлы проекта, где используется класс MCPRegistry. "
            "Для каждого файла покажи: "
            "1) Как импортируется MCPRegistry "
            "2) Как создаётся экземпляр "
            "3) Какие методы вызываются "
            "Дай итоговую сводку: сколько файлов используют MCPRegistry и какие паттерны использования."
        ),
    },
    {
        "id": 2,
        "title": "Генерация CHANGELOG.md",
        "task": (
            "Проанализируй структуру проекта и сгенерируй файл CHANGELOG.md. "
            "Для этого: "
            "1) Прочитай pyproject.toml для получения версии и метаданных "
            "2) Просмотри основные компоненты проекта (list_project_files корня) "
            "3) Прочитай README.md для описания проекта "
            "4) Создай CHANGELOG.md с разделами: версия, дата, основные компоненты, "
            "ключевые возможности каждого компонента. "
            "Сохрани файл через write_file."
        ),
    },
    {
        "id": 3,
        "title": "Проверка MCP-серверов на path traversal",
        "task": (
            "Проверь все MCP-серверы проекта (файлы, имя которых начинается с 'mcp_' и заканчивается на '_server.py') "
            "на наличие защиты от path traversal атак. "
            "Для каждого сервера: "
            "1) Найди файл и прочитай его "
            "2) Проверь есть ли проверка путей (resolve, startswith, safe_path, или аналоги) "
            "3) Отметь: защищён / не защищён / не применимо "
            "Дай итоговый отчёт в виде таблицы: файл | статус защиты | описание механизма."
        ),
    },
]


def run_scenario(scenario: dict, model: str | None = None):
    """Запускает один сценарий и выводит результат."""
    print(f"\n{'═'*70}")
    print(f"  Сценарий {scenario['id']}: {scenario['title']}")
    print(f"{'═'*70}")

    start = time.time()

    with FileAssistant(model=model, verbose=True) as assistant:
        result = assistant.run(scenario["task"])

        elapsed = time.time() - start
        print(f"\n{'─'*70}")
        print(f"📋 РЕЗУЛЬТАТ (сценарий {scenario['id']}):")
        print(f"{'─'*70}")
        print(result)
        print(f"\n⏱️  Время: {elapsed:.1f}с | "
              f"Вызовов инструментов: {assistant.tool_calls_total} | "
              f"Токены: {assistant.token_usage['total_tokens']}")
        print(f"{'═'*70}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Демо File Assistant — 3 сценария работы с файлами")
    parser.add_argument(
        "--scenario", "-s",
        type=int,
        choices=[1, 2, 3],
        help="Запустить только указанный сценарий (1, 2 или 3)"
    )
    parser.add_argument("--model", default=None, help="Модель LLM (по умолчанию из конфига)")
    args = parser.parse_args()

    print("🗂️  File Assistant Demo — AI для автономной работы с файлами проекта")
    print("=" * 70)

    if args.scenario:
        scenarios_to_run = [s for s in SCENARIOS if s["id"] == args.scenario]
    else:
        scenarios_to_run = SCENARIOS

    results = []
    for scenario in scenarios_to_run:
        result = run_scenario(scenario, model=args.model)
        results.append((scenario["id"], result))

    # Итог
    print(f"\n{'═'*70}")
    print("📊 ИТОГО:")
    for sid, _ in results:
        print(f"  ✅ Сценарий {sid} выполнен")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
