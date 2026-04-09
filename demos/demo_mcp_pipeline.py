"""
Демо: MCP Pipeline — контент-завод (День 19).

Запуск:
    python demos/demo_mcp_pipeline.py "тема для поиска"
    python demos/demo_mcp_pipeline.py "AI в медицине" --platform telegram
    python demos/demo_mcp_pipeline.py "нейросети" --platform website --style digest
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_pipeline import PipelineExecutor, Step


def main():
    parser = argparse.ArgumentParser(description="MCP Pipeline — контент-завод")
    parser.add_argument("query", help="Тема/запрос для поиска")
    parser.add_argument(
        "--platform",
        choices=["telegram", "website", "rss", "plain"],
        default="telegram",
        help="Целевая платформа (default: telegram)",
    )
    parser.add_argument(
        "--style",
        choices=["news", "digest", "brief"],
        default="news",
        help="Стиль суммаризации (default: news)",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Не сохранять в БД",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = args.query[:30].replace(" ", "_").lower()
    filename = f"{safe_name}_{timestamp}.md"

    print(f"\n{'='*60}")
    print(f"  Контент-завод TinyAI")
    print(f"  Запрос:    {args.query}")
    print(f"  Платформа: {args.platform}")
    print(f"  Стиль:     {args.style}")
    print(f"{'='*60}\n")

    steps = [
        Step(tool="search", args={"query": args.query}, name="Поиск"),
        Step(tool="summarize", args={
            "text": "$prev.result",
            "style": args.style,
        }, name="Суммаризация"),
        Step(tool="format_content", args={
            "content": "$prev.result",
            "platform": args.platform,
            "title": args.query,
        }, name="Форматирование"),
        Step(tool="save_to_file", args={
            "content": "$prev.result",
            "filename": filename,
        }, name="Сохранение в файл"),
        Step(tool="save_to_db", args={
            "title": args.query,
            "content": "$steps.2.result",
            "platform": args.platform,
            "tags": [args.style, args.platform],
            "summary": "$steps.1.json.summary",
            "source_query": args.query,
        }, name="Сохранение в БД"),
    ]

    executor = PipelineExecutor(verbose=True, save_to_db=not args.no_db)
    result = executor.run(steps)

    print(f"\n{'─'*60}")
    print(f"  Статус: {result.status}")
    print(f"  Время:  {result.total_ms / 1000:.1f}с")
    print(f"  Шагов:  {len(result.steps)}")
    if result.error:
        print(f"  Ошибка: {result.error}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
