#!/usr/bin/env python3
"""
CLI для автоматического AI-ревью, анализа и рефакторинга кода.

Используется как standalone или из GitHub Actions.

Примеры:
    # Ревью PR (текущая ветка vs main)
    python3 review_cli.py --base-branch main

    # Ревью staged изменений
    python3 review_cli.py --staged

    # Поиск паттерна в коде + AI-анализ
    python3 review_cli.py --search "eval(" --file-pattern "*.py"

    # Анализ файла на проблемы
    python3 review_cli.py --analyze core/config.py

    # Найти и предложить исправления
    python3 review_cli.py --fix mcp_server.py
    python3 review_cli.py --fix mcp_server.py --issue "утечка ресурсов"

    # Рефакторинг
    python3 review_cli.py --refactor mcp_server.py
    python3 review_cli.py --refactor mcp_server.py --goal "разделить на модули"

    # Сохранить результат в файл
    python3 review_cli.py --base-branch main --output review.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from code_reviewer import CodeReviewer


def main():
    parser = argparse.ArgumentParser(
        description="AI Code Reviewer — ревью, анализ, исправления и рефакторинг кода с GPT-5-nano"
    )

    # Режимы ревью diff
    review_group = parser.add_argument_group("Ревью diff")
    review_group.add_argument(
        "--base-branch",
        default="",
        help="Базовая ветка для сравнения (например 'main'). Режим PR-ревью.",
    )
    review_group.add_argument(
        "--staged",
        action="store_true",
        help="Ревью только staged изменений.",
    )
    review_group.add_argument(
        "--diff-file",
        default="",
        help="Путь к файлу с diff (для CI-пайплайнов).",
    )

    # Поиск и анализ
    analysis_group = parser.add_argument_group("Поиск и анализ кода")
    analysis_group.add_argument(
        "--search",
        default="",
        help="Поиск паттерна в закоммиченном коде + AI-анализ.",
    )
    analysis_group.add_argument(
        "--file-pattern",
        default="*.py",
        help="Фильтр файлов для поиска (по умолчанию '*.py').",
    )
    analysis_group.add_argument(
        "--analyze",
        default="",
        help="Анализ файла: баги, безопасность, качество.",
    )

    # Исправления и рефакторинг
    fix_group = parser.add_argument_group("Исправления и рефакторинг")
    fix_group.add_argument(
        "--fix",
        default="",
        help="Найти и предложить исправления для файла.",
    )
    fix_group.add_argument(
        "--issue",
        default="",
        help="Описание проблемы для --fix (опционально).",
    )
    fix_group.add_argument(
        "--refactor",
        default="",
        help="Предложить рефакторинг файла.",
    )
    fix_group.add_argument(
        "--goal",
        default="",
        help="Цель рефакторинга для --refactor (опционально).",
    )

    # Общие
    parser.add_argument(
        "--output", "-o",
        default="",
        help="Сохранить результат в файл (по умолчанию — stdout).",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help="LLM-модель (по умолчанию gpt-5-nano).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Подробный вывод.",
    )

    args = parser.parse_args()

    reviewer = CodeReviewer(model=args.model, verbose=args.verbose)

    try:
        if args.search:
            report = reviewer.search_code(args.search, file_pattern=args.file_pattern)
        elif args.analyze:
            report = reviewer.analyze_file(args.analyze)
        elif args.fix:
            report = reviewer.fix_code(args.fix, issue=args.issue)
        elif args.refactor:
            report = reviewer.refactor(args.refactor, goal=args.goal)
        elif args.diff_file:
            diff_path = Path(args.diff_file)
            if not diff_path.exists():
                print(f"Ошибка: файл {args.diff_file} не найден.", file=sys.stderr)
                sys.exit(1)
            diff_text = diff_path.read_text(encoding="utf-8")
            report = reviewer.review_diff(diff_text)
        elif args.base_branch:
            report = reviewer.review_pr(base_branch=args.base_branch)
        elif args.staged:
            report = reviewer.review_staged()
        else:
            report = reviewer.review_working()

        # Вывод
        if args.output:
            Path(args.output).write_text(report, encoding="utf-8")
            print(f"✅ Результат сохранён: {args.output}")
        else:
            print(report)

    finally:
        reviewer.close()


if __name__ == "__main__":
    main()
