"""
Анализ результатов benchmark RAG с цитатами (TASK24).

Читает сохранённые .json-файлы из logs/ и показывает:
1. Общую статистику по всем запускам или последнему
2. Детализацию по каждому вопросу
3. Ошибки и проблемы
4. Сравнение нескольких запусков

Запуск:
    python demos/analyze_benchmark.py                                  # последний запуск
    python demos/analyze_benchmark.py --file logs/benchmark_citations_20260417_120000.json
    python demos/analyze_benchmark.py --compare-all                    # все запуски
    python demos/analyze_penchmark.py --verbose                        # полный текст ответов
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

SEP = "═" * 80
SUB = "─" * 80


def find_latest_benchmark(logs_dir: Path) -> Optional[Path]:
    """Найти самый свежий benchmark-файл."""
    files = sorted(logs_dir.glob("benchmark_citations_*.json"))
    if files:
        return files[-1]
    # Попробуем общий поиск всех benchmark файлов
    files = sorted(logs_dir.glob("benchmark_*.json"))
    if files:
        return files[-1]
    return None


def load_benchmark(filepath: Path) -> Dict[str, Any]:
    """Загрузить benchmark JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def print_header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_subheader(title: str) -> None:
    print(f"\n{SUB}")
    print(f"  {title}")
    print(SUB)


def print_single_analysis(data: Dict[str, Any], verbose: bool = False) -> None:
    """Анализ одного benchmark-запуска."""
    meta = data.get("benchmark", "N/A")
    timestamp = data.get("timestamp", "N/A")
    config = data.get("config", {})
    summary = data.get("summary", {})
    questions = data.get("questions", [])

    print_header(f"ANALYSIS: {meta}")
    print(f"  Timestamp: {timestamp}")

    # Config
    print_subheader("Конфигурация")
    for key, val in config.items():
        print(f"  {key:25s}: {val}")

    # Summary
    print_subheader("Сводка")
    total = summary.get("total", 0)
    match_rate = summary.get("match_rate", 0)
    print(f"  Всего вопросов:          {total}")
    print(f"  Confident:               {summary.get('confident', 0)}")
    print(f"  Uncertain:               {summary.get('uncertain', 0)}")
    print(f"  Ошибки:                  {summary.get('errors', 0)}")
    print(f"  Совпадений:              {summary.get('matches', 0)}/{total} ({100*match_rate:.0f}%)")
    print(f"  Ср. время ответа:        {summary.get('avg_time_ms', 0):.0f}ms")
    print(f"  Ср. уверенность:         {summary.get('avg_confidence', 0):.3f}")
    print(f"  Ср. источников/вопрос:   {summary.get('avg_sources_per_question', 0):.1f}")
    print(f"  Ср. цитат/вопрос:        {summary.get('avg_quotes_per_question', 0):.1f}")

    # Detail by question
    print_subheader("Детализация по вопросам")
    for q in questions:
        q_id = q.get("id", "?")
        question = q.get("question", "")
        
        # Handle both formats
        if "expected_hits" in q:
            # Retrieval benchmark format
            expected = "hits" if q.get("expected_hits") else "no hits"
            actual = "hits" if q.get("actual_hits") else "no hits"
            is_match = q.get("match", False)
            is_uncertain = None
            confidence = q.get("max_score")
            elapsed = q.get("search_time_ms")
            error = q.get("error")
        else:
            # Citation benchmark format
            expected = q.get("expected", "?")
            actual = q.get("actual", "?")
            is_match = q.get("match", False)
            is_uncertain = q.get("is_uncertain")
            confidence = q.get("confidence")
            elapsed = q.get("elapsed_ms")
            error = q.get("error")

        status_icon = "✓" if is_match else "✗"
        if error:
            status_icon = "❌"

        cat = q.get("category", "")
        sources = q.get("sources", [])
        quotes = q.get("quotes", [])

        print(f"\n  [{status_icon}] #{q_id} [{cat}] {question}")
        
        if error:
            print(f"       ОШИБКА: {error}")
            continue

        if "expected_hits" in q:
            # Retrieval format
            status_text = "hits" if q.get("actual_hits") else "no hits"
            chunks = q.get("num_chunks", 0)
            max_s = q.get("max_score", 0)
            print(f"       Ожидалось: {expected} → Получено: {status_text} ({'OK' if is_match else 'MISMATCH'})")
            conf_str = f"{max_s:.3f}" 
            time_str = f"{elapsed:.0f}ms" if elapsed is not None else "N/A"
            print(f"       Max Score: {conf_str} | Время поиска: {time_str} | Чанков: {chunks}")
        else:
            # Citation format
            actual_status = "uncertain" if is_uncertain else "confident"
            print(f"       Ожидалось: {expected} → Получено: {actual_status} ({'OK' if is_match else 'MISMATCH'})")
            conf_str = f"{confidence:.3f}" if confidence is not None else "N/A"
            time_str = f"{elapsed:.0f}ms" if elapsed is not None else "N/A"
            print(f"       Уверенность: {conf_str} | Время: {time_str}")
            print(f"       Источников: {len(sources)} | Цитат: {len(quotes)}")

        # Показать источники
        if sources:
            for s in sources:
                src_name = s.get("source", "unknown")
                src_section = s.get("section", "")
                src_score = s.get("score", 0)
                print(f"         [{s.get('index')}] {src_name} / {src_section} (score={src_score:.3f})")

        # Проверка верификации цитат
        if quotes:
            verified_count = sum(1 for qt in quotes if qt.get("verified"))
            total_quotes = len(quotes)
            if verified_count < total_quotes:
                print(f"       ⚠️  Неверифицированные цитаты: {total_quotes - verified_count}/{total_quotes}")

        # Полный ответ если verbose
        if verbose and q.get("full_answer"):
            print(f"\n       Ответ: {q['full_answer'][:500]}{'...' if len(q.get('full_answer', '')) > 500 else ''}")

    # Проблемы
    print_subheader("Проблемы и рекомендации")
    problems = []
    # Handle both benchmark formats: citations (expected/actual) and retrieval (expected_hits/actual_hits)
    mismatches = [q for q in questions if not q.get("match") and not q.get("error")]
    errors = [q for q in questions if q.get("error")]
    # Citation format checks
    uncertain_confident = [q for q in questions if q.get("expected") == "confident" and q.get("is_uncertain")]
    confident_external = [q for q in questions if q.get("expected") == "uncertain" and not q.get("is_uncertain")]
    # Retrieval format checks
    expected_no_hits_got_hits = [q for q in questions if q.get("expected_hits") is False and q.get("actual_hits") is True]
    expected_hits_got_no = [q for q in questions if q.get("expected_hits") is True and q.get("actual_hits") is False]

    if mismatches:
        problems.append(f"  ⚠️  {len(mismatches)} вопросов не совпали с ожидаемым статусом")
        for q in mismatches:
            if "expected_hits" in q:
                exp = "hits" if q["expected_hits"] else "no hits"
                act = "hits" if q["actual_hits"] else "no hits"
            else:
                exp = q.get("expected", "?")
                act = q.get("actual", "?")
            problems.append(f"     #{q['id']}: {q['question']} (ожидалось {exp}, получено {act})")

    if errors:
        problems.append(f"  ❌ {len(errors)} ошибок при обработке:")
        for q in errors:
            problems.append(f"     #{q['id']}: {q['question']} → {q['error']}")

    if uncertain_confident:
        problems.append(f"  ⚠️  RAG не нашёл источников для внутренних вопросов:")
        for q in uncertain_confident:
            problems.append(f"     #{q['id']}: {q['question']}")

    if confident_external:
        problems.append(f"  ⚠️  RAG нашёл источники для внешних вопросов (ложные срабатывания):")
        for q in confident_external:
            problems.append(f"     #{q['id']}: {q['question']}")

    if not problems:
        problems.append("  ✅ Проблем не выявлено!")

    for p in problems:
        print(p)


def print_comparison(filepaths: List[Path]) -> None:
    """Сравнить несколько benchmark-запусков."""
    print_header(f"COMPARISON: {len(filepaths)} запусков")

    headers = ["#Файл", "Timestamp", "Match%", "Conf", "Unc", "Err", "Avg.ms", "AvgConf"]
    rows = []

    for fp in filepaths:
        data = load_benchmark(fp)
        summary = data.get("summary", {})
        timestamp = data.get("timestamp", "")[:19]
        fname = fp.stem.replace("benchmark_citations_", "")

        rows.append([
            fname,
            timestamp,
            f"{summary.get('match_rate', 0)*100:.0f}%",
            summary.get("confident", 0),
            summary.get("uncertain", 0),
            summary.get("errors", 0),
            f"{summary.get('avg_time_ms', 0):.0f}",
            f"{summary.get('avg_confidence', 0):.3f}",
        ])

    # Красивая таблица
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]

    def fmt_row(vals):
        return "".join(str(v).ljust(w) for v, w in zip(vals, col_widths))

    print(fmt_row(headers))
    print("=" * sum(col_widths))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    import argparse

    logs_dir = Path(__file__).parent.parent / "logs"

    p = argparse.ArgumentParser(description="Анализ результатов benchmark RAG Citations")
    p.add_argument("--file", type=str, help="Путь к конкретному JSON-файлу")
    p.add_argument("--compare-all", action="store_true", help="Сравнить все запуски")
    p.add_argument("--verbose", action="store_true", help="Показать полные тексты ответов")
    args = p.parse_args()

    if args.compare_all:
        # Сравнение всех запусков
        files = sorted(logs_dir.glob("benchmark_citations_*.json"))
        if not files:
            print("❌ Benchmark-файлы не найдены в logs/")
            sys.exit(1)

        if len(files) < 2:
            print("⚠️  Только один benchmark-файл. Сравнение невозможно. Анализирую его.")
            data = load_benchmark(files[-1])
            print_single_analysis(data, verbose=args.verbose)
        else:
            print_comparison(files)
            print(f"\n  Последнюю детальную сводку:")
            data = load_benchmark(files[-1])
            print_single_analysis(data, verbose=args.verbose)

    elif args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"❌ Файл не найден: {filepath}")
            sys.exit(1)
        data = load_benchmark(filepath)
        print_single_analysis(data, verbose=args.verbose)

    else:
        # По умолчанию — последний запуск
        latest = find_latest_benchmark(logs_dir)
        if not latest:
            print("❌ Benchmark-файлы не найдены в logs/")
            print("   Запустите: python demos/benchmark_rag_citations.py")
            sys.exit(1)

        print(f"  Анализирую: {latest.name}")
        data = load_benchmark(latest)
        print_single_analysis(data, verbose=args.verbose)


if __name__ == "__main__":
    main()
