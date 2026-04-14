"""
Сравнение двух стратегий chunking: fixed_size vs structure.

Метрики:
    - Количество чанков
    - Средний/мин/макс размер чанка (токены)
    - Дисперсия размера
    - Покрытие исходного текста
    - Retrieval precision@5
    - Средний cosine similarity score

CLI:
    python -m rag.compare_strategies
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import Chunk, SearchResult
from rag.index_store import get_chunk_stats, get_all_chunks, FAISSIndexStore
from rag.search import search

DEFAULT_INDEX_DIR = Path(__file__).parent.parent / "rag_data"

# Тестовые запросы с ожидаемыми релевантными файлами
TEST_QUERIES: List[Dict[str, Any]] = [
    {
        "query": "Как работает memory_agent и управление памятью?",
        "expected_sources": ["docs/ARCHITECTURE.md"],
    },
    {
        "query": "Какие инструменты и tools у MCP Agent?",
        "expected_sources": ["docs/tasks/TASK17_README.md", "docs/tasks/TASK16_README.md"],
    },
    {
        "query": "Как считаются токены и работает TokenCounter?",
        "expected_sources": ["docs/tasks/TASK8_README.md"],
    },
    {
        "query": "Стратегии контекста SlidingWindow StickyFacts Branching",
        "expected_sources": ["docs/tasks/TASK10_PLAN.md"],
    },
    {
        "query": "FSM Agent и конечный автомат ArticleFSMAgent",
        "expected_sources": ["docs/tasks/TASK13_README.md"],
    },
    {
        "query": "Оркестрация нескольких MCP серверов MCPRegistry MCPRouter",
        "expected_sources": ["docs/tasks/TASK20_README.md"],
    },
    {
        "query": "Pipeline composition и композиция пайплайнов",
        "expected_sources": ["docs/tasks/TASK19_README.md"],
    },
    {
        "query": "Температура модели и влияние на генерацию",
        "expected_sources": [
            "docs/TEMPERATURE_GUIDE.md",
            "docs/GPT54_TEMPERATURE_ANALYSIS.md",
            "docs/TEMPERATURE_TEST_RESULTS.md",
        ],
    },
    {
        "query": "Модели OpenAI GPT цены стоимость pricing",
        "expected_sources": ["docs/MODELS_INFO.md", "docs/BILLING_UPDATE.md"],
    },
    {
        "query": "Инварианты журналиста JournalistAgent constraints",
        "expected_sources": ["docs/tasks/TASK14_README.md"],
    },
    {
        "query": "Персонализация UserProfile профили пользователей",
        "expected_sources": ["docs/tasks/TASK12_README.md"],
    },
    {
        "query": "Компрессия контекста ContextCompressor sliding window",
        "expected_sources": ["docs/tasks/TASK9_README.md"],
    },
    {
        "query": "Новостной агент news agent pipeline генерация постов",
        "expected_sources": ["docs/tasks/TASK6_README.md"],
    },
    {
        "query": "Тестирование pytest тесты unit test",
        "expected_sources": ["docs/TESTING.md", "docs/TEST_SUMMARY.md"],
    },
    {
        "query": "Сессии разговора ConversationSession SessionStorage",
        "expected_sources": ["docs/tasks/TASK7_README.md"],
    },
]


def compute_chunking_stats(
    index_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Статистика чанков для каждой стратегии."""
    db_path = index_dir / "chunks.db"
    result: Dict[str, Dict[str, Any]] = {}

    for strategy in ["fixed_size", "structure"]:
        chunks = get_all_chunks(db_path, strategy=strategy)
        if not chunks:
            continue

        token_counts = [c["token_count"] for c in chunks]
        char_counts = [c["char_count"] for c in chunks]

        # Уникальные источники
        sources = set(c["source"] for c in chunks)
        # Уникальные секции (для structure)
        sections = set(c["section"] for c in chunks if c["section"] != "N/A")

        result[strategy] = {
            "chunk_count": len(chunks),
            "sources_count": len(sources),
            "sections_count": len(sections),
            "avg_tokens": statistics.mean(token_counts),
            "median_tokens": statistics.median(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "stdev_tokens": statistics.stdev(token_counts) if len(token_counts) > 1 else 0,
            "total_tokens": sum(token_counts),
            "total_chars": sum(char_counts),
            "avg_chars": statistics.mean(char_counts),
        }

    return result


def compute_retrieval_metrics(
    index_dir: Path,
    top_k: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """Retrieval precision@k для каждой стратегии."""
    result: Dict[str, Dict[str, Any]] = {}

    for strategy in ["fixed_size", "structure"]:
        index_file = index_dir / f"index_{strategy}.faiss"
        if not index_file.exists():
            continue

        hits = 0
        total_score = 0.0
        scores_list: List[float] = []
        per_query: List[Dict[str, Any]] = []

        for tq in TEST_QUERIES:
            try:
                results = search(
                    query=tq["query"],
                    top_k=top_k,
                    strategy=strategy,
                    index_dir=index_dir,
                )
            except Exception as e:
                per_query.append({"query": tq["query"], "error": str(e)})
                continue

            # Проверяем, есть ли ожидаемый source в top-k
            found_sources = [r.chunk.metadata.get("source", "") for r in results]
            expected = tq["expected_sources"]
            hit = any(
                any(exp in src for src in found_sources)
                for exp in expected
            )
            if hit:
                hits += 1

            query_scores = [r.score for r in results]
            avg_score = statistics.mean(query_scores) if query_scores else 0
            total_score += avg_score
            scores_list.extend(query_scores)

            per_query.append({
                "query": tq["query"][:50],
                "hit": hit,
                "avg_score": round(avg_score, 4),
                "top_source": found_sources[0] if found_sources else "N/A",
            })

        n = len(TEST_QUERIES)
        result[strategy] = {
            "precision_at_k": hits / n if n else 0,
            "hits": hits,
            "total_queries": n,
            "avg_similarity": statistics.mean(scores_list) if scores_list else 0,
            "per_query": per_query,
        }

    return result


def generate_report(
    chunk_stats: Dict[str, Dict[str, Any]],
    retrieval_stats: Dict[str, Dict[str, Any]],
    index_dir: Path,
) -> str:
    """Генерирует Markdown-отчёт сравнения стратегий."""
    lines = [
        "# Сравнение стратегий chunking",
        "",
        "## 1. Статистика чанков",
        "",
        "| Метрика | fixed_size | structure |",
        "|---------|-----------|-----------|",
    ]

    fs = chunk_stats.get("fixed_size", {})
    st = chunk_stats.get("structure", {})

    metrics = [
        ("Кол-во чанков", "chunk_count", "d"),
        ("Avg токенов", "avg_tokens", ".1f"),
        ("Median токенов", "median_tokens", ".1f"),
        ("Min токенов", "min_tokens", "d"),
        ("Max токенов", "max_tokens", "d"),
        ("Std dev токенов", "stdev_tokens", ".1f"),
        ("Всего токенов", "total_tokens", ",d"),
        ("Всего символов", "total_chars", ",d"),
        ("Источников", "sources_count", "d"),
        ("Секций", "sections_count", "d"),
    ]

    for label, key, fmt in metrics:
        fs_val = fs.get(key, 0)
        st_val = st.get(key, 0)
        lines.append(f"| {label} | {fs_val:{fmt}} | {st_val:{fmt}} |")

    lines.extend([
        "",
        "## 2. Retrieval Precision@5",
        "",
        "| Метрика | fixed_size | structure |",
        "|---------|-----------|-----------|",
    ])

    fs_r = retrieval_stats.get("fixed_size", {})
    st_r = retrieval_stats.get("structure", {})

    lines.append(
        f"| Precision@5 | {fs_r.get('precision_at_k', 0):.1%} "
        f"| {st_r.get('precision_at_k', 0):.1%} |"
    )
    lines.append(
        f"| Hits / Total | {fs_r.get('hits', 0)}/{fs_r.get('total_queries', 0)} "
        f"| {st_r.get('hits', 0)}/{st_r.get('total_queries', 0)} |"
    )
    lines.append(
        f"| Avg similarity | {fs_r.get('avg_similarity', 0):.4f} "
        f"| {st_r.get('avg_similarity', 0):.4f} |"
    )

    # Per-query details
    lines.extend(["", "## 3. Детализация по запросам", ""])

    for strategy in ["fixed_size", "structure"]:
        r = retrieval_stats.get(strategy, {})
        per_query = r.get("per_query", [])
        if not per_query:
            continue

        lines.append(f"### {strategy}")
        lines.append("")
        lines.append("| # | Запрос | Hit | Score | Top Source |")
        lines.append("|---|--------|-----|-------|------------|")

        for i, pq in enumerate(per_query, 1):
            hit_mark = "✅" if pq.get("hit") else "❌"
            lines.append(
                f"| {i} | {pq.get('query', '')} | {hit_mark} "
                f"| {pq.get('avg_score', 0):.4f} "
                f"| {pq.get('top_source', 'N/A')} |"
            )
        lines.append("")

    # Analysis
    lines.extend([
        "## 4. Анализ",
        "",
    ])

    if fs and st:
        lines.append("### Fixed-size chunking:")
        lines.append(f"- Создаёт **{fs.get('chunk_count', 0)}** чанков "
                     f"стабильного размера (~{fs.get('avg_tokens', 0):.0f} токенов)")
        lines.append(f"- Стандартное отклонение: {fs.get('stdev_tokens', 0):.1f} "
                     "— **равномерные** чанки")
        lines.append(f"- Не учитывает структуру документа (section = N/A)")
        lines.append("")

        lines.append("### Structure-based chunking:")
        lines.append(f"- Создаёт **{st.get('chunk_count', 0)}** чанков "
                     f"с переменным размером ({st.get('min_tokens', 0)}–"
                     f"{st.get('max_tokens', 0)} токенов)")
        lines.append(f"- Стандартное отклонение: {st.get('stdev_tokens', 0):.1f} "
                     "— **неравномерные** но семантически целостные чанки")
        lines.append(f"- Сохраняет структуру: {st.get('sections_count', 0)} уникальных секций")
        lines.append("")

    if fs_r and st_r:
        fs_prec = fs_r.get("precision_at_k", 0)
        st_prec = st_r.get("precision_at_k", 0)
        if st_prec > fs_prec:
            winner = "Structure-based"
            reason = ("семантическая целостность секций повышает релевантность "
                     "при поиске по темам")
        elif fs_prec > st_prec:
            winner = "Fixed-size"
            reason = ("равномерные чанки обеспечивают более стабильное "
                     "покрытие документов")
        else:
            winner = "Обе стратегии"
            reason = "показали одинаковую точность"

        lines.append(f"### Вывод: **{winner}** показывает лучшую retrieval "
                     f"precision ({reason}).")

    return "\n".join(lines)


def main():
    """Запускает сравнение стратегий и генерирует отчёт."""
    print(f"{'='*60}")
    print("Сравнение стратегий chunking")
    print(f"{'='*60}")

    index_dir = DEFAULT_INDEX_DIR

    # 1. Chunk statistics
    print("\n📊 Вычисление статистики чанков...")
    chunk_stats = compute_chunking_stats(index_dir)

    for strategy, stats in chunk_stats.items():
        print(f"\n  [{strategy}]")
        print(f"    Чанков: {stats['chunk_count']}")
        print(f"    Avg: {stats['avg_tokens']:.1f} tok, "
              f"Median: {stats['median_tokens']:.1f} tok")
        print(f"    Min/Max: {stats['min_tokens']}/{stats['max_tokens']} tok")
        print(f"    StdDev: {stats['stdev_tokens']:.1f}")

    # 2. Retrieval metrics
    print(f"\n🔍 Тестирование retrieval ({len(TEST_QUERIES)} запросов)...")
    retrieval_stats = compute_retrieval_metrics(index_dir, top_k=5)

    for strategy, stats in retrieval_stats.items():
        print(f"\n  [{strategy}]")
        print(f"    Precision@5: {stats['precision_at_k']:.1%} "
              f"({stats['hits']}/{stats['total_queries']})")
        print(f"    Avg similarity: {stats['avg_similarity']:.4f}")

    # 3. Generate report
    report = generate_report(chunk_stats, retrieval_stats, index_dir)
    report_path = index_dir / "comparison_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n📄 Отчёт сохранён: {report_path}")

    print(f"\n{'='*60}")
    print("✅ Сравнение завершено!")


if __name__ == "__main__":
    main()
