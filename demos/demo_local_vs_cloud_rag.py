"""
Демо: сравнение локальной и облачной RAG-системы.

Прогоняет контрольные вопросы через:
    Mode A — Cloud embeddings + Cloud LLM  (baseline)
    Mode B — Local embeddings + Local LLM  (fully local, Ollama)

Оценивает качество, скорость и стабильность.

Запуск:
    # Полностью локальный бенчмарк (только Mode B):
    python demos/demo_local_vs_cloud_rag.py --local-only

    # Сравнение cloud vs local (нужны оба API-ключа и Ollama):
    python demos/demo_local_vs_cloud_rag.py

    # Быстрый режим (3 вопроса вместо 10):
    python demos/demo_local_vs_cloud_rag.py --quick

Предварительные шаги:
    1. ./run_ollama.sh           — запуск Ollama + скачивание моделей
    2. python -m rag.indexer --local  — индексация документов локально
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_agent import RagAgent
from rag.benchmark import CONTROL_QUESTIONS

PROJECT_ROOT = Path(__file__).parent.parent

_SEP = "═" * 72
_SUB = "─" * 72

# Вопросы для быстрого режима
QUICK_IDS = {1, 3, 10}

# Модели
OLLAMA_MODEL = "qwen3:8b"
CLOUD_MODEL = "zai-org/GLM-4.7"


def _keyword_hits(answer: str, keywords: List[str]) -> int:
    answer_lower = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in answer_lower)


def _source_hits(sources: List[str], expected: List[str]) -> int:
    hits = 0
    for exp in expected:
        exp_lower = exp.lower()
        if any(exp_lower in s.lower() or s.lower() in exp_lower for s in sources):
            hits += 1
    return hits


def run_single(
    agent: RagAgent,
    spec: Dict[str, Any],
    label: str,
    stream: bool = True,
) -> Dict[str, Any]:
    """Запускает один вопрос через RAG-агент, возвращает результаты."""
    question = spec["question"]

    print(f"\n  ▶ {label}")
    print(f"  {_SUB}")

    t0 = time.monotonic()
    result = agent.ask_with_rag(question, stream=stream)
    total_ms = (time.monotonic() - t0) * 1000

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    chunks = result.get("chunks_used", 0)
    elapsed = result.get("elapsed_ms", total_ms)

    kw_hits = _keyword_hits(answer, spec["expected_keywords"])
    src_hits = _source_hits(sources, spec["expected_sources"])

    print(f"\n  ⏱  {elapsed:.0f}ms  │  chunks: {chunks}")
    print(f"  Keywords: {kw_hits}/{len(spec['expected_keywords'])}  │  "
          f"Sources: {src_hits}/{len(spec['expected_sources'])}")

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": chunks,
        "elapsed_ms": round(elapsed, 1),
        "keyword_hits": kw_hits,
        "total_keywords": len(spec["expected_keywords"]),
        "source_hits": src_hits,
        "total_sources": len(spec["expected_sources"]),
        "token_usage": result.get("token_usage", {}),
    }


def run_stability_check(
    agent: RagAgent,
    spec: Dict[str, Any],
    runs: int = 3,
) -> Dict[str, Any]:
    """Запускает вопрос N раз, измеряет стабильность ответов."""
    answers = []
    latencies = []
    kw_counts = []

    for i in range(runs):
        result = agent.ask_with_rag(spec["question"], stream=False)
        answers.append(result.get("answer", ""))
        latencies.append(result.get("elapsed_ms", 0))
        kw_counts.append(
            _keyword_hits(result.get("answer", ""), spec["expected_keywords"])
        )

    # Считаем overlap ключевых слов между прогонами
    kw_variance = max(kw_counts) - min(kw_counts) if kw_counts else 0

    # Средняя задержка и разброс
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    latency_spread = max(latencies) - min(latencies) if latencies else 0

    return {
        "runs": runs,
        "avg_latency_ms": round(avg_latency, 1),
        "latency_spread_ms": round(latency_spread, 1),
        "keyword_counts": kw_counts,
        "keyword_variance": kw_variance,
    }


def create_agent(provider: str, verbose: bool = False) -> Optional[RagAgent]:
    """Создаёт RagAgent для указанного провайдера."""
    try:
        if provider == "ollama":
            return RagAgent(
                model=OLLAMA_MODEL,
                provider="ollama",
                index_dir=PROJECT_ROOT / "rag_data" / "local",
                verbose=verbose,
                top_k=5,
                top_k_before=7,
                top_k_after=5,
                enable_query_rewrite=True,
                enable_rerank=True,
            )
        else:
            return RagAgent(
                model=CLOUD_MODEL,
                provider="cloud",
                index_dir=PROJECT_ROOT / "rag_data",
                verbose=verbose,
                top_k=5,
                top_k_before=7,
                top_k_after=5,
                enable_query_rewrite=True,
                enable_rerank=True,
            )
    except Exception as e:
        print(f"  ⚠ Не удалось создать {provider} агента: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Сравнение локальной и облачной RAG-системы"
    )
    parser.add_argument(
        "--local-only", action="store_true",
        help="Запустить только локальный режим (не нужны облачные API-ключи)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Быстрый режим: 3 вопроса вместо 10",
    )
    parser.add_argument(
        "--stability", action="store_true",
        help="Проверка стабильности: каждый вопрос прогоняется 3 раза",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Отключить стриминг (для чистых метрик)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Подробный вывод",
    )
    args = parser.parse_args()

    stream = not args.no_stream
    specs = CONTROL_QUESTIONS
    if args.quick:
        specs = [q for q in specs if q["id"] in QUICK_IDS]

    print(f"\n{_SEP}")
    print(f"  🔬 Сравнение RAG: локальная vs облачная модель")
    print(f"  Вопросов: {len(specs)}  │  Стриминг: {'да' if stream else 'нет'}")
    print(f"{_SEP}")

    # ── Создаём агентов ──
    agents: Dict[str, Optional[RagAgent]] = {}

    if not args.local_only:
        print("\n📡 Инициализация облачного агента...")
        agents["cloud"] = create_agent("cloud", verbose=args.verbose)

    print("\n🏠 Инициализация локального агента (Ollama)...")
    agents["local"] = create_agent("ollama", verbose=args.verbose)

    active = {k: v for k, v in agents.items() if v is not None}
    if not active:
        print("\n❌ Ни один агент не доступен. Проверьте настройки.")
        sys.exit(1)

    # ── Основной бенчмарк ──
    all_results: Dict[str, List[Dict[str, Any]]] = {k: [] for k in active}

    for spec in specs:
        q_id = spec["id"]
        question = spec["question"]

        print(f"\n{_SEP}")
        print(f"  Q{q_id:02d}.")
        for line in question.splitlines():
            print(f"  {line}")
        print(_SEP)

        for name, agent in active.items():
            label_emoji = "📡" if name == "cloud" else "🏠"
            label = f"{label_emoji} {name.upper()}"
            result = run_single(agent, spec, label, stream=stream)
            result["question_id"] = q_id
            all_results[name].append(result)

    # ── Стабильность (опционально) ──
    stability_results: Dict[str, List[Dict[str, Any]]] = {}
    if args.stability:
        print(f"\n\n{_SEP}")
        print(f"  🔄 Проверка стабильности (3 прогона × {len(specs)} вопросов)")
        print(f"{_SEP}")

        stability_spec = specs[:3] if not args.quick else specs[:2]

        for name, agent in active.items():
            stability_results[name] = []
            label_emoji = "📡" if name == "cloud" else "🏠"
            print(f"\n  {label_emoji} {name.upper()}")

            for spec in stability_spec:
                print(f"    Q{spec['id']:02d}: ", end="", flush=True)
                stab = run_stability_check(agent, spec, runs=3)
                stab["question_id"] = spec["id"]
                stability_results[name].append(stab)
                print(
                    f"avg={stab['avg_latency_ms']:.0f}ms  "
                    f"spread={stab['latency_spread_ms']:.0f}ms  "
                    f"kw_var={stab['keyword_variance']}"
                )

    # ── Итоговая таблица ──
    print(f"\n\n{_SEP}")
    print(f"  📊 ИТОГОВОЕ СРАВНЕНИЕ")
    print(f"{_SEP}\n")

    header = f"{'Метрика':<30}"
    for name in active:
        header += f"  {'📡 Cloud' if name == 'cloud' else '🏠 Local':>14}"
    print(header)
    print("─" * (30 + 16 * len(active)))

    for name in active:
        results = all_results[name]
        total_kw = sum(r["keyword_hits"] for r in results)
        total_kw_max = sum(r["total_keywords"] for r in results)
        total_src = sum(r["source_hits"] for r in results)
        total_src_max = sum(r["total_sources"] for r in results)
        avg_latency = sum(r["elapsed_ms"] for r in results) / len(results)

        metrics = {
            "Keyword precision": f"{total_kw}/{total_kw_max} ({100*total_kw/total_kw_max:.0f}%)" if total_kw_max else "N/A",
            "Source hits": f"{total_src}/{total_src_max} ({100*total_src/total_src_max:.0f}%)" if total_src_max else "N/A",
            "Avg latency (ms)": f"{avg_latency:.0f}",
            "Total questions": f"{len(results)}",
        }

        if name == list(active.keys())[0]:
            for metric_name, value in metrics.items():
                print(f"  {metric_name:<28}  {value:>14}", end="")
                # Placeholder для второй колонки — будет перезаписана ниже
                break

    # Упрощённая таблица для сравнения
    print()
    for metric_name in ["Keyword precision", "Source hits", "Avg latency (ms)"]:
        row = f"  {metric_name:<28}"
        for name in active:
            results = all_results[name]
            if metric_name == "Keyword precision":
                total = sum(r["keyword_hits"] for r in results)
                total_max = sum(r["total_keywords"] for r in results)
                val = f"{total}/{total_max} ({100*total/total_max:.0f}%)" if total_max else "N/A"
            elif metric_name == "Source hits":
                total = sum(r["source_hits"] for r in results)
                total_max = sum(r["total_sources"] for r in results)
                val = f"{total}/{total_max} ({100*total/total_max:.0f}%)" if total_max else "N/A"
            elif metric_name == "Avg latency (ms)":
                avg = sum(r["elapsed_ms"] for r in results) / len(results)
                val = f"{avg:.0f}"
            else:
                val = "—"
            row += f"  {val:>20}"
        print(row)

    # Стоимость
    cost_row = f"  {'Cost per query':<28}"
    for name in active:
        if name == "local":
            cost_row += f"  {'$0 (free)':>20}"
        else:
            cost_row += f"  {'~$0.002':>20}"
    print(cost_row)

    # ── Стабильность ──
    if stability_results:
        print(f"\n  {'Stability (kw variance)':<28}", end="")
        for name in active:
            if name in stability_results:
                avg_var = sum(
                    s["keyword_variance"] for s in stability_results[name]
                ) / len(stability_results[name])
                print(f"  {avg_var:>20.1f}", end="")
        print()

        print(f"  {'Stability (latency spread)':<28}", end="")
        for name in active:
            if name in stability_results:
                avg_spread = sum(
                    s["latency_spread_ms"] for s in stability_results[name]
                ) / len(stability_results[name])
                print(f"  {avg_spread:>17.0f}ms", end="")
        print()

    # ── Сохранение результатов ──
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = logs_dir / f"local_vs_cloud_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "questions_count": len(specs),
        "modes": list(active.keys()),
        "results": {k: v for k, v in all_results.items()},
    }
    if stability_results:
        report["stability"] = stability_results

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n  💾 Отчёт сохранён: {report_path}")
    print()


if __name__ == "__main__":
    main()
