"""
Сравнение ответов RagAgent в двух режимах (с RAG / без RAG).

Содержит:
    - структуру результата одного сравнения
    - агрегацию метрик по всем вопросам
    - форматированный вывод side-by-side
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

MODE_KEYS = ["baseline", "rewrite_only", "rerank_only", "combined"]


# ── Одиночное сравнение ───────────────────────────────────────────────────────

def build_comparison(
    compare_result: Dict[str, Any],
    question_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Формирует структуру одного сравнения с оценкой качества.

    Args:
        compare_result: возвращаемый RagAgent.compare()
        question_spec:  запись из benchmark.CONTROL_QUESTIONS

    Returns:
        dict со структурой:
        {
          question_id, question,
          no_rag: {answer, tokens_total, elapsed_ms},
          rag:    {answer, sources, chunks_used, tokens_total, elapsed_ms},
          evaluation: {
            keyword_hits_no_rag, keyword_hits_rag, total_keywords,
            source_hits, total_sources, rag_wins
          }
        }
    """
    keywords = [kw.lower() for kw in question_spec["expected_keywords"]]
    expected_sources = question_spec["expected_sources"]

    no_rag_data = compare_result["no_rag"]
    rag_data = compare_result["rag"]

    no_rag_answer = (no_rag_data.get("answer") or "").lower()
    rag_answer = (rag_data.get("answer") or "").lower()
    rag_sources = [s.lower() for s in rag_data.get("sources", [])]

    kw_hits_no_rag = sum(1 for kw in keywords if kw in no_rag_answer)
    kw_hits_rag = sum(1 for kw in keywords if kw in rag_answer)

    source_hits = 0
    for expected in expected_sources:
        expected_lower = expected.lower()
        if any(expected_lower in s or s in expected_lower for s in rag_sources):
            source_hits += 1

    return {
        "question_id": question_spec["id"],
        "question": question_spec["question"],
        "no_rag": {
            "answer": no_rag_data.get("answer", ""),
            "tokens_total": no_rag_data.get("token_usage", {}).get("total_tokens", 0),
            "elapsed_ms": no_rag_data.get("elapsed_ms", 0.0),
        },
        "rag": {
            "answer": rag_data.get("answer", ""),
            "sources": rag_data.get("sources", []),
            "chunks_used": rag_data.get("chunks_used", 0),
            "tokens_total": rag_data.get("token_usage", {}).get("total_tokens", 0),
            "elapsed_ms": rag_data.get("elapsed_ms", 0.0),
        },
        "evaluation": {
            "keyword_hits_no_rag": kw_hits_no_rag,
            "keyword_hits_rag": kw_hits_rag,
            "total_keywords": len(keywords),
            "source_hits": source_hits,
            "total_sources": len(expected_sources),
            "rag_wins": kw_hits_rag > kw_hits_no_rag,
        },
    }


def build_modes_comparison(
    modes_result: Dict[str, Any],
    question_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Формирует структуру сравнения 4 RAG-режимов.

    Ожидаемый формат modes_result:
        {
          "question": "...",
          "baseline": {...},
          "rewrite_only": {...},
          "rerank_only": {...},
          "combined": {...}
        }
    """
    keywords = [kw.lower() for kw in question_spec["expected_keywords"]]
    expected_sources = question_spec["expected_sources"]

    mode_data: Dict[str, Any] = {}
    for mode in MODE_KEYS:
        data = modes_result.get(mode, {})
        answer = (data.get("answer") or "").lower()
        sources = [s.lower() for s in data.get("sources", [])]

        kw_hits = sum(1 for kw in keywords if kw in answer)
        source_hits = 0
        for expected in expected_sources:
            expected_lower = expected.lower()
            if any(expected_lower in s or s in expected_lower for s in sources):
                source_hits += 1

        mode_data[mode] = {
            "answer": data.get("answer", ""),
            "sources": data.get("sources", []),
            "chunks_used": data.get("chunks_used", 0),
            "tokens_total": data.get("token_usage", {}).get("total_tokens", 0),
            "elapsed_ms": data.get("elapsed_ms", 0.0),
            "keyword_hits": kw_hits,
            "source_hits": source_hits,
        }

    baseline_hits = mode_data["baseline"]["keyword_hits"]
    wins_vs_baseline = {
        mode: mode_data[mode]["keyword_hits"] > baseline_hits
        for mode in MODE_KEYS
        if mode != "baseline"
    }

    return {
        "question_id": question_spec["id"],
        "question": question_spec["question"],
        "total_keywords": len(keywords),
        "total_sources": len(expected_sources),
        "modes": mode_data,
        "wins_vs_baseline": wins_vs_baseline,
    }


# ── Агрегация метрик ──────────────────────────────────────────────────────────

def compare_all(comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Агрегирует метрики по всем сравнениям.

    Args:
        comparisons: список результатов build_comparison()

    Returns:
        dict с агрегированными метриками
    """
    if not comparisons:
        return {}

    total = len(comparisons)
    rag_wins = sum(1 for c in comparisons if c["evaluation"]["rag_wins"])

    avg_kw_nr = sum(c["evaluation"]["keyword_hits_no_rag"] for c in comparisons) / total
    avg_kw_r = sum(c["evaluation"]["keyword_hits_rag"] for c in comparisons) / total
    avg_kw_t = sum(c["evaluation"]["total_keywords"] for c in comparisons) / total

    src_prec_list = [
        c["evaluation"]["source_hits"] / c["evaluation"]["total_sources"]
        for c in comparisons
        if c["evaluation"]["total_sources"] > 0
    ]
    avg_src_prec = sum(src_prec_list) / len(src_prec_list) if src_prec_list else 0.0

    avg_tokens_nr = sum(c["no_rag"]["tokens_total"] for c in comparisons) / total
    avg_tokens_r = sum(c["rag"]["tokens_total"] for c in comparisons) / total
    avg_lat_nr = sum(c["no_rag"]["elapsed_ms"] for c in comparisons) / total
    avg_lat_r = sum(c["rag"]["elapsed_ms"] for c in comparisons) / total

    return {
        "total_questions": total,
        "rag_wins": rag_wins,
        "rag_win_rate": rag_wins / total,
        "avg_keyword_hit_no_rag": avg_kw_nr,
        "avg_keyword_hit_rag": avg_kw_r,
        "avg_total_keywords": avg_kw_t,
        "avg_source_precision": avg_src_prec,
        "avg_tokens_no_rag": avg_tokens_nr,
        "avg_tokens_rag": avg_tokens_r,
        "avg_latency_no_rag_ms": avg_lat_nr,
        "avg_latency_rag_ms": avg_lat_r,
    }


def compare_modes_all(mode_comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Агрегирует метрики для 4 режимов относительно baseline."""
    if not mode_comparisons:
        return {}

    total = len(mode_comparisons)
    modes_agg: Dict[str, Dict[str, float]] = {}

    for mode in MODE_KEYS:
        kw_hits = [c["modes"][mode]["keyword_hits"] for c in mode_comparisons]
        src_hits = [c["modes"][mode]["source_hits"] for c in mode_comparisons]
        tokens = [c["modes"][mode]["tokens_total"] for c in mode_comparisons]
        lat = [c["modes"][mode]["elapsed_ms"] for c in mode_comparisons]
        chunks = [c["modes"][mode]["chunks_used"] for c in mode_comparisons]

        src_prec = []
        for c in mode_comparisons:
            total_sources = c.get("total_sources", 0)
            if total_sources > 0:
                src_prec.append(c["modes"][mode]["source_hits"] / total_sources)

        modes_agg[mode] = {
            "avg_keyword_hits": sum(kw_hits) / total,
            "avg_source_precision": sum(src_prec) / len(src_prec) if src_prec else 0.0,
            "avg_tokens": sum(tokens) / total,
            "avg_latency_ms": sum(lat) / total,
            "avg_chunks_used": sum(chunks) / total,
        }

    wins = {
        "rewrite_only": sum(1 for c in mode_comparisons if c["wins_vs_baseline"]["rewrite_only"]),
        "rerank_only": sum(1 for c in mode_comparisons if c["wins_vs_baseline"]["rerank_only"]),
        "combined": sum(1 for c in mode_comparisons if c["wins_vs_baseline"]["combined"]),
    }

    return {
        "total_questions": total,
        "avg_total_keywords": sum(c["total_keywords"] for c in mode_comparisons) / total,
        "modes": modes_agg,
        "wins_vs_baseline": {
            "rewrite_only": {"count": wins["rewrite_only"], "rate": wins["rewrite_only"] / total},
            "rerank_only": {"count": wins["rerank_only"], "rate": wins["rerank_only"] / total},
            "combined": {"count": wins["combined"], "rate": wins["combined"] / total},
        },
    }


# ── Форматированный вывод ─────────────────────────────────────────────────────

_SEP = "═" * 80


def _wrap(text: str, indent: str = "  ") -> str:
    """Оборачивает текст ответа с отступом, сохраняя оригинальные переводы строк."""
    lines = text.strip().splitlines()
    return "\n".join(f"{indent}{line}" if line.strip() else "" for line in lines)


def print_final_summary(comparisons: List[Dict[str, Any]]) -> None:
    """Печатает итоговую таблицу метрик по всем сравнениям."""
    if not comparisons:
        return
    agg = compare_all(comparisons)
    win_pct = agg["rag_win_rate"] * 100
    print("\n" + _SEP)
    print(f"  ИТОГ: RAG побеждает в {agg['rag_wins']}/{agg['total_questions']} "
          f"вопросах ({win_pct:.0f}%)")
    print(f"  Avg keyword precision:  "
          f"no_rag={agg['avg_keyword_hit_no_rag']:.1f}/{agg['avg_total_keywords']:.1f}  "
          f"→  rag={agg['avg_keyword_hit_rag']:.1f}/{agg['avg_total_keywords']:.1f}")
    print(f"  Avg source precision:   {agg['avg_source_precision'] * 100:.0f}%")
    print(f"  Avg tokens:             "
          f"no_rag={agg['avg_tokens_no_rag']:.0f}  "
          f"rag={agg['avg_tokens_rag']:.0f}  "
          f"(overhead ×{agg['avg_tokens_rag'] / max(agg['avg_tokens_no_rag'], 1):.1f})")
    print(f"  Avg latency:            "
          f"no_rag={agg['avg_latency_no_rag_ms']:.0f}ms  "
          f"rag={agg['avg_latency_rag_ms']:.0f}ms")
    print(_SEP)


def print_comparison_report(comparisons: List[Dict[str, Any]]) -> None:
    """Выводит полные тексты ответов для каждого вопроса."""
    for c in comparisons:
        q_id = c["question_id"]
        ev = c["evaluation"]
        no_rag = c["no_rag"]
        rag = c["rag"]

        win_mark = "✅ RAG WIN" if ev["rag_wins"] else "─ ничья/no_rag"
        kw_nr = f"{ev['keyword_hits_no_rag']}/{ev['total_keywords']}"
        kw_r = f"{ev['keyword_hits_rag']}/{ev['total_keywords']}"
        src_mark = f"{ev['source_hits']}/{ev['total_sources']}"
        sources_str = "\n  ".join(rag["sources"]) if rag["sources"] else "—"

        print(f"\n{_SEP}")
        print(f"  Q{q_id:02d}. {c['question']}")
        print(_SEP)

        # ── БЕЗ RAG ──
        print(f"\n  ▶ БЕЗ RAG  │ keywords: {kw_nr}  │  "
              f"tokens: {no_rag['tokens_total']}  │  latency: {no_rag['elapsed_ms']:.0f}ms")
        print("  " + "─" * 76)
        print(_wrap(no_rag["answer"]))

        # ── С RAG ──
        print(f"\n  ▶ С RAG    │ keywords: {kw_r}  │  "
              f"tokens: {rag['tokens_total']}  │  latency: {rag['elapsed_ms']:.0f}ms  │  "
              f"chunks: {rag['chunks_used']}  │  {win_mark}")
        print(f"  Источники: {sources_str}")
        print("  " + "─" * 76)
        print(_wrap(rag["answer"]))

        print(f"\n  [src precision: {src_mark}]")

    # Итоговые метрики
    print_final_summary(comparisons)


def print_modes_summary(mode_comparisons: List[Dict[str, Any]]) -> None:
    """Печатает итог по 4 режимам RAG."""
    agg = compare_modes_all(mode_comparisons)
    if not agg:
        return

    print("\n" + _SEP)
    print("  ИТОГ: 4-режимное сравнение RAG")
    print(_SEP)

    avg_total_kw = agg["avg_total_keywords"]
    for mode in MODE_KEYS:
        m = agg["modes"][mode]
        print(
            f"  {mode:12} | keywords={m['avg_keyword_hits']:.1f}/{avg_total_kw:.1f} "
            f"| src_precision={m['avg_source_precision'] * 100:.0f}% "
            f"| tokens={m['avg_tokens']:.0f} "
            f"| latency={m['avg_latency_ms']:.0f}ms"
        )

    wins = agg["wins_vs_baseline"]
    print("  " + "─" * 78)
    print(
        "  Победы над baseline: "
        f"rewrite_only={wins['rewrite_only']['count']}/{agg['total_questions']} "
        f"({wins['rewrite_only']['rate'] * 100:.0f}%), "
        f"rerank_only={wins['rerank_only']['count']}/{agg['total_questions']} "
        f"({wins['rerank_only']['rate'] * 100:.0f}%), "
        f"combined={wins['combined']['count']}/{agg['total_questions']} "
        f"({wins['combined']['rate'] * 100:.0f}%)"
    )
    print(_SEP)


def print_modes_comparison_report(mode_comparisons: List[Dict[str, Any]]) -> None:
    """Печатает наглядное сравнение 4 режимов по каждому вопросу."""
    if not mode_comparisons:
        return

    for c in mode_comparisons:
        q_id = c["question_id"]
        total_keywords = c["total_keywords"]
        total_sources = c["total_sources"]
        baseline_hits = c["modes"]["baseline"]["keyword_hits"]
        combined_hits = c["modes"]["combined"]["keyword_hits"]
        combined_delta = combined_hits - baseline_hits
        delta_str = f"+{combined_delta}" if combined_delta > 0 else str(combined_delta)

        print(f"\n{_SEP}")
        print(f"  Q{q_id:02d}. {c['question']}")
        print(_SEP)
        print(
            "  Режим         | keywords | source hits | chunks | tokens | latency | vs baseline"
        )
        print("  " + "─" * 78)

        for mode in MODE_KEYS:
            mode_data = c["modes"][mode]
            if mode == "baseline":
                verdict = "base"
            else:
                verdict = "win" if c["wins_vs_baseline"][mode] else "no win"

            print(
                f"  {mode:12} | "
                f"{mode_data['keyword_hits']}/{total_keywords:>8} | "
                f"{mode_data['source_hits']}/{total_sources:>11} | "
                f"{mode_data['chunks_used']:>6} | "
                f"{mode_data['tokens_total']:>6} | "
                f"{mode_data['elapsed_ms']:>7.0f}ms | "
                f"{verdict}"
            )

        print("  " + "─" * 78)
        print(
            f"  Сравнение по задаче: baseline={baseline_hits}/{total_keywords}, "
            f"combined={combined_hits}/{total_keywords} ({delta_str})"
        )

    print_modes_summary(mode_comparisons)
