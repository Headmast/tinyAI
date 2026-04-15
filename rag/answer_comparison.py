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


# ── Форматированный вывод ─────────────────────────────────────────────────────

_W = 70  # ширина блока ответа


def _truncate(text: str, max_chars: int = 300) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def print_comparison_report(comparisons: List[Dict[str, Any]]) -> None:
    """Выводит side-by-side сравнение для каждого вопроса."""
    for c in comparisons:
        q_id = c["question_id"]
        question_short = c["question"][:65] + ("…" if len(c["question"]) > 65 else "")
        ev = c["evaluation"]
        no_rag = c["no_rag"]
        rag = c["rag"]

        win_mark = "✅ RAG WIN" if ev["rag_wins"] else "  ─ draw/no_rag"
        kw_nr = f"{ev['keyword_hits_no_rag']}/{ev['total_keywords']}"
        kw_r = f"{ev['keyword_hits_rag']}/{ev['total_keywords']}"
        src_mark = f"{ev['source_hits']}/{ev['total_sources']}"

        print(f"\n╔══ Q{q_id:02d}: {question_short:{_W - 10}}")
        print("╠" + "─" * (_W + 12))

        # NO RAG block
        answer_nr = _truncate(no_rag["answer"])
        print(f"║ NO RAG │ {answer_nr}")
        print(f"║        │ keywords: {kw_nr}  "
              f"tokens: {no_rag['tokens_total']}  "
              f"latency: {no_rag['elapsed_ms']:.0f}ms")

        print("╠" + "─" * (_W + 12))

        # RAG block
        answer_r = _truncate(rag["answer"])
        sources_str = ", ".join(rag["sources"]) if rag["sources"] else "—"
        print(f"║ RAG    │ {answer_r}")
        print(f"║        │ keywords: {kw_r}  "
              f"tokens: {rag['tokens_total']}  "
              f"latency: {rag['elapsed_ms']:.0f}ms  "
              f"chunks: {rag['chunks_used']}  {win_mark}")
        print(f"║        │ sources: {sources_str}")
        print(f"╚══ src hits: {src_mark}")

    # Итоговые метрики
    if comparisons:
        agg = compare_all(comparisons)
        win_pct = agg["rag_win_rate"] * 100
        print("\n" + "═" * (_W + 14))
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
        print("═" * (_W + 14))
