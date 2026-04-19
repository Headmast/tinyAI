"""
benchmark_v2 — Улучшенный бенчмарк для RAG-системы с метриками, 
вдохновлёнными RAGAS (arxiv:2309.15217).

Метрики:
    1. semantic_similarity  — косинусное сходство embedding ответа и reference
    2. faithfulness_proxy   — доля предложений ответа, подтверждённых контекстом (n-gram overlap)
    3. context_utilization  — доля чанков, реально использованных в ответе
    4. keyword_precision    — совпадение ожидаемых ключевых слов (из benchmark.py)
    5. answer_completeness  — доля покрытых ключевых слов (recall)

Использование:
    from rag.benchmark_v2 import evaluate_v2, run_benchmark_v2
    
    # Одиночная оценка
    scores = evaluate_v2(answer, context_chunks, reference_keywords)
    
    # Полный бенчмарк на 10 вопросах
    results = run_benchmark_v2(agent)
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.benchmark import CONTROL_QUESTIONS


# ── Утилиты ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Простая токенизация: lowercase, split по не-буквенным символам."""
    return re.findall(r"[a-zа-яё0-9_]+", text.lower())


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Возвращает n-граммы из списка токенов."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство двух векторов."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _split_sentences(text: str) -> List[str]:
    """Разбивает текст на предложения."""
    sentences = re.split(r"[.!?]\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ── Метрики ───────────────────────────────────────────────────────────────────

def semantic_similarity(
    answer_embedding: Optional[np.ndarray],
    reference_embedding: Optional[np.ndarray],
) -> float:
    """
    Косинусное сходство между embedding ответа и reference.
    Требует предварительно вычисленные embeddings.
    Возвращает 0.0, если embeddings не предоставлены.
    """
    if answer_embedding is None or reference_embedding is None:
        return 0.0
    return _cosine_similarity(answer_embedding, reference_embedding)


def faithfulness_proxy(
    answer: str,
    context_chunks: List[str],
    ngram_size: int = 3,
    threshold: float = 0.5,
) -> float:
    """
    Proxy-метрика faithfulness (без LLM-as-judge).
    
    Для каждого предложения ответа считает долю n-грамм,
    встречающихся в контексте. Предложение считается 
    «подтверждённым», если overlap >= threshold.
    
    Возвращает долю подтверждённых предложений [0.0, 1.0].
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    # Собираем все n-граммы контекста
    context_ngrams: set[tuple] = set()
    for chunk in context_chunks:
        tokens = _tokenize(chunk)
        for ng in _ngrams(tokens, ngram_size):
            context_ngrams.add(ng)

    if not context_ngrams:
        return 0.0

    supported = 0
    for sent in sentences:
        tokens = _tokenize(sent)
        sent_ngrams = _ngrams(tokens, ngram_size)
        if not sent_ngrams:
            continue
        overlap = sum(1 for ng in sent_ngrams if ng in context_ngrams)
        ratio = overlap / len(sent_ngrams)
        if ratio >= threshold:
            supported += 1

    return supported / len(sentences)


def context_utilization(
    answer: str,
    context_chunks: List[str],
    min_overlap_tokens: int = 3,
) -> float:
    """
    Доля контекстных чанков, информация из которых
    реально встречается в ответе.
    
    Чанк считается «использованным», если >= min_overlap_tokens
    общих токенов с ответом.
    
    Возвращает [0.0, 1.0].
    """
    if not context_chunks:
        return 0.0

    answer_tokens = set(_tokenize(answer))
    used = 0

    for chunk in context_chunks:
        chunk_tokens = set(_tokenize(chunk))
        common = answer_tokens & chunk_tokens
        if len(common) >= min_overlap_tokens:
            used += 1

    return used / len(context_chunks)


def keyword_precision(
    answer: str,
    expected_keywords: List[str],
) -> float:
    """
    Доля ожидаемых ключевых слов, найденных в ответе.
    Регистронезависимый поиск.
    """
    if not expected_keywords:
        return 0.0

    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def answer_completeness(
    answer: str,
    expected_keywords: List[str],
) -> Dict[str, Any]:
    """
    Подробный отчёт по покрытию ключевых слов.
    """
    answer_lower = answer.lower()
    found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]

    return {
        "found": found,
        "missing": missing,
        "recall": len(found) / len(expected_keywords) if expected_keywords else 0.0,
    }


# ── Комплексная оценка ────────────────────────────────────────────────────────

def evaluate_v2(
    answer: str,
    context_chunks: List[str],
    expected_keywords: List[str],
    answer_embedding: Optional[np.ndarray] = None,
    reference_embedding: Optional[np.ndarray] = None,
    ngram_size: int = 3,
    faithfulness_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Комплексная оценка ответа RAG-системы.
    
    Returns:
        dict с метриками:
            - semantic_similarity: float [0, 1]
            - faithfulness: float [0, 1]
            - context_utilization: float [0, 1]
            - keyword_precision: float [0, 1]
            - completeness: dict {found, missing, recall}
            - composite_score: float [0, 1] — взвешенное среднее
    """
    sem_sim = semantic_similarity(answer_embedding, reference_embedding)
    faith = faithfulness_proxy(
        answer, context_chunks,
        ngram_size=ngram_size,
        threshold=faithfulness_threshold,
    )
    ctx_util = context_utilization(answer, context_chunks)
    kw_prec = keyword_precision(answer, expected_keywords)
    completeness = answer_completeness(answer, expected_keywords)

    # Взвешенное среднее (веса настраиваемые)
    weights = {
        "faithfulness": 0.30,
        "keyword_precision": 0.30,
        "context_utilization": 0.20,
        "semantic_similarity": 0.20,
    }
    composite = (
        weights["faithfulness"] * faith
        + weights["keyword_precision"] * kw_prec
        + weights["context_utilization"] * ctx_util
        + weights["semantic_similarity"] * sem_sim
    )

    return {
        "semantic_similarity": round(sem_sim, 4),
        "faithfulness": round(faith, 4),
        "context_utilization": round(ctx_util, 4),
        "keyword_precision": round(kw_prec, 4),
        "completeness": completeness,
        "composite_score": round(composite, 4),
    }


def compare_v2(
    no_rag_answer: str,
    rag_answer: str,
    context_chunks: List[str],
    expected_keywords: List[str],
    no_rag_embedding: Optional[np.ndarray] = None,
    rag_embedding: Optional[np.ndarray] = None,
    reference_embedding: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Сравнение двух режимов (no_rag vs rag) по всем метрикам.
    """
    no_rag_scores = evaluate_v2(
        answer=no_rag_answer,
        context_chunks=[],  # no context for no_rag
        expected_keywords=expected_keywords,
        answer_embedding=no_rag_embedding,
        reference_embedding=reference_embedding,
    )
    rag_scores = evaluate_v2(
        answer=rag_answer,
        context_chunks=context_chunks,
        expected_keywords=expected_keywords,
        answer_embedding=rag_embedding,
        reference_embedding=reference_embedding,
    )

    return {
        "no_rag": no_rag_scores,
        "rag": rag_scores,
        "rag_improvement": {
            k: round(rag_scores[k] - no_rag_scores[k], 4)
            for k in ["faithfulness", "keyword_precision", "context_utilization", "composite_score"]
        },
        "rag_wins": rag_scores["composite_score"] > no_rag_scores["composite_score"],
    }


# ── Отчёт ─────────────────────────────────────────────────────────────────────

def print_v2_report(results: List[Dict[str, Any]]) -> None:
    """Печатает сводный отчёт по результатам benchmark_v2."""
    if not results:
        print("Нет результатов для отчёта.")
        return

    print("\n" + "=" * 70)
    print("BENCHMARK V2 — RAGAS-inspired metrics")
    print("=" * 70)

    total = len(results)
    rag_wins = sum(1 for r in results if r.get("rag_wins", False))

    # Средние метрики
    avg_metrics: Dict[str, float] = {}
    for mode in ("no_rag", "rag"):
        for metric in ("faithfulness", "keyword_precision", "context_utilization", "composite_score"):
            key = f"{mode}_{metric}"
            values = [r[mode][metric] for r in results if mode in r]
            avg_metrics[key] = sum(values) / len(values) if values else 0.0

    print(f"\nВопросов: {total}")
    print(f"RAG побеждает: {rag_wins}/{total} ({rag_wins/total*100:.0f}%)")

    print(f"\n{'Метрика':<25} {'No RAG':>10} {'RAG':>10} {'Δ':>10}")
    print("-" * 57)
    for metric in ("faithfulness", "keyword_precision", "context_utilization", "composite_score"):
        no_rag_val = avg_metrics.get(f"no_rag_{metric}", 0)
        rag_val = avg_metrics.get(f"rag_{metric}", 0)
        delta = rag_val - no_rag_val
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<25} {no_rag_val:>10.3f} {rag_val:>10.3f} {sign}{delta:>9.3f}")

    print("=" * 70)
