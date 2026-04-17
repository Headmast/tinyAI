"""
Reranker для второго этапа retrieval.

Поддерживает:
- similarity threshold filtering
- отдельную LLM-модель для reranking кандидатов
- математический reranker (BM25 + cosine blend) — без LLM

При ошибках автоматически возвращает исходный порядок (fallback).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Protocol

from rag import SearchResult


DEFAULT_RERANKER_MODEL = "zai-org/GLM-4.7-Flash"

_RERANK_SYSTEM_PROMPT = (
    "Ты reranker для RAG. "
    "Твоя задача: отсортировать кандидаты от самого релевантного к наименее релевантному. "
    "Верни только JSON формата: "
    "{\"ranking\": [{\"candidate_id\": 0, \"relevance\": 0.93}, ...]}. "
    "candidate_id должен ссылаться на id из входного списка. "
    "relevance в диапазоне [0, 1]."
)


class RerankerProtocol(Protocol):
    def rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        ...


class LLMReranker:
    """Reranker на отдельной LLM-модели."""

    def __init__(
        self,
        client: Any,
        model: str = DEFAULT_RERANKER_MODEL,
        enabled: bool = True,
        verbose: bool = False,
        max_candidates: int = 20,
    ) -> None:
        self.client = client
        self.model = model
        self.enabled = enabled
        self.verbose = verbose
        self.max_candidates = max_candidates

    def rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        if not self.enabled or not candidates:
            return candidates

        pool = candidates[: self.max_candidates]
        payload = self._build_payload(query, pool)
        raw = self._call_model(payload)
        ranking = self._extract_ranking(raw)
        if not ranking:
            return candidates

        reranked = self._apply_ranking(pool, ranking)
        # Сохраняем оставшиеся кандидаты в исходном порядке после reranked.
        seen_ids = {id(item) for item in reranked}
        tail = [c for c in candidates if id(c) not in seen_ids]
        return reranked + tail

    def _call_model(self, payload: Dict[str, Any]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=600,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if self.verbose:
                print(f"  [rerank] fallback: {e}")
            return ""

    @staticmethod
    def _build_payload(query: str, candidates: List[SearchResult]) -> Dict[str, Any]:
        payload_candidates = []
        for i, c in enumerate(candidates):
            md = c.chunk.metadata
            payload_candidates.append(
                {
                    "candidate_id": i,
                    "similarity": round(float(c.score), 6),
                    "source": md.get("source", ""),
                    "section": md.get("section", ""),
                    "text": c.chunk.text[:1200],
                }
            )
        return {
            "query": query,
            "candidates": payload_candidates,
        }

    @staticmethod
    def _extract_ranking(raw_content: str) -> List[Dict[str, float]]:
        content = (raw_content or "").strip()
        if not content:
            return []

        try:
            data = json.loads(content)
            ranking = data.get("ranking", [])
            if not isinstance(ranking, list):
                return []
            parsed: List[Dict[str, float]] = []
            for item in ranking:
                if not isinstance(item, dict):
                    continue
                if "candidate_id" not in item:
                    continue
                cand_id = int(item["candidate_id"])
                relevance = float(item.get("relevance", 0.0))
                parsed.append({"candidate_id": cand_id, "relevance": relevance})
            return parsed
        except Exception:
            return []

    @staticmethod
    def _apply_ranking(
        candidates: List[SearchResult],
        ranking: List[Dict[str, float]],
    ) -> List[SearchResult]:
        by_id = {i: c for i, c in enumerate(candidates)}
        result: List[SearchResult] = []

        for row in ranking:
            cand_id = row["candidate_id"]
            cand = by_id.get(cand_id)
            if cand is None:
                continue
            # relevance становится новым score (этап rerank)
            cand.score = float(row.get("relevance", cand.score))
            result.append(cand)
            by_id.pop(cand_id, None)

        return result


# ── Математический (BM25 + cosine) reranker ──────────────────────────────────

_STOPWORDS_RU = frozenset(
    "и в на не что как это для по из к с но а о же ли бы то ты мы он она"
    " они его её их мой все уже так да нет ну вот ещё будет этот эта эти"
    " был была были быть от до при или без между через".split()
)
_STOPWORDS_EN = frozenset(
    "the and for with from this that are was were been have has had will"
    " would could should may might can not but also its about into".split()
)
_STOPWORDS = _STOPWORDS_RU | _STOPWORDS_EN

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яёЁ0-9_\-]+")


def _tokenize(text: str) -> List[str]:
    """Разбить текст на lowercase токены, убрав стоп-слова."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """BM25 score для одного документа."""
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        f = tf.get(qt, 0)
        if f == 0:
            continue
        numerator = f * (k1 + 1)
        denominator = f + k1 * (1 - b + b * dl / max(avg_dl, 1))
        score += numerator / denominator
    return score


class MathReranker:
    """
    Reranker на базе BM25 + cosine similarity blend.

    Не использует LLM — чисто математический, работает мгновенно.
    Комбинирует:
      - cosine similarity (из FAISS, уже в score)
      - BM25 term overlap (слова запроса → слова чанка)
    """

    def __init__(
        self,
        cosine_weight: float = 0.6,
        bm25_weight: float = 0.4,
        enabled: bool = True,
        verbose: bool = False,
    ) -> None:
        self.cosine_weight = cosine_weight
        self.bm25_weight = bm25_weight
        self.enabled = enabled
        self.verbose = verbose

    def rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        if not self.enabled or not candidates:
            return candidates

        query_tokens = _tokenize(query)
        if not query_tokens:
            return candidates

        # Токенизируем все документы
        doc_tokens_list = [_tokenize(c.chunk.text) for c in candidates]
        avg_dl = sum(len(dt) for dt in doc_tokens_list) / max(len(doc_tokens_list), 1)

        # Считаем BM25 для каждого кандидата
        bm25_scores = [
            _bm25_score(query_tokens, dt, avg_dl)
            for dt in doc_tokens_list
        ]

        # Нормализуем BM25 в [0, 1]
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        if max_bm25 > 0:
            bm25_norm = [s / max_bm25 for s in bm25_scores]
        else:
            bm25_norm = [0.0] * len(bm25_scores)

        # Нормализуем cosine в [0, 1]
        cosine_scores = [c.score for c in candidates]
        max_cos = max(cosine_scores) if cosine_scores else 1.0
        min_cos = min(cosine_scores) if cosine_scores else 0.0
        cos_range = max_cos - min_cos if max_cos > min_cos else 1.0
        cos_norm = [(s - min_cos) / cos_range for s in cosine_scores]

        # Blend
        for i, cand in enumerate(candidates):
            cand.score = (
                self.cosine_weight * cos_norm[i]
                + self.bm25_weight * bm25_norm[i]
            )

        # Сортировка по итоговому score
        candidates.sort(key=lambda c: c.score, reverse=True)

        if self.verbose:
            print(f"  [math_rerank] {len(candidates)} candidates reranked")

        return candidates
