"""
Reranker для второго этапа retrieval.

Поддерживает:
- similarity threshold filtering
- отдельную LLM-модель для reranking кандидатов

При ошибках автоматически возвращает исходный порядок (fallback).
"""

from __future__ import annotations

import json
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
