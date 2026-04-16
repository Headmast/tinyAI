"""
Query rewrite для RAG-поиска.

Задача: переформулировать вопрос пользователя так, чтобы улучшить retrieval.
Безопасность: при любой ошибке возвращает исходный query.
"""

from __future__ import annotations

import json
import re
from typing import Any


DEFAULT_REWRITE_MODEL = "zai-org/GLM-4.7-Flash"

_REWRITE_SYSTEM_PROMPT = (
    "Ты оптимизатор поисковых запросов для RAG по технической документации. "
    "Переформулируй запрос так, чтобы он лучше находил релевантные фрагменты. "
    "Сохраняй исходный смысл. Не добавляй фактов, которых нет в вопросе. "
    "Верни только JSON: {\"rewritten_query\": \"...\"}."
)


class QueryRewriter:
    """LLM-based query rewrite с безопасным fallback."""

    def __init__(
        self,
        client: Any,
        model: str = DEFAULT_REWRITE_MODEL,
        enabled: bool = True,
        verbose: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.enabled = enabled
        self.verbose = verbose

    def rewrite(self, query: str) -> str:
        """
        Переписывает query для поиска.

        Возвращает исходный query, если rewrite отключен или произошла ошибка.
        """
        if not self.enabled:
            return query

        raw = self._call_model(query)
        rewritten = self._extract_rewrite(raw)
        if not rewritten:
            return self._heuristic_rewrite(query)
        if self._normalize_query(rewritten) == self._normalize_query(query):
            return self._heuristic_rewrite(query)
        return rewritten

    def _call_model(self, query: str) -> str:
        messages = [
            {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=220,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if self.verbose:
                print(f"  [rewrite] fallback: {e}")
            return ""

    @staticmethod
    def _extract_rewrite(raw_content: str) -> str:
        content = (raw_content or "").strip()
        if not content:
            return ""

        try:
            data = json.loads(content)
            rewritten = str(data.get("rewritten_query", "")).strip()
            return rewritten
        except Exception:
            # Мягкий fallback: если LLM не вернул JSON, берём первую непустую строку.
            first_line = content.splitlines()[0].strip()
            if first_line.startswith("{") and first_line.endswith("}"):
                return ""
            return first_line

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join((query or "").strip().lower().split())

    @staticmethod
    def _heuristic_rewrite(query: str) -> str:
        """
        Детерминированный fallback-rewrite, если модель вернула пустой
        или идентичный запрос.
        """
        text = (query or "").strip()
        if not text:
            return query

        words = re.findall(r"[A-Za-zА-Яа-я0-9_\-]+", text)
        stopwords = {
            "как", "что", "это", "для", "при", "или", "и", "в", "на", "по",
            "из", "к", "о", "а", "но", "the", "and", "for", "with",
        }
        keywords = []
        for w in words:
            wl = w.lower()
            if len(wl) < 4 or wl in stopwords:
                continue
            if wl not in keywords:
                keywords.append(wl)
            if len(keywords) >= 8:
                break

        if not keywords:
            return f"Технический запрос по документации TinyAI: {text}"

        key_str = ", ".join(keywords)
        return f"Технический запрос по документации TinyAI: {text}. Ключевые термины: {key_str}."
