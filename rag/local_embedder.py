"""
Локальный эмбеддер через Ollama API.

Использует nomic-embed-text (768-dim) — бесплатный, быстрый,
работает полностью локально через Ollama.

Установка модели:
    ollama pull nomic-embed-text
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import List

import numpy as np
from openai import OpenAI


DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_EMBED_DIM = 768
OLLAMA_BATCH_SIZE = 50  # Меньше чем OpenAI — локальная модель медленнее


class OllamaEmbedder:
    """
    Генератор эмбеддингов через Ollama (OpenAI-совместимый /v1/embeddings).

    Батчирует запросы по OLLAMA_BATCH_SIZE элементов.
    Нормализует вектора (L2) для использования с FAISS IndexFlatIP.
    Кэширует эмбеддинги запросов для повторных вызовов.
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_EMBED_MODEL,
        base_url: str | None = None,
        cache_size: int = 256,
    ) -> None:
        self.model = model
        self._base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434/v1"
        )
        self._client = OpenAI(api_key="ollama", base_url=self._base_url)
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is not None:
            return self._dim
        # Определяем размерность по первому вызову
        probe = self.embed_texts(["test"])
        self._dim = probe.shape[1]
        return self._dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Генерирует эмбеддинги для списка текстов через Ollama.

        Возвращает np.ndarray shape (len(texts), dim), L2-нормализованный.
        """
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), OLLAMA_BATCH_SIZE):
            batch = texts[i : i + OLLAMA_BATCH_SIZE]
            batch = [t if t.strip() else " " for t in batch]

            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            if i + OLLAMA_BATCH_SIZE < len(texts):
                time.sleep(0.05)

        arr = np.array(all_embeddings, dtype=np.float32)

        # Запоминаем размерность
        if self._dim is None and arr.shape[0] > 0:
            self._dim = arr.shape[1]

        # L2-нормализация для cosine similarity через inner product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        arr = arr / norms

        return arr

    def embed_query(self, query: str) -> np.ndarray:
        """Генерирует эмбеддинг одного запроса. Возвращает shape (1, dim). Кэширует (LRU)."""
        key = query.strip()
        if key in self._query_cache:
            self._query_cache.move_to_end(key)
            return self._query_cache[key]
        result = self.embed_texts([query])
        if len(self._query_cache) >= self._cache_size:
            self._query_cache.popitem(last=False)
        self._query_cache[key] = result
        return result
