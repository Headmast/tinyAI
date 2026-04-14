"""
Генерация эмбеддингов через OpenAI API.

Использует text-embedding-3-small (1536-dim) — минимальная стоимость,
достаточное качество для документов курса (~288 страниц).
"""

from __future__ import annotations

import os
import time
from typing import List

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100  # OpenAI рекомендует до 2048, но 100 — безопасный размер


class OpenAIEmbedder:
    """
    Генератор эмбеддингов через OpenAI API.

    Батчирует запросы по BATCH_SIZE элементов.
    Нормализует вектора (L2) для использования с FAISS IndexFlatIP.
    """

    def __init__(self, model: str = EMBEDDING_MODEL) -> None:
        self.model = model
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Генерирует эмбеддинги для списка текстов.

        Возвращает np.ndarray shape (len(texts), EMBEDDING_DIM),
        L2-нормализованный.
        """
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            # Заменяем пустые строки пробелом (API не принимает пустые)
            batch = [t if t.strip() else " " for t in batch]

            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            # Небольшая пауза между батчами для rate limiting
            if i + BATCH_SIZE < len(texts):
                time.sleep(0.1)

        arr = np.array(all_embeddings, dtype=np.float32)
        # L2-нормализация для cosine similarity через inner product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        arr = arr / norms

        return arr

    def embed_query(self, query: str) -> np.ndarray:
        """Генерирует эмбеддинг одного запроса. Возвращает shape (1, dim)."""
        return self.embed_texts([query])
