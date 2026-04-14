"""
Хранилище индекса: FAISS (вектора) + SQLite (метаданные чанков).

FAISS IndexFlatIP с L2-нормализованными векторами ≡ cosine similarity.
SQLite хранит текст, метаданные и маппинг FAISS-id → chunk_id.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore

from rag import Chunk

BASE_DIR = Path(__file__).parent.parent
DEFAULT_INDEX_DIR = BASE_DIR / "rag_data"


# ── SQLite ────────────────────────────────────────────────────────────────────

def _get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    with _get_connection(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id      TEXT    UNIQUE NOT NULL,
                faiss_id      INTEGER,
                text          TEXT    NOT NULL,
                source        TEXT,
                title         TEXT,
                section       TEXT,
                strategy      TEXT,
                chunk_index   INTEGER,
                token_count   INTEGER,
                char_count    INTEGER,
                embedding_dim INTEGER,
                created_at    TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_strategy
                ON chunks(strategy);

            CREATE INDEX IF NOT EXISTS idx_chunks_source
                ON chunks(source);
        """)


def save_chunks(db_path: Path, chunks: List[Chunk], embedding_dim: int) -> None:
    now = datetime.now().isoformat()
    init_db(db_path)

    # Определяем стратегию из первого чанка для очистки старых данных
    if chunks:
        strategy = chunks[0].metadata.get("strategy", "")
        if strategy:
            with _get_connection(db_path) as conn:
                conn.execute("DELETE FROM chunks WHERE strategy = ?", (strategy,))

    with _get_connection(db_path) as conn:
        for i, chunk in enumerate(chunks):
            m = chunk.metadata
            conn.execute(
                """INSERT OR REPLACE INTO chunks
                   (chunk_id, faiss_id, text, source, title, section,
                    strategy, chunk_index, token_count, char_count,
                    embedding_dim, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    m.get("chunk_id", f"chunk_{i}"),
                    i,  # faiss_id = порядковый индекс в FAISS
                    chunk.text,
                    m.get("source", ""),
                    m.get("title", ""),
                    m.get("section", ""),
                    m.get("strategy", ""),
                    m.get("chunk_index", i),
                    m.get("token_count", 0),
                    m.get("char_count", len(chunk.text)),
                    embedding_dim,
                    now,
                ),
            )


def get_chunks_by_faiss_ids(
    db_path: Path, faiss_ids: List[int], strategy: Optional[str] = None
) -> List[Dict[str, Any]]:
    init_db(db_path)
    with _get_connection(db_path) as conn:
        placeholders = ",".join("?" * len(faiss_ids))
        query = f"SELECT * FROM chunks WHERE faiss_id IN ({placeholders})"
        params: list = list(faiss_ids)

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_all_chunks(
    db_path: Path, strategy: Optional[str] = None
) -> List[Dict[str, Any]]:
    init_db(db_path)
    with _get_connection(db_path) as conn:
        if strategy:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE strategy = ? ORDER BY id", (strategy,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM chunks ORDER BY id").fetchall()
        return [dict(row) for row in rows]


def get_chunk_stats(db_path: Path) -> Dict[str, Any]:
    init_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT strategy,
                   COUNT(*) as count,
                   AVG(token_count) as avg_tokens,
                   MIN(token_count) as min_tokens,
                   MAX(token_count) as max_tokens,
                   SUM(token_count) as total_tokens,
                   SUM(char_count) as total_chars
            FROM chunks
            GROUP BY strategy
        """).fetchall()
        return {row["strategy"]: dict(row) for row in rows}


# ── FAISS ─────────────────────────────────────────────────────────────────────

class FAISSIndexStore:
    """
    Обёртка над FAISS IndexFlatIP.

    Хранит индекс в файле {index_dir}/index_{strategy}.faiss.
    Метаданные — в {index_dir}/chunks.db.
    """

    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR) -> None:
        if faiss is None:
            raise ImportError("faiss-cpu не установлен: pip install faiss-cpu")
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.index_dir / "chunks.db"
        self._indices: Dict[str, Any] = {}  # strategy -> faiss.Index

    def build_index(
        self,
        embeddings: np.ndarray,
        chunks: List[Chunk],
        strategy: str,
    ) -> None:
        """
        Создаёт FAISS-индекс и сохраняет метаданные в SQLite.

        Args:
            embeddings: np.ndarray shape (n, dim), L2-нормализованные
            chunks: список чанков (того же размера что embeddings)
            strategy: имя стратегии ('fixed_size' или 'structure')
        """
        n, dim = embeddings.shape
        assert n == len(chunks), f"embeddings ({n}) != chunks ({len(chunks)})"

        # Создаём FAISS индекс (inner product = cosine для нормализованных)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self._indices[strategy] = index

        # Сохраняем метаданные
        save_chunks(self.db_path, chunks, embedding_dim=dim)

    def save(self, strategy: str) -> Path:
        """Сохраняет FAISS-индекс на диск."""
        index = self._indices.get(strategy)
        if index is None:
            raise ValueError(f"Индекс для стратегии '{strategy}' не создан")

        path = self.index_dir / f"index_{strategy}.faiss"
        faiss.write_index(index, str(path))
        return path

    def load(self, strategy: str) -> None:
        """Загружает FAISS-индекс с диска."""
        path = self.index_dir / f"index_{strategy}.faiss"
        if not path.exists():
            raise FileNotFoundError(f"Индекс не найден: {path}")
        self._indices[strategy] = faiss.read_index(str(path))

    def search(
        self,
        query_embedding: np.ndarray,
        strategy: str,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Поиск ближайших соседей.

        Args:
            query_embedding: shape (1, dim), L2-нормализованный
            strategy: стратегия для поиска
            top_k: количество результатов

        Returns:
            (scores, indices): shape (1, top_k) каждый
        """
        index = self._indices.get(strategy)
        if index is None:
            self.load(strategy)
            index = self._indices[strategy]

        scores, indices = index.search(query_embedding, top_k)
        return scores, indices

    def index_size(self, strategy: str) -> int:
        """Количество векторов в индексе."""
        index = self._indices.get(strategy)
        if index is None:
            try:
                self.load(strategy)
                index = self._indices[strategy]
            except FileNotFoundError:
                return 0
        return index.ntotal
