"""
Оркестратор индексации: загрузка документов → chunking → embedding → сохранение.

CLI:
    python -m rag.indexer --strategy fixed_size|structure|both --docs-dir docs/
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import Document, Chunk
from rag.chunker import FixedSizeChunker, StructureChunker
from rag.embedder import OpenAIEmbedder
from rag.index_store import FAISSIndexStore, get_chunk_stats


# ── Document Loading ──────────────────────────────────────────────────────────

_TITLE_RE = re.compile(r"^#{1,2}\s+(.+)$", re.MULTILINE)


def load_documents(docs_dir: str | Path) -> List[Document]:
    """
    Рекурсивно загружает все .md файлы из директории.

    Извлекает метаданные:
        source   — относительный путь
        title    — первый H1/H2 заголовок или имя файла
        file_type — 'markdown'
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Директория не найдена: {docs_path}")

    documents: List[Document] = []

    for md_file in sorted(docs_path.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        if not text.strip():
            continue

        # Извлекаем заголовок
        title_match = _TITLE_RE.search(text)
        title = title_match.group(1).strip() if title_match else md_file.stem

        # Относительный путь от корня проекта
        try:
            source = str(md_file.relative_to(Path(__file__).parent.parent))
        except ValueError:
            source = str(md_file)

        documents.append(Document(
            text=text,
            metadata={
                "source": source,
                "title": title,
                "file_type": "markdown",
            },
        ))

    return documents


# ── Indexing Pipeline ─────────────────────────────────────────────────────────

def build_index(
    strategy: str = "both",
    docs_dir: str = "docs/",
    index_dir: str = "rag_data/",
) -> dict:
    """
    Основной пайплайн индексации.

    1. Загрузка документов
    2. Chunking (одна или обе стратегии)
    3. Генерация эмбеддингов
    4. Построение FAISS-индекса
    5. Сохранение на диск

    Args:
        strategy: 'fixed_size', 'structure', or 'both'
        docs_dir: путь к директории с документами
        index_dir: путь для сохранения индекса

    Returns:
        dict со статистикой индексации
    """
    # Resolve relative paths from project root
    project_root = Path(__file__).parent.parent
    docs_path = project_root / docs_dir
    index_path = project_root / index_dir

    print(f"{'='*60}")
    print(f"RAG Indexing Pipeline")
    print(f"{'='*60}")

    # Step 1: Load documents
    print(f"\n📄 Загрузка документов из {docs_path}...")
    documents = load_documents(docs_path)
    total_chars = sum(len(d.text) for d in documents)
    print(f"   Загружено: {len(documents)} документов, {total_chars:,} символов")

    strategies = []
    if strategy in ("fixed_size", "both"):
        strategies.append("fixed_size")
    if strategy in ("structure", "both"):
        strategies.append("structure")

    embedder = OpenAIEmbedder()
    store = FAISSIndexStore(index_dir=index_path)
    stats: dict = {"documents": len(documents), "total_chars": total_chars}

    for strat in strategies:
        print(f"\n{'─'*60}")
        print(f"🔧 Стратегия: {strat}")
        print(f"{'─'*60}")

        # Step 2: Chunking
        t0 = time.time()
        if strat == "fixed_size":
            chunker = FixedSizeChunker(chunk_size=512, chunk_overlap=64)
        else:
            chunker = StructureChunker(max_chunk_tokens=1024)

        all_chunks: List[Chunk] = []
        for doc in documents:
            doc_chunks = chunker.chunk(doc)
            all_chunks.extend(doc_chunks)

        chunk_time = time.time() - t0
        total_tokens = sum(c.metadata.get("token_count", 0) for c in all_chunks)
        print(f"   Чанки: {len(all_chunks)}, токенов: {total_tokens:,}, "
              f"время: {chunk_time:.2f}с")

        # Step 3: Embeddings
        print(f"   ⏳ Генерация эмбеддингов ({len(all_chunks)} чанков)...")
        t0 = time.time()
        texts = [c.text for c in all_chunks]
        embeddings = embedder.embed_texts(texts)
        embed_time = time.time() - t0
        print(f"   ✅ Эмбеддинги: shape {embeddings.shape}, время: {embed_time:.2f}с")

        # Step 4: Build FAISS index
        print(f"   📦 Построение FAISS-индекса...")
        store.build_index(embeddings, all_chunks, strategy=strat)

        # Step 5: Save
        index_file = store.save(strat)
        file_size_mb = index_file.stat().st_size / (1024 * 1024)
        print(f"   💾 Сохранено: {index_file} ({file_size_mb:.2f} MB)")

        stats[strat] = {
            "chunks": len(all_chunks),
            "total_tokens": total_tokens,
            "avg_tokens": total_tokens / len(all_chunks) if all_chunks else 0,
            "embedding_shape": list(embeddings.shape),
            "index_file_mb": round(file_size_mb, 2),
            "chunk_time_s": round(chunk_time, 2),
            "embed_time_s": round(embed_time, 2),
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Индексация завершена!")
    print(f"{'='*60}")

    db_stats = get_chunk_stats(store.db_path)
    for strat_name, s in db_stats.items():
        print(f"\n  [{strat_name}]")
        print(f"    Чанков:      {s['count']}")
        print(f"    Avg токенов: {s['avg_tokens']:.0f}")
        print(f"    Min/Max:     {s['min_tokens']}/{s['max_tokens']}")
        print(f"    Всего:       {s['total_tokens']:,} токенов, "
              f"{s['total_chars']:,} символов")

    print(f"\n  Файлы индекса: {index_path}/")

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Document Indexer")
    parser.add_argument(
        "--strategy",
        choices=["fixed_size", "structure", "both"],
        default="both",
        help="Стратегия chunking (по умолчанию: both)",
    )
    parser.add_argument(
        "--docs-dir",
        default="docs/",
        help="Директория с документами (по умолчанию: docs/)",
    )
    parser.add_argument(
        "--index-dir",
        default="rag_data/",
        help="Директория для индекса (по умолчанию: rag_data/)",
    )
    args = parser.parse_args()

    build_index(
        strategy=args.strategy,
        docs_dir=args.docs_dir,
        index_dir=args.index_dir,
    )


if __name__ == "__main__":
    main()
