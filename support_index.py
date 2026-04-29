#!/usr/bin/env python3
"""
Индексация документации поддержки для Support Assistant.

Загружает все .md файлы из support_data/ в RAG-индекс (rag_data_support/).
Используется командой /support в chat_cli.py.

Запуск:
    python support_index.py
    python support_index.py --strategy structure
    python support_index.py --index-dir my_index/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag import Document, Chunk
from rag.chunker import StructureChunker, FixedSizeChunker
from rag.embedder import OpenAIEmbedder
from rag.index_store import FAISSIndexStore

PROJECT_ROOT = Path(__file__).parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "rag_data_support"
SUPPORT_DATA_DIR = PROJECT_ROOT / "support_data"


def load_support_docs() -> list[Document]:
    """Загружает все .md файлы из support_data/."""
    documents: list[Document] = []

    if not SUPPORT_DATA_DIR.exists():
        print(f"  ❌ Директория {SUPPORT_DATA_DIR} не найдена!")
        return documents

    md_files = sorted(SUPPORT_DATA_DIR.glob("*.md"))
    if not md_files:
        print(f"  ❌ Нет .md файлов в {SUPPORT_DATA_DIR}")
        return documents

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        if not text.strip():
            continue

        # Извлекаем заголовок из первой строки (# Title)
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        title = first_line.lstrip("# ").strip() if first_line.startswith("#") else md_path.stem

        documents.append(Document(
            text=text,
            metadata={
                "source": f"support_data/{md_path.name}",
                "title": title,
                "file_type": "markdown",
            },
        ))
        print(f"  ✅ {md_path.name} ({len(text):,} символов)")

    return documents

    return documents


def build_support_index(strategy: str = "structure", index_dir: Path = DEFAULT_INDEX_DIR) -> dict:
    """Строит RAG-индекс документации поддержки."""
    print(f"{'=' * 60}")
    print(f"Support Assistant — Индексация документации")
    print(f"{'=' * 60}")

    # Загрузка документов
    print(f"\n📄 Загрузка документации поддержки...")
    documents = load_support_docs()
    total_chars = sum(len(d.text) for d in documents)
    print(f"\n   Итого: {len(documents)} документов, {total_chars:,} символов")

    if not documents:
        print("   ❌ Нет документов для индексации!")
        return {"documents": 0}

    # Определяем стратегии
    strategies = []
    if strategy in ("fixed_size", "both"):
        strategies.append("fixed_size")
    if strategy in ("structure", "both"):
        strategies.append("structure")

    embedder = OpenAIEmbedder()
    index_dir.mkdir(parents=True, exist_ok=True)
    store = FAISSIndexStore(index_dir=index_dir)
    stats: dict = {"documents": len(documents), "total_chars": total_chars}

    for strat in strategies:
        print(f"\n{'─' * 60}")
        print(f"🔧 Стратегия: {strat}")
        print(f"{'─' * 60}")

        # Chunking
        t0 = time.time()
        if strat == "fixed_size":
            chunker = FixedSizeChunker(chunk_size=512, chunk_overlap=64)
        else:
            chunker = StructureChunker(max_chunk_tokens=1024)

        all_chunks: list[Chunk] = []
        for doc in documents:
            doc_chunks = chunker.chunk(doc)
            all_chunks.extend(doc_chunks)

        chunk_time = time.time() - t0
        total_tokens = sum(c.metadata.get("token_count", 0) for c in all_chunks)
        print(f"   Чанки: {len(all_chunks)}, токенов: {total_tokens:,}, "
              f"время: {chunk_time:.2f}с")

        # Embeddings
        print(f"   ⏳ Генерация эмбеддингов ({len(all_chunks)} чанков)...")
        t0 = time.time()
        texts = [c.text for c in all_chunks]
        embeddings = embedder.embed_texts(texts)
        embed_time = time.time() - t0
        print(f"   ✅ Эмбеддинги: shape {embeddings.shape}, время: {embed_time:.2f}с")

        # FAISS index
        print(f"   📦 Построение FAISS-индекса...")
        store.build_index(embeddings, all_chunks, strategy=strat)

        # Save
        index_file = store.save(strat)
        file_size_mb = index_file.stat().st_size / (1024 * 1024)
        print(f"   💾 Сохранено: {index_file} ({file_size_mb:.2f} MB)")

        stats[strat] = {
            "chunks": len(all_chunks),
            "total_tokens": total_tokens,
            "index_file_mb": round(file_size_mb, 2),
        }

    print(f"\n{'=' * 60}")
    print(f"✅ Индексация документации завершена! Индекс: {index_dir}")
    print(f"{'=' * 60}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Индексация документации поддержки для /support")
    parser.add_argument(
        "--strategy", default="structure",
        choices=["fixed_size", "structure", "both"],
        help="Стратегия чанкинга (default: structure)",
    )
    parser.add_argument(
        "--index-dir", default=str(DEFAULT_INDEX_DIR),
        help=f"Путь к индексу (default: {DEFAULT_INDEX_DIR})",
    )
    args = parser.parse_args()
    build_support_index(strategy=args.strategy, index_dir=Path(args.index_dir))


if __name__ == "__main__":
    main()
