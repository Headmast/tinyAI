#!/usr/bin/env python3
"""
Индексация документации проекта для Developer Assistant.

Загружает README.md + все .md файлы из docs/ в отдельный RAG-индекс (rag_data_dev/).
Используется командой /help в chat_cli.py.

Запуск:
    python dev_assistant_index.py
    python dev_assistant_index.py --strategy structure
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
from rag.indexer import load_documents

PROJECT_ROOT = Path(__file__).parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "rag_data_dev"


def load_project_docs() -> list[Document]:
    """Загружает README.md + все .md из docs/."""
    documents: list[Document] = []

    # 1. README.md из корня проекта
    readme_path = PROJECT_ROOT / "README.md"
    if readme_path.exists():
        text = readme_path.read_text(encoding="utf-8")
        if text.strip():
            documents.append(Document(
                text=text,
                metadata={
                    "source": "README.md",
                    "title": "README — TinyAI Project",
                    "file_type": "markdown",
                },
            ))
            print(f"  ✅ README.md ({len(text):,} символов)")

    # 2. Все .md файлы из docs/
    docs_dir = PROJECT_ROOT / "docs"
    if docs_dir.exists():
        docs_from_dir = load_documents(docs_dir)
        documents.extend(docs_from_dir)
        print(f"  ✅ docs/ — {len(docs_from_dir)} файлов")
    else:
        print(f"  ⚠️  Директория docs/ не найдена")

    return documents


def build_dev_index(strategy: str = "structure", index_dir: Path = DEFAULT_INDEX_DIR) -> dict:
    """Строит RAG-индекс документации проекта."""
    print(f"{'=' * 60}")
    print(f"Developer Assistant — Индексация документации")
    print(f"{'=' * 60}")

    # Загрузка документов
    print(f"\n📄 Загрузка документации проекта...")
    documents = load_project_docs()
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
    print(f"✅ Индексация завершена! Индекс: {index_dir}")
    print(f"{'=' * 60}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Индексация документации проекта для /help")
    parser.add_argument(
        "--strategy", default="structure",
        choices=["fixed_size", "structure", "both"],
        help="Стратегия чанкинга (default: structure)",
    )
    parser.add_argument(
        "--index-dir", default=str(DEFAULT_INDEX_DIR),
        help=f"Директория для индекса (default: {DEFAULT_INDEX_DIR})",
    )
    args = parser.parse_args()

    stats = build_dev_index(strategy=args.strategy, index_dir=Path(args.index_dir))
    print(f"\nСтатистика: {stats}")


if __name__ == "__main__":
    main()
