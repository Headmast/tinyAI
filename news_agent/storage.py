"""
PostStorage — хранилище сгенерированных новостных постов.
Сохраняет посты в папку posts/ в форматах JSON + Markdown.
Ведёт индексный файл posts/index.json.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class PostStorage:
    """
    Файловое хранилище постов.
    Структура:
        posts/
            index.json          — индекс всех постов
            <id>/
                post.json       — метаданные + контент
                post.md         — Markdown-версия
    """

    def __init__(self, base_dir: str = "posts"):
        self.base_dir = Path(base_dir)
        self.index_file = self.base_dir / "index.json"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_file.exists():
            self._write_index([])

    def _read_index(self) -> List[Dict[str, Any]]:
        with open(self.index_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_index(self, index: List[Dict[str, Any]]) -> None:
        tmp = self.index_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.index_file)

    def save(self, post_data: Dict[str, Any]) -> str:
        """
        Сохраняет пост и возвращает его уникальный ID.
        post_data должен содержать: title, content, post_type.
        """
        post_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        post = {
            "id": post_id,
            "title": post_data.get("title", "Без заголовка"),
            "content": post_data.get("content", ""),
            "post_type": post_data.get("post_type", "breaking"),
            "tags": post_data.get("tags", []),
            "meta_description": post_data.get("meta_description", ""),
            "model": post_data.get("model", "unknown"),
            "pipeline_stats": post_data.get("pipeline_stats", {}),
            "created_at": timestamp,
            "word_count": len(post_data.get("content", "").split()),
            "char_count": len(post_data.get("content", "")),
        }

        post_dir = self.base_dir / post_id
        post_dir.mkdir(exist_ok=True)

        with open(post_dir / "post.json", "w", encoding="utf-8") as f:
            json.dump(post, f, ensure_ascii=False, indent=2)

        md_content = self._to_markdown(post)
        with open(post_dir / "post.md", "w", encoding="utf-8") as f:
            f.write(md_content)

        index = self._read_index()
        index.insert(0, {
            "id": post_id,
            "title": post["title"],
            "post_type": post["post_type"],
            "tags": post["tags"],
            "created_at": timestamp,
            "word_count": post["word_count"],
        })
        self._write_index(index)

        return post_id

    def load(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Загружает полный пост по ID."""
        post_file = self.base_dir / post_id / "post.json"
        if not post_file.exists():
            return None
        with open(post_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_posts(
        self,
        n: int = 10,
        post_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Возвращает последние N постов из индекса."""
        index = self._read_index()
        if post_type:
            index = [p for p in index if p.get("post_type") == post_type]
        return index[:n]

    def find_by_keywords(
        self, topic: str, min_overlap: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Поиск постов по пересечению ключевых слов (Jaccard similarity).
        Не является семантическим поиском — сравниваются слова заголовков.
        В production можно заменить на векторный поиск.

        Args:
            topic:       строка поискового запроса
            min_overlap: минимальный коэффициент пересечения слов (0–1)
        """
        topic_words = set(topic.lower().split())
        stop_words = {"в", "на", "с", "по", "о", "из", "и", "а", "но", "или", "что", "как"}
        topic_words -= stop_words

        if not topic_words:
            return []

        results = []
        index = self._read_index()

        for entry in index:
            title_words = set(entry["title"].lower().split()) - stop_words
            if not title_words:
                continue
            overlap = len(topic_words & title_words)
            score = overlap / max(len(topic_words), len(title_words))
            if score >= min_overlap:
                results.append({**entry, "overlap_score": round(score, 2)})

        results.sort(key=lambda x: x["overlap_score"], reverse=True)
        return results

    def search_similar(
        self, topic: str, threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Устаревший алиас для find_by_keywords(). Используйте find_by_keywords()."""
        return self.find_by_keywords(topic, min_overlap=threshold)

    def delete(self, post_id: str) -> bool:
        """Удаляет пост из хранилища."""
        import shutil
        post_dir = self.base_dir / post_id
        if not post_dir.exists():
            return False
        shutil.rmtree(post_dir)
        index = [p for p in self._read_index() if p["id"] != post_id]
        self._write_index(index)
        return True

    def export_post(self, post_id: str, fmt: str = "markdown") -> Optional[str]:
        """Экспортирует пост в указанном формате."""
        from news_agent.formatter import OutputFormatter
        post = self.load(post_id)
        if not post:
            return None
        formatter = OutputFormatter()
        return formatter.convert(
            content=post["content"],
            fmt=fmt,
            title=post.get("title"),
            tags=post.get("tags", []),
            meta=post,
        )

    @staticmethod
    def _to_markdown(post: Dict[str, Any]) -> str:
        lines = [
            f"# {post['title']}",
            "",
            f"> **Тип:** {post['post_type']}  ",
            f"> **Дата:** {post['created_at'][:10]}  ",
            f"> **Теги:** {', '.join(post['tags']) if post['tags'] else '—'}",
            "",
            "---",
            "",
            post["content"],
            "",
        ]
        if post.get("meta_description"):
            lines += ["---", "", f"*{post['meta_description']}*", ""]
        return "\n".join(lines)
