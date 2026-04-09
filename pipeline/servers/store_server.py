"""
MCP Server 4: Store — сохранение контента в файл и БД (День 19).

Инструменты: save_to_file, save_to_db
Протокол: MCP v2024-11-05, JSON-RPC 2.0, транспорт stdio.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.servers.base import MCPServerBase

BASE_DIR = Path(__file__).parent.parent.parent
POSTS_DIR = BASE_DIR / "pipeline_data" / "posts"

# ── Инструменты ───────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "save_to_file",
        "description": "Сохранение контента в файл в pipeline_data/posts/.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Контент для сохранения"},
                "filename": {"type": "string", "description": "Имя файла (без пути)"},
            },
            "required": ["content", "filename"],
        },
    },
    {
        "name": "save_to_db",
        "description": "Сохранение контента в SQLite (pipeline_data/content.db).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Заголовок контента"},
                "content": {"type": "string", "description": "Основной текст"},
                "platform": {
                    "type": "string",
                    "description": "Целевая платформа",
                    "enum": ["telegram", "website", "rss", "plain"],
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Теги",
                    "default": [],
                },
                "summary": {"type": "string", "description": "Краткое описание"},
                "source_query": {"type": "string", "description": "Исходный запрос"},
                "pipeline_run": {"type": "string", "description": "ID запуска пайплайна"},
            },
            "required": ["title", "content", "platform"],
        },
    },
]


def handle_save_to_file(args: dict) -> str:
    content = args.get("content", "")
    filename = args.get("filename", "").strip()
    if not filename:
        return json.dumps({"error": "filename is required"}, ensure_ascii=False)

    safe_name = Path(filename).name
    if not safe_name or safe_name.startswith("."):
        return json.dumps({"error": "invalid filename"}, ensure_ascii=False)

    POSTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = POSTS_DIR / safe_name

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return json.dumps({
        "saved": True,
        "path": str(filepath),
        "size_bytes": len(content.encode("utf-8")),
    }, ensure_ascii=False)


def handle_save_to_db(args: dict) -> str:
    title = args.get("title", "").strip()
    content = args.get("content", "").strip()
    platform = args.get("platform", "plain")

    if not title or not content:
        return json.dumps({"error": "title and content are required"}, ensure_ascii=False)

    from pipeline.db import init_db, save_content

    init_db()
    content_id = save_content(
        title=title,
        content=content,
        platform=platform,
        summary=args.get("summary"),
        tags=args.get("tags"),
        source_query=args.get("source_query"),
        pipeline_run=args.get("pipeline_run"),
    )
    return json.dumps({
        "saved": True,
        "content_id": content_id,
        "platform": platform,
        "status": "draft",
    }, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    server = MCPServerBase(
        name="tinyai-store-server",
        version="19.0.0",
        tools=TOOLS,
        handlers={
            "save_to_file": handle_save_to_file,
            "save_to_db": handle_save_to_db,
        },
    )
    server.run()
