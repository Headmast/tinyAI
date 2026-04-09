"""
MCP Server 1: Search — поиск по логам и данным проекта (День 19).

Инструмент: search
Протокол: MCP v2024-11-05, JSON-RPC 2.0, транспорт stdio.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.servers.base import MCPServerBase

BASE_DIR = Path(__file__).parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"

# ── Инструменты ───────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search",
        "description": "Поиск по логам разговоров проекта. Возвращает найденные фрагменты.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Текст для поиска"},
            },
            "required": ["query"],
        },
    },
]


def handle_search(args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)

    results = []
    pattern = re.compile(re.escape(query), re.IGNORECASE)

    if LOGS_DIR.exists():
        for log_file in sorted(LOGS_DIR.glob("conversation_*.json")):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                messages = data if isinstance(data, list) else data.get("messages", [])
                for msg in messages:
                    content_parts = []
                    if "content" in msg:
                        content_parts.append(msg["content"])
                    if "user_input" in msg:
                        content_parts.append(msg["user_input"])
                    if "assistant_response" in msg:
                        content_parts.append(msg["assistant_response"])
                    content = " ".join(content_parts)
                    if pattern.search(content):
                        results.append({
                            "file": log_file.name,
                            "role": msg.get("role", "dialog"),
                            "snippet": content[:300],  # короче = меньше токенов
                        })
            except (json.JSONDecodeError, OSError):
                continue

    limit = args.get("limit", 5)  # топ-5 по умолчанию
    return json.dumps({
        "query": query,
        "found": len(results),
        "results": results[:limit],
    }, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    server = MCPServerBase(
        name="tinyai-search-server",
        version="19.0.0",
        tools=TOOLS,
        handlers={"search": handle_search},
    )
    server.run()
