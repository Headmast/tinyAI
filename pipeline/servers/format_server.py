"""
MCP Server 3: Format — форматирование контента для платформ (День 19).

Инструмент: format_content
Платформы: telegram, website, rss, plain
Протокол: MCP v2024-11-05, JSON-RPC 2.0, транспорт stdio.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.servers.base import MCPServerBase

# ── Инструменты ───────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "format_content",
        "description": "Форматирование контента для целевой платформы (telegram, website, rss, plain).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Контент для форматирования"},
                "platform": {
                    "type": "string",
                    "description": "Целевая платформа",
                    "enum": ["telegram", "website", "rss", "plain"],
                },
                "title": {"type": "string", "description": "Заголовок (опционально)"},
            },
            "required": ["content", "platform"],
        },
    },
]


def handle_format_content(args: dict) -> str:
    content = args.get("content", "").strip()
    if not content:
        return json.dumps({"error": "content is required"}, ensure_ascii=False)

    platform = args.get("platform", "plain")
    title = args.get("title", "")

    if platform == "telegram":
        lines = []
        if title:
            lines.append(f"<b>{title}</b>")
            lines.append("")
        lines.append(content)
        formatted = "\n".join(lines)

    elif platform == "website":
        paragraphs = content.strip().split("\n\n")
        html_body = "\n".join(
            f"<p>{p.strip()}</p>" for p in paragraphs if p.strip()
        )
        date_str = datetime.now().strftime("%Y-%m-%d")
        formatted = (
            "<!DOCTYPE html>\n<html lang=\"ru\">\n<head>\n"
            "  <meta charset=\"UTF-8\">\n"
            f"  <title>{title or 'Публикация'}</title>\n"
            "</head>\n<body>\n"
            f"<article>\n<h1>{title or ''}</h1>\n"
            f"<time>{date_str}</time>\n"
            f"{html_body}\n</article>\n</body>\n</html>"
        )

    elif platform == "rss":
        date_str = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
        formatted = (
            "<item>\n"
            f"  <title>{title or 'Без заголовка'}</title>\n"
            f"  <description><![CDATA[{content}]]></description>\n"
            f"  <pubDate>{date_str}</pubDate>\n"
            "</item>"
        )

    else:
        formatted = f"{title}\n\n{content}" if title else content

    return json.dumps({
        "formatted": formatted,
        "platform": platform,
        "char_count": len(formatted),
    }, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    server = MCPServerBase(
        name="tinyai-format-server",
        version="19.0.0",
        tools=TOOLS,
        handlers={"format_content": handle_format_content},
    )
    server.run()
