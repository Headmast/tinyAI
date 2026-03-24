"""
OutputFormatter — конвертация постов в различные форматы.
Поддерживает: markdown, html, telegram, json, plain.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


class OutputFormatter:
    """Конвертирует текст поста в различные выходные форматы."""

    def convert(
        self,
        content: str,
        fmt: str,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        fmt = fmt.lower().strip()
        handlers = {
            "markdown": self.to_markdown,
            "md": self.to_markdown,
            "html": self.to_html,
            "telegram": self.to_telegram,
            "tg": self.to_telegram,
            "json": self.to_json,
            "plain": self.to_plain,
            "txt": self.to_plain,
        }
        handler = handlers.get(fmt, self.to_plain)
        return handler(content=content, title=title, tags=tags or [], meta=meta or {})

    def to_markdown(
        self,
        content: str,
        title: Optional[str] = None,
        tags: List[str] = [],
        meta: Dict[str, Any] = {},
    ) -> str:
        lines = []
        if title:
            lines.append(f"# {title}")
            lines.append("")
        date_str = meta.get("created_at", datetime.now().isoformat())[:10]
        post_type = meta.get("post_type", "")
        if date_str or post_type:
            lines.append(f"*{date_str}*{' · ' + post_type if post_type else ''}")
            lines.append("")
        lines.append(content)
        if tags:
            lines.append("")
            lines.append("---")
            lines.append(f"**Теги:** {', '.join(f'`{t}`' for t in tags)}")
        return "\n".join(lines)

    def to_html(
        self,
        content: str,
        title: Optional[str] = None,
        tags: List[str] = [],
        meta: Dict[str, Any] = {},
    ) -> str:
        paragraphs = content.strip().split("\n\n")
        html_paragraphs = "\n".join(
            f"<p>{p.strip().replace(chr(10), '<br>')}</p>"
            for p in paragraphs
            if p.strip()
        )
        date_str = meta.get("created_at", datetime.now().isoformat())[:10]
        meta_desc = meta.get("meta_description", "")

        html = [
            "<!DOCTYPE html>",
            '<html lang="ru">',
            "<head>",
            '  <meta charset="UTF-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        ]
        if title:
            html.append(f"  <title>{title}</title>")
        if meta_desc:
            html.append(f'  <meta name="description" content="{meta_desc}">'),
        html += [
            "  <style>",
            "    body { font-family: Georgia, serif; max-width: 780px; margin: 40px auto; padding: 0 20px; line-height: 1.7; color: #1a1a1a; }",
            "    h1 { font-size: 2em; margin-bottom: 0.3em; }",
            "    .meta { color: #666; font-size: 0.9em; margin-bottom: 1.5em; }",
            "    .tags { margin-top: 2em; padding-top: 1em; border-top: 1px solid #eee; font-size: 0.85em; color: #666; }",
            "    p { margin: 0.8em 0; }",
            "  </style>",
            "</head>",
            "<body>",
        ]
        if title:
            html.append(f"  <h1>{title}</h1>")
        if date_str:
            html.append(f'  <div class="meta">{date_str}</div>')
        html.append(f"  <article>{html_paragraphs}</article>")
        if tags:
            tags_html = " ".join(f"<span>#{t}</span>" for t in tags)
            html.append(f'  <div class="tags">{tags_html}</div>')
        html += ["</body>", "</html>"]
        return "\n".join(html)

    def to_telegram(
        self,
        content: str,
        title: Optional[str] = None,
        tags: List[str] = [],
        meta: Dict[str, Any] = {},
    ) -> str:
        lines = []
        if title:
            lines.append(f"**{title}**")
            lines.append("")
        lines.append(content.strip())
        if tags:
            lines.append("")
            lines.append(" ".join(f"#{t.replace(' ', '_')}" for t in tags))
        return "\n".join(lines)

    def to_json(
        self,
        content: str,
        title: Optional[str] = None,
        tags: List[str] = [],
        meta: Dict[str, Any] = {},
    ) -> str:
        payload = {
            "title": title or "",
            "content": content,
            "tags": tags,
            "post_type": meta.get("post_type", ""),
            "word_count": len(content.split()),
            "char_count": len(content),
            "created_at": meta.get("created_at", datetime.now().isoformat()),
            "meta_description": meta.get("meta_description", ""),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def to_plain(
        self,
        content: str,
        title: Optional[str] = None,
        tags: List[str] = [],
        meta: Dict[str, Any] = {},
    ) -> str:
        lines = []
        if title:
            lines.append(title.upper())
            lines.append("=" * len(title))
            lines.append("")
        lines.append(content.strip())
        if tags:
            lines.append("")
            lines.append(f"Теги: {', '.join(tags)}")
        return "\n".join(lines)
