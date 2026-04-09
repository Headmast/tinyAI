"""
MCP Pipeline Server — сервер контент-пайплайна (День 19).

Инструменты:
  search         — поиск по логам проекта
  summarize      — суммаризация текста через LLM
  format_content — форматирование для платформы (telegram, website, rss, plain)
  save_to_file   — сохранение контента в файл
  save_to_db     — сохранение контента в SQLite

Протокол: MCP v2024-11-05, JSON-RPC 2.0, транспорт stdio.
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
POSTS_DIR = BASE_DIR / "pipeline_data" / "posts"

# ── Lazy LLM client ──────────────────────────────────────────────────────────

_llm_client = None

def _get_llm():
    global _llm_client
    if _llm_client is None:
        provider = os.getenv("LLM_PROVIDER", "auto")
        cloud_key = os.getenv("CLOUD_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")

        if provider == "openai" or (provider == "auto" and not cloud_key and openai_key):
            _llm_client = OpenAI(api_key=openai_key, timeout=120.0)
        elif cloud_key:
            _llm_client = OpenAI(
                api_key=cloud_key,
                base_url="https://foundation-models.api.cloud.ru/v1",
                timeout=120.0,
            )
        else:
            _llm_client = OpenAI(api_key=openai_key, timeout=120.0)
    return _llm_client

def _get_model():
    provider = os.getenv("LLM_PROVIDER", "auto")
    explicit = os.getenv("LLM_MODEL", "")
    if explicit:
        return explicit
    if provider == "openai":
        return "gpt-4o-mini"
    cloud_key = os.getenv("CLOUD_API_KEY", "")
    if cloud_key:
        return "zai-org/GLM-4.7-Flash"
    return "gpt-4o-mini"

LLM_MODEL = _get_model()

# ── Определения инструментов ─────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search",
        "description": "Поиск по логам разговоров проекта. Возвращает найденные фрагменты.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Текст для поиска",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "summarize",
        "description": "Суммаризация текста через LLM. Поддерживает стили: news, digest, brief.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Текст для суммаризации",
                },
                "style": {
                    "type": "string",
                    "description": "Стиль: news (новостной), digest (дайджест), brief (краткий)",
                    "enum": ["news", "digest", "brief"],
                    "default": "news",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "format_content",
        "description": "Форматирование контента для целевой платформы.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Контент для форматирования",
                },
                "platform": {
                    "type": "string",
                    "description": "Платформа: telegram, website, rss, plain",
                    "enum": ["telegram", "website", "rss", "plain"],
                },
                "title": {
                    "type": "string",
                    "description": "Заголовок (опционально)",
                },
            },
            "required": ["content", "platform"],
        },
    },
    {
        "name": "save_to_file",
        "description": "Сохранение контента в файл в директории pipeline_data/posts/.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Контент для сохранения",
                },
                "filename": {
                    "type": "string",
                    "description": "Имя файла (без пути, только имя + расширение)",
                },
            },
            "required": ["content", "filename"],
        },
    },
    {
        "name": "save_to_db",
        "description": "Сохранение контента в базу данных SQLite (pipeline_data/content.db).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Заголовок контента",
                },
                "content": {
                    "type": "string",
                    "description": "Основной текст контента",
                },
                "platform": {
                    "type": "string",
                    "description": "Целевая платформа",
                    "enum": ["telegram", "website", "rss", "plain"],
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Теги/ключевые слова",
                    "default": [],
                },
                "summary": {
                    "type": "string",
                    "description": "Краткое описание (опционально)",
                },
                "source_query": {
                    "type": "string",
                    "description": "Исходный поисковый запрос (опционально)",
                },
                "pipeline_run": {
                    "type": "string",
                    "description": "ID запуска пайплайна (опционально)",
                },
            },
            "required": ["title", "content", "platform"],
        },
    },
]

# ── Обработчики инструментов ─────────────────────────────────────────────────

def handle_search(args: dict) -> str:
    """Поиск по лог-файлам проекта."""
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
                    # Поддержка обоих форматов логов
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
                            "role": msg.get("role", msg.get("user_input", "")[:20] and "dialog"),
                            "snippet": content[:500],
                        })
            except (json.JSONDecodeError, OSError):
                continue

    return json.dumps({
        "query": query,
        "found": len(results),
        "results": results[:20],
    }, ensure_ascii=False)


def handle_summarize(args: dict) -> str:
    """Суммаризация текста через LLM."""
    text = args.get("text", "").strip()
    if not text:
        return json.dumps({"error": "text is required"}, ensure_ascii=False)

    style = args.get("style", "news")
    style_prompts = {
        "news": "Напиши краткую новостную сводку по данному тексту. Стиль — информационное агентство, без воды.",
        "digest": "Составь дайджест из ключевых пунктов данного текста. Формат — нумерованный список.",
        "brief": "Сократи текст до 2-3 предложений, сохранив главную мысль.",
    }
    system_prompt = style_prompts.get(style, style_prompts["news"])

    try:
        client = _get_llm()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text[:4000]},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        summary = (response.choices[0].message.content or "").strip()
        if not summary:
            summary = "Не удалось суммаризировать текст."
        return json.dumps({
            "summary": summary,
            "style": style,
            "original_length": len(text),
            "summary_length": len(summary),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"LLM error: {e}"}, ensure_ascii=False)


def handle_format_content(args: dict) -> str:
    """Форматирование контента для платформы."""
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


def handle_save_to_file(args: dict) -> str:
    """Сохранение контента в файл."""
    content = args.get("content", "")
    filename = args.get("filename", "").strip()
    if not filename:
        return json.dumps({"error": "filename is required"}, ensure_ascii=False)

    # Защита от path traversal
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
    """Сохранение контента в базу данных."""
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


# ── Диспетчер ─────────────────────────────────────────────────────────────────

TOOL_HANDLERS = {
    "search": handle_search,
    "summarize": handle_summarize,
    "format_content": handle_format_content,
    "save_to_file": handle_save_to_file,
    "save_to_db": handle_save_to_db,
}

# ── MCP JSON-RPC ──────────────────────────────────────────────────────────────

SERVER_INFO = {"name": "tinyai-pipeline-server", "version": "19.0.0"}
PROTOCOL_VERSION = "2024-11-05"


def handle_request(req: dict) -> dict:
    """Обрабатывает JSON-RPC запрос."""
    method = req.get("method", "")
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        }

    if method == "notifications/initialized":
        return None  # no response for notifications

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True,
                },
            }
        try:
            result_text = handler(arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


# ── Main loop (stdio) ────────────────────────────────────────────────────────

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
