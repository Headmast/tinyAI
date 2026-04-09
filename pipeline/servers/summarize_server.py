"""
MCP Server 2: Summarize — суммаризация текста через LLM (День 19).

Инструмент: summarize
Протокол: MCP v2024-11-05, JSON-RPC 2.0, транспорт stdio.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.servers.base import MCPServerBase

# ── LLM client ────────────────────────────────────────────────────────────────

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
    explicit = os.getenv("LLM_MODEL", "")
    if explicit:
        return explicit
    provider = os.getenv("LLM_PROVIDER", "auto")
    if provider == "openai":
        return "gpt-4o-mini"
    if os.getenv("CLOUD_API_KEY", ""):
        return "zai-org/GLM-4.7-Flash"
    return "gpt-4o-mini"


LLM_MODEL = _get_model()

# ── Инструменты ───────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "summarize",
        "description": "Суммаризация текста через LLM. Стили: news, digest, brief.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Текст для суммаризации"},
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
]


def handle_summarize(args: dict) -> str:
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
                {"role": "user", "content": text[:2000]},  # 2k chars ≈ ~500 токенов входа
            ],
            temperature=0.3,
            max_tokens=350,  # краткая сводка не нуждается в большем
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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    server = MCPServerBase(
        name="tinyai-summarize-server",
        version="19.0.0",
        tools=TOOLS,
        handlers={"summarize": handle_summarize},
    )
    server.run()
