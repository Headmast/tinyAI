"""
Pipeline Templates — готовые цепочки инструментов для типичных задач.

Шаблоны превращают PipelineExecutor в удобный инструмент «в один клик»:
  - social_media_post  — поиск → суммаризация → формат(telegram) → сохранение
  - daily_digest       — поиск по логам → суммаризация(digest) → файл
  - research_report    — поиск → суммаризация(brief) → формат(website) → БД
  - content_publish    — поиск → суммаризация → формат → файл + БД

Использование:
    from pipeline_templates import get_template, list_templates, build_steps

    # Список доступных шаблонов
    for t in list_templates():
        print(t["name"], "—", t["description"])

    # Построить шаги для конкретной темы
    steps = build_steps("social_media_post", topic="Нейросети в 2026")

    # Запустить через PipelineExecutor
    from mcp_pipeline import PipelineExecutor
    result = PipelineExecutor().run(steps)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mcp_pipeline import Step


@dataclass
class PipelineTemplate:
    """Описание шаблона пайплайна."""
    name: str
    description: str
    tags: List[str]
    default_params: Dict[str, str]
    build: Callable[[Dict[str, str]], List[Step]]


# ══════════════════════════════════════════════════════════════════════════════
# Builders — функции, создающие список Step из параметров
# ══════════════════════════════════════════════════════════════════════════════

def _build_social_media_post(params: Dict[str, str]) -> List[Step]:
    topic = params["topic"]
    platform = params.get("platform", "telegram")
    style = params.get("style", "news")
    filename = params.get("filename", "social_post.md")
    return [
        Step(tool="search", args={"query": topic}, name="search"),
        Step(tool="summarize", args={"text": "$prev.result", "style": style}, name="summarize"),
        Step(tool="format_content", args={
            "content": "$prev.result",
            "platform": platform,
            "title": topic,
        }, name="format"),
        Step(tool="save_to_file", args={
            "content": "$prev.result",
            "filename": filename,
        }, name="save"),
    ]


def _build_daily_digest(params: Dict[str, str]) -> List[Step]:
    topic = params.get("topic", "")
    query = topic or "итоги дня"
    filename = params.get("filename", "daily_digest.md")
    return [
        Step(tool="search", args={"query": query}, name="search"),
        Step(tool="summarize", args={"text": "$prev.result", "style": "digest"}, name="digest"),
        Step(tool="save_to_file", args={
            "content": "$prev.result",
            "filename": filename,
        }, name="save"),
    ]


def _build_research_report(params: Dict[str, str]) -> List[Step]:
    topic = params["topic"]
    filename = params.get("filename", "report.html")
    return [
        Step(tool="search", args={"query": topic}, name="search"),
        Step(tool="summarize", args={"text": "$prev.result", "style": "brief"}, name="summarize"),
        Step(tool="format_content", args={
            "content": "$prev.result",
            "platform": "website",
            "title": topic,
        }, name="format"),
        Step(tool="save_to_file", args={
            "content": "$prev.result",
            "filename": filename,
        }, name="save_file"),
        Step(tool="save_to_db", args={
            "title": topic,
            "content": "$steps.2.result",
            "platform": "website",
            "summary": "$steps.1.result",
        }, name="save_db"),
    ]


def _build_content_publish(params: Dict[str, str]) -> List[Step]:
    topic = params["topic"]
    platform = params.get("platform", "telegram")
    style = params.get("style", "news")
    filename = params.get("filename", "published.md")
    return [
        Step(tool="search", args={"query": topic}, name="search"),
        Step(tool="summarize", args={"text": "$prev.result", "style": style}, name="summarize"),
        Step(tool="format_content", args={
            "content": "$prev.result",
            "platform": platform,
            "title": topic,
        }, name="format"),
        Step(tool="save_to_file", args={
            "content": "$prev.result",
            "filename": filename,
        }, name="save_file"),
        Step(tool="save_to_db", args={
            "title": topic,
            "content": "$steps.2.result",
            "platform": platform,
            "summary": "$steps.1.result",
        }, name="save_db"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Реестр шаблонов
# ══════════════════════════════════════════════════════════════════════════════

TEMPLATES: Dict[str, PipelineTemplate] = {
    "social_media_post": PipelineTemplate(
        name="social_media_post",
        description="Поиск → суммаризация → формат для соцсети → сохранение в файл",
        tags=["social", "telegram", "quick"],
        default_params={"platform": "telegram", "style": "news", "filename": "social_post.md"},
        build=_build_social_media_post,
    ),
    "daily_digest": PipelineTemplate(
        name="daily_digest",
        description="Поиск по теме → дайджест → сохранение в файл",
        tags=["digest", "summary", "daily"],
        default_params={"filename": "daily_digest.md"},
        build=_build_daily_digest,
    ),
    "research_report": PipelineTemplate(
        name="research_report",
        description="Поиск → краткий анализ → формат(website) → файл + БД",
        tags=["research", "report", "website"],
        default_params={"filename": "report.html"},
        build=_build_research_report,
    ),
    "content_publish": PipelineTemplate(
        name="content_publish",
        description="Полный цикл: поиск → суммаризация → формат → файл + БД",
        tags=["publish", "full"],
        default_params={"platform": "telegram", "style": "news", "filename": "published.md"},
        build=_build_content_publish,
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Публичный API
# ══════════════════════════════════════════════════════════════════════════════

def list_templates() -> List[Dict[str, Any]]:
    """Список всех доступных шаблонов с описаниями."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "tags": t.tags,
            "default_params": t.default_params,
        }
        for t in TEMPLATES.values()
    ]


def get_template(name: str) -> Optional[PipelineTemplate]:
    """Получить шаблон по имени. None если не найден."""
    return TEMPLATES.get(name)


def build_steps(template_name: str, **params) -> List[Step]:
    """
    Построить список Step из шаблона с переданными параметрами.

    Args:
        template_name: Имя шаблона (social_media_post, daily_digest, и т.д.)
        **params: Параметры шаблона (topic, platform, style, filename)

    Returns:
        Список Step для передачи в PipelineExecutor.run()

    Raises:
        ValueError: Если шаблон не найден или отсутствует обязательный параметр.
    """
    template = TEMPLATES.get(template_name)
    if not template:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Шаблон '{template_name}' не найден. Доступные: {available}")

    # Мержим default_params с переданными
    merged = {**template.default_params, **params}

    # Проверяем topic для шаблонов, где он обязателен
    if template_name != "daily_digest" and "topic" not in merged:
        raise ValueError(f"Шаблон '{template_name}' требует параметр 'topic'")

    return template.build(merged)
