"""
Промпты для каждого шага каждого типа контента.

Промпты формируются через _build_context(task), который извлекает
плоский словарь из step_results. Плейсхолдеры вида {key} заменяются
значениями, недостающие ключи — заглушкой "(нет данных)".

Структура STEP_PROMPTS:
  {content_type_value: {state_value: template_string}}
"""

from __future__ import annotations

import json
from typing import Any, Dict

from journalist_agent.workflow import ContentType, JournalistTask


# ─────────────────────────────────────────────────────────────
# Системный промпт (общий для всех шагов)
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Ты — профессиональный журналист и редактор. "
    "Ты работаешь строго поэтапно — выполняешь ровно один шаг рабочего процесса "
    "и возвращаешь результат в точно указанном формате. "
    "Не выполняй шаги, которые ещё не наступили. "
    "Если ожидается JSON — верни только JSON без пояснений вне блока."
)


# ─────────────────────────────────────────────────────────────
# Шаблоны промптов
# ─────────────────────────────────────────────────────────────

STEP_PROMPTS: Dict[str, Dict[str, str]] = {

    # ── ARTICLE ──────────────────────────────────────────────
    ContentType.ARTICLE.value: {
        "planning": (
            "Составь детальный план статьи на тему: «{topic}».\n\n"
            "Верни строго JSON (без текста вне JSON):\n"
            "{{\n"
            '  "title": "заголовок статьи",\n'
            '  "subtitle": "подзаголовок (опционально)",\n'
            '  "target_audience": "целевая аудитория",\n'
            '  "tone": "analytical|educational|engaging|critical",\n'
            '  "word_count_target": 1200,\n'
            '  "sections": [\n'
            '    {{"name": "Введение", "key_points": ["тезис 1"], "word_count": 150}},\n'
            '    {{"name": "Раздел 1", "key_points": ["..."], "word_count": 300}},\n'
            '    {{"name": "Заключение", "key_points": ["..."], "word_count": 150}}\n'
            "  ]\n"
            "}}"
        ),
        "research": (
            "Проведи журналистское исследование для статьи «{title}».\n\n"
            "Структура статьи (план):\n{plan_summary}\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "key_facts": ["факт 1", "факт 2", "факт 3"],\n'
            '  "data_points": ["цифра/статистика 1", "цифра 2"],\n'
            '  "angles": ["угол подачи 1", "угол 2"],\n'
            '  "sources": ["источник 1", "источник 2"],\n'
            '  "context": "краткий контекст темы (2-3 предложения)"\n'
            "}}"
        ),
        "drafting": (
            "Напиши полный текст статьи «{title}».\n\n"
            "Используй план:\n{plan_summary}\n\n"
            "Используй исследование:\n{research_summary}\n\n"
            "Требования:\n"
            "- Структура: ## для каждого раздела\n"
            "- Тональность: {tone}\n"
            "- Целевой объём: {word_count_target} слов\n"
            "- Только текст статьи, никакого JSON\n\n"
            "Напиши полный текст статьи:"
        ),
        "editing": (
            "Отредактируй черновик статьи «{title}».\n\n"
            "Черновик:\n{draft_text}\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "edited_text": "ПОЛНЫЙ отредактированный текст статьи",\n'
            '  "changes": ["изменение 1", "изменение 2"],\n'
            '  "word_count": 1200\n'
            "}}"
        ),
        "validation": (
            "Проверь статью «{title}» на качество и достоверность.\n\n"
            "Статья:\n{edited_text}\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "quality_score": 8,\n'
            '  "fact_accuracy": 9,\n'
            '  "structure_score": 8,\n'
            '  "issues": ["замечание 1 (или пустой список)"],\n'
            '  "approved": true\n'
            "}}"
        ),
    },

    # ── NEWS_RESEARCH ─────────────────────────────────────────
    ContentType.NEWS_RESEARCH.value: {
        "topic_selection": (
            "Определи тему новостного исследования: «{topic}».\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "topic": "точная формулировка темы",\n'
            '  "relevance": "почему это важно сейчас",\n'
            '  "audience": "целевая аудитория",\n'
            '  "news_angle": "главный угол подачи",\n'
            '  "time_frame": "временной контекст (последние N дней/месяцев)"\n'
            "}}"
        ),
        "source_gathering": (
            "Собери источники для исследования темы «{topic}».\n\n"
            "Контекст темы: {relevance}\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "sources": [\n'
            '    {{"name": "название источника", "type": "news|research|official|blog", "credibility": "high|medium|low"}}\n'
            "  ],\n"
            '  "primary_sources": ["самые важные источники"],\n'
            '  "coverage_gaps": ["чего не хватает в публичных источниках"]\n'
            "}}"
        ),
        "analysis": (
            "Проанализируй данные по теме «{topic}».\n\n"
            "Источники: {sources_summary}\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "findings": ["вывод 1", "вывод 2", "вывод 3"],\n'
            '  "key_insights": ["инсайт 1", "инсайт 2"],\n'
            '  "contradictions": ["противоречие 1 (или пустой список)"],\n'
            '  "timeline": ["событие 1 (дата — что произошло)"],\n'
            '  "significance": "почему это важно"\n'
            "}}"
        ),
        "drafting": (
            "Напиши новостное исследование на тему «{topic}».\n\n"
            "Угол подачи: {news_angle}\n"
            "Ключевые выводы: {findings_summary}\n\n"
            "Требования:\n"
            "- Структура: заголовок, лид, основной текст с подзаголовками, вывод\n"
            "- Нейтральный журналистский стиль\n"
            "- Опирайся на собранные источники\n\n"
            "Напиши полный текст исследования:"
        ),
        "fact_check": (
            "Проверь факты в новостном исследовании «{topic}».\n\n"
            "Текст:\n{draft_text}\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "verified_claims": ["утверждение 1 — подтверждено"],\n'
            '  "disputed_claims": ["утверждение X — требует уточнения"],\n'
            '  "fact_check_score": 8,\n'
            '  "corrections": ["исправление 1 (или пустой список)"],\n'
            '  "approved": true\n'
            "}}"
        ),
    },

    # ── REVIEW ────────────────────────────────────────────────
    ContentType.REVIEW.value: {
        "subject_definition": (
            "Определи объект ревью: «{topic}».\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "subject_name": "точное название объекта",\n'
            '  "subject_type": "книга|фильм|продукт|сервис|статья|событие|другое",\n'
            '  "context": "краткий контекст (автор/создатель, дата выхода, жанр)",\n'
            '  "review_scope": "что именно будет оцениваться",\n'
            '  "reviewer_perspective": "с позиции какого читателя делается ревью"\n'
            "}}"
        ),
        "criteria_setting": (
            "Разработай систему критериев для ревью «{subject_name}».\n\n"
            "Область оценки: {review_scope}\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "criteria": [\n'
            '    {{"name": "критерий 1", "description": "что именно оценивается", "weight": 0.3}}\n'
            "  ],\n"
            '  "scoring_scale": "1-10",\n'
            '  "must_have": ["обязательный элемент 1"]\n'
            "}}"
        ),
        "examination": (
            "Проведи детальный анализ «{subject_name}» по установленным критериям.\n\n"
            "Критерии оценки: {criteria_summary}\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "scores": {{"критерий 1": 8, "критерий 2": 7}},\n'
            '  "strengths": ["сильная сторона 1", "сильная сторона 2"],\n'
            '  "weaknesses": ["слабая сторона 1"],\n'
            '  "notable_moments": ["важный момент 1"],\n'
            '  "overall_impression": "общее впечатление (2-3 предложения)"\n'
            "}}"
        ),
        "drafting": (
            "Напиши ревью «{subject_name}».\n\n"
            "Анализ: {examination_summary}\n"
            "Критерии: {criteria_summary}\n\n"
            "Требования:\n"
            "- Структура: вводный параграф, анализ по критериям, итоговая оценка\n"
            "- Обоснованный, аналитический тон\n"
            "- Конкретные примеры\n\n"
            "Напиши полный текст ревью:"
        ),
        "validation": (
            "Проверь качество и объективность ревью «{subject_name}».\n\n"
            "Текст ревью:\n{draft_text}\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "overall_score": 8.5,\n'
            '  "objectivity_score": 9,\n'
            '  "coverage_score": 8,\n'
            '  "verdict": "Рекомендуется|Нейтрально|Не рекомендуется",\n'
            '  "issues": ["замечание (или пустой список)"],\n'
            '  "approved": true\n'
            "}}"
        ),
    },

    # ── NOTE ──────────────────────────────────────────────────
    ContentType.NOTE.value: {
        "idea_capture": (
            "Зафикcируй и структурируй идею для заметки: «{topic}».\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "core_idea": "главная мысль одним предложением",\n'
            '  "context": "зачем это важно/интересно",\n'
            '  "key_points": ["тезис 1", "тезис 2"],\n'
            '  "tone": "personal|analytical|observational|humorous",\n'
            '  "target_length": "short (100-200 слов)"\n'
            "}}"
        ),
        "drafting": (
            "Напиши заметку по идее «{core_idea}».\n\n"
            "Ключевые тезисы: {key_points_text}\n"
            "Тональность: {tone}\n\n"
            "Требования:\n"
            "- Короткий связный текст (100-250 слов)\n"
            "- Без формальной структуры с заголовками разделов\n"
            "- Живой, непосредственный стиль\n\n"
            "Напиши текст заметки:"
        ),
    },

    # ── BLOG_ANALYSIS ─────────────────────────────────────────
    ContentType.BLOG_ANALYSIS.value: {
        "blog_selection": (
            "Определи объект анализа — блог «{topic}».\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "blog_name": "название блога",\n'
            '  "url": "адрес (если известен)",\n'
            '  "focus_areas": ["на что обратить внимание"],\n'
            '  "analysis_goals": "цель анализа — что хотим узнать",\n'
            '  "comparison_context": "с чем сравниваем или зачем изучаем"\n'
            "}}"
        ),
        "metrics_collection": (
            "Собери количественные метрики блога «{blog_name}».\n\n"
            "Цели анализа: {analysis_goals}\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "post_frequency": "как часто выходят посты",\n'
            '  "avg_length": "средний объём материала",\n'
            '  "topics": ["основные темы"],\n'
            '  "content_formats": ["форматы: статьи/видео/кейсы/etc"],\n'
            '  "engagement_signals": ["признаки вовлечённости аудитории"],\n'
            '  "posting_pattern": "когда и как часто"\n'
            "}}"
        ),
        "content_analysis": (
            "Проведи качественный анализ контента блога «{blog_name}».\n\n"
            "Метрики: {metrics_summary}\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "writing_style": "описание стиля письма",\n'
            '  "content_patterns": ["повторяющиеся паттерны в контенте"],\n'
            '  "niche_positioning": "как блог позиционирует себя в нише",\n'
            '  "audience_fit": "насколько контент соответствует аудитории",\n'
            '  "unique_value": "уникальная ценность блога",\n'
            '  "tone_and_voice": "тон и голос блога"\n'
            "}}"
        ),
        "insights_extraction": (
            "Извлеки применимые инсайты из анализа блога «{blog_name}».\n\n"
            "Анализ контента: {content_analysis_summary}\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "insights": ["инсайт 1", "инсайт 2", "инсайт 3"],\n'
            '  "lessons": ["урок 1 — что можно применить"],\n'
            '  "what_works": ["что работает у этого блога"],\n'
            '  "what_doesnt": ["что можно было бы улучшить"],\n'
            '  "key_takeaway": "главный вывод одним предложением"\n'
            "}}"
        ),
        "drafting": (
            "Напиши аналитический отчёт по блогу «{blog_name}».\n\n"
            "Цели: {analysis_goals}\n"
            "Ключевые инсайты: {insights_summary}\n\n"
            "Требования:\n"
            "- Структура: введение, метрики, анализ, выводы, рекомендации\n"
            "- Аналитический тон, конкретные примеры\n"
            "- ## для заголовков разделов\n\n"
            "Напиши полный текст отчёта:"
        ),
        "validation": (
            "Проверь полноту и обоснованность анализа блога «{blog_name}».\n\n"
            "Отчёт:\n{draft_text}\n\n"
            "Верни строго JSON:\n"
            "{{\n"
            '  "completeness_score": 8,\n'
            '  "bias_check": "есть ли предвзятость в оценках",\n'
            '  "evidence_strength": 7,\n'
            '  "missing_aspects": ["что не охвачено (или пустой список)"],\n'
            '  "approved": true\n'
            "}}"
        ),
    },
}


# ─────────────────────────────────────────────────────────────
# Контекст для подстановки в промпт
# ─────────────────────────────────────────────────────────────

def _safe_json_preview(data: Any, max_len: int = 600) -> str:
    """Краткое JSON-представление данных для инъекции в промпт."""
    if isinstance(data, str):
        return data[:max_len]
    try:
        s = json.dumps(data, ensure_ascii=False, indent=2)
        return s[:max_len] + ("..." if len(s) > max_len else "")
    except Exception:
        return str(data)[:max_len]


def _build_context(task: JournalistTask) -> Dict[str, str]:
    """
    Извлекает плоский словарь переменных из task для подстановки в промпт.

    Ключи: все верхнеуровневые ключи из step_results, плюс составные
    _summary-версии для сложных структур.
    """
    ctx: Dict[str, str] = {
        "topic": task.topic or "(тема не указана)",
    }

    sr = task.step_results

    # ── Общие поля ─────────────────────────────────────────
    for key, val in sr.items():
        if isinstance(val, str):
            ctx[key] = val
        elif isinstance(val, list):
            ctx[key] = _safe_json_preview(val)
        elif isinstance(val, dict):
            ctx[key] = _safe_json_preview(val)
        else:
            ctx[key] = str(val)

    # ── Составные summary-ключи ───────────────────────────

    # plan_summary: sections из step planning
    if "sections" in sr:
        secs = sr["sections"]
        if isinstance(secs, list):
            ctx["plan_summary"] = "\n".join(
                f"  - {s.get('name', '?')}: {', '.join(s.get('key_points', []))}"
                for s in secs[:6]
            )
    elif "title" in sr:
        ctx.setdefault("plan_summary", f"Тема: {sr.get('title', task.topic)}")

    # research_summary
    if "key_facts" in sr:
        facts = sr["key_facts"]
        ctx["research_summary"] = "\n".join(f"  • {f}" for f in facts[:5])

    # sources_summary
    if "sources" in sr:
        src = sr["sources"]
        if isinstance(src, list):
            names = [
                s.get("name", s) if isinstance(s, dict) else str(s)
                for s in src[:5]
            ]
            ctx["sources_summary"] = "; ".join(names)

    # findings_summary
    if "findings" in sr:
        ctx["findings_summary"] = "\n".join(f"  • {f}" for f in sr["findings"][:4])

    # criteria_summary
    if "criteria" in sr:
        crit = sr["criteria"]
        if isinstance(crit, list):
            ctx["criteria_summary"] = "\n".join(
                f"  - {c.get('name', '?')}: {c.get('description', '')}"
                for c in crit[:6]
            )

    # examination_summary
    if "scores" in sr:
        scores = sr["scores"]
        if isinstance(scores, dict):
            ctx["examination_summary"] = ", ".join(
                f"{k}: {v}" for k, v in list(scores.items())[:5]
            )
        ctx.setdefault("examination_summary", _safe_json_preview(sr.get("scores", {})))

    # metrics_summary
    if "topics" in sr or "post_frequency" in sr:
        ctx["metrics_summary"] = _safe_json_preview({
            k: sr[k] for k in ("post_frequency", "avg_length", "topics", "content_formats")
            if k in sr
        })

    # content_analysis_summary
    if "writing_style" in sr:
        ctx["content_analysis_summary"] = _safe_json_preview({
            k: sr[k] for k in ("writing_style", "content_patterns", "niche_positioning", "unique_value")
            if k in sr
        })

    # insights_summary
    if "insights" in sr:
        ctx["insights_summary"] = "\n".join(f"  • {i}" for i in sr["insights"][:4])

    # key_points_text
    if "key_points" in sr:
        ctx["key_points_text"] = "; ".join(sr["key_points"][:5])

    # defaults for template placeholders
    ctx.setdefault("title", sr.get("title", task.topic or "(без заголовка)"))
    ctx.setdefault("tone", sr.get("tone", "аналитический"))
    ctx.setdefault("word_count_target", str(sr.get("word_count_target", 1200)))
    ctx.setdefault("subject_name", sr.get("subject_name", task.topic or "(объект)"))
    ctx.setdefault("blog_name", sr.get("blog_name", task.topic or "(блог)"))
    ctx.setdefault("analysis_goals", sr.get("analysis_goals", "(цели не определены)"))
    ctx.setdefault("news_angle", sr.get("news_angle", "(угол не определён)"))
    ctx.setdefault("relevance", sr.get("relevance", ""))
    ctx.setdefault("core_idea", sr.get("core_idea", task.topic or "(идея)"))
    ctx.setdefault("draft_text", sr.get("draft_text", sr.get("edited_text", "(черновик не написан)")))
    ctx.setdefault("edited_text", sr.get("edited_text", sr.get("draft_text", "(черновик не написан)")))
    ctx.setdefault("plan_summary", "(план не составлен)")
    ctx.setdefault("research_summary", "(исследование не проводилось)")
    ctx.setdefault("sources_summary", "(источники не собраны)")
    ctx.setdefault("findings_summary", "(анализ не проведён)")
    ctx.setdefault("criteria_summary", "(критерии не установлены)")
    ctx.setdefault("examination_summary", "(объект не изучен)")
    ctx.setdefault("metrics_summary", "(метрики не собраны)")
    ctx.setdefault("content_analysis_summary", "(анализ не проведён)")
    ctx.setdefault("insights_summary", "(выводы не извлечены)")
    ctx.setdefault("key_points_text", "(тезисы не определены)")

    return ctx


def build_step_prompt(task: JournalistTask, state: str) -> str:
    """
    Строит финальный промпт для указанного шага задачи.

    Args:
        task:  текущая задача (с накопленными step_results)
        state: состояние, для которого нужен промпт

    Returns:
        Готовый промпт для передачи в LLM.
    """
    ct_val = task.content_type.value
    templates = STEP_PROMPTS.get(ct_val, {})
    template = templates.get(state, f"Выполни шаг «{state}» для задачи «{task.topic}». Верни JSON.")

    ctx = _build_context(task)

    # Безопасная подстановка: незнакомые ключи оставляем как есть
    try:
        return template.format_map(ctx)
    except KeyError:
        return template
