"""
ArticleFSMAgent — агент написания статей с конечным автоматом (FSM).

Реализует формализованное состояние задачи:
  planning → execution → validation → done

На каждом этапе агент знает:
  - текущий этап (phase)
  - текущий шаг (current_step)
  - ожидаемое действие (expected_action)

Поддерживает паузу и продолжение без повторных объяснений.
Модель: GLM-4.7 (Cloud.ru)
"""

import json
import signal
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from news_agent.fsm_state import (
    TaskPhase,
    TaskState,
    TaskStateStorage,
    PHASE_STEPS,
    TOTAL_STEPS,
)


SYSTEM_PROMPT = """Ты — AI-редактор, пишущий статьи по заданной теме.
Ты работаешь строго поэтапно: планирование → написание → проверка.
На каждом шаге ты выполняешь одно конкретное действие и возвращаешь результат в JSON.
Не повторяй пройденные шаги — продолжай именно с того места, где остановился."""


STEP_PROMPTS: Dict[str, str] = {
    "choose_topic": (
        "Предложи одну актуальную тему для статьи в сфере технологий или AI.\n"
        "Верни строго JSON (без пояснений вне JSON):\n"
        "{\n"
        '  "topic": "название темы",\n'
        '  "rationale": "почему эта тема актуальна сейчас",\n'
        '  "target_audience": "на кого рассчитана статья",\n'
        '  "article_type": "analysis|tutorial|news|opinion"\n'
        "}"
    ),
    "analyze_topic": (
        'Проанализируй тему: «{topic}».\n'
        "Верни строго JSON:\n"
        "{\n"
        '  "main_angle": "главный угол подачи материала",\n'
        '  "key_questions": ["вопрос 1", "вопрос 2", "вопрос 3"],\n'
        '  "key_facts": ["тезис 1", "тезис 2", "тезис 3"],\n'
        '  "tone": "analytical|educational|engaging|critical",\n'
        '  "word_count_target": 800\n'
        "}"
    ),
    "create_outline": (
        'На основе анализа создай детальный план статьи о «{topic}».\n'
        "Контекст анализа: {analysis_context}\n"
        "Верни строго JSON:\n"
        "{\n"
        '  "title": "заголовок статьи",\n'
        '  "subtitle": "подзаголовок (опционально)",\n'
        '  "sections": [\n'
        '    {"name": "Введение", "key_points": ["тезис 1", "тезис 2"], "word_count": 150},\n'
        '    {"name": "Раздел 1", "key_points": ["..."], "word_count": 200},\n'
        '    {"name": "Раздел 2", "key_points": ["..."], "word_count": 200},\n'
        '    {"name": "Заключение", "key_points": ["..."], "word_count": 150}\n'
        "  ],\n"
        '  "total_word_count": 800\n'
        "}"
    ),
    "write_intro": (
        'Напиши вступление для статьи «{title}».\n'
        "Структура статьи: {outline_context}\n"
        "Верни строго JSON:\n"
        "{\n"
        '  "intro_text": "полный текст вступления (150-200 слов)",\n'
        '  "hook": "первое предложение-крючок",\n'
        '  "word_count": 150\n'
        "}"
    ),
    "write_body": (
        'Напиши основной текст статьи «{title}» по разделам плана.\n'
        "Разделы плана: {sections_context}\n"
        "Вступление уже написано (не повторяй его): {intro_text}\n\n"
        "Требования:\n"
        "- Используй ## для заголовка каждого раздела\n"
        "- Пиши связный журналистский текст, 3-5 предложений на раздел\n"
        "- Только текст статьи — без JSON, без пояснений, без markdown-кода\n"
        "- НЕ пиши Введение и Заключение (они пишутся отдельно)"
    ),
    "write_conclusion": (
        'Напиши заключение для статьи «{title}».\n'
        "Краткое содержание написанных разделов: {body_summary}\n"
        "Верни строго JSON:\n"
        "{\n"
        '  "conclusion_text": "полный текст заключения (100-150 слов)",\n'
        '  "key_takeaway": "главная мысль одним предложением",\n'
        '  "word_count": 120\n'
        "}"
    ),
    "check_structure": (
        'Проверь структуру написанной статьи «{title}».\n'
        "Статья:\n{full_article}\n"
        "Верни строго JSON:\n"
        "{\n"
        '  "has_intro": true,\n'
        '  "has_body": true,\n'
        '  "has_conclusion": true,\n'
        '  "section_count": 4,\n'
        '  "issues": [],\n'
        '  "structure_score": 8\n'
        "}"
    ),
    "score_quality": (
        'Оцени качество статьи «{title}».\n'
        "Статья:\n{full_article}\n"
        "Верни строго JSON:\n"
        "{\n"
        '  "scores": {\n'
        '    "clarity": 8,\n'
        '    "accuracy": 7,\n'
        '    "engagement": 8,\n'
        '    "structure": 9\n'
        "  },\n"
        '  "overall_score": 8,\n'
        '  "strengths": ["сильная сторона 1", "сильная сторона 2"],\n'
        '  "improvements": ["рекомендация 1", "рекомендация 2"]\n'
        "}"
    ),
    "final_edit": (
        'Выполни финальное редактирование статьи «{title}».\n'
        "Текущая статья:\n{full_article}\n"
        "Оценки качества: {quality_scores}\n"
        "Верни строго JSON:\n"
        "{\n"
        '  "final_article": "ПОЛНЫЙ отредактированный текст статьи",\n'
        '  "changes": ["изменение 1", "изменение 2"],\n'
        '  "final_word_count": 800\n'
        "}"
    ),
}


def _build_full_article(article_data: Dict[str, Any]) -> str:
    """Собирает полный текст статьи из накопленных данных."""
    parts: List[str] = []

    title = article_data.get("title", article_data.get("topic", "Статья"))
    parts.append(f"# {title}")

    subtitle = article_data.get("subtitle", "")
    if subtitle:
        parts.append(f"_{subtitle}_")

    intro = article_data.get("intro_text", "")
    if intro:
        parts.append(intro)

    for section in article_data.get("sections", []):
        name = section.get("name", "")
        text = section.get("text", "")
        if name:
            parts.append(f"## {name}")
        if text:
            parts.append(text)

    conclusion = article_data.get("conclusion_text", "")
    if conclusion:
        parts.append("## Заключение")
        parts.append(conclusion)

    return "\n\n".join(parts)


def _build_step_prompt(step_name: str, article_data: Dict[str, Any]) -> str:
    """Строит промпт для шага, подставляя накопленные данные."""
    template = STEP_PROMPTS.get(step_name, f"Выполни шаг {step_name}. Верни JSON.")

    ctx: Dict[str, Any] = dict(article_data)
    ctx.setdefault("topic", "технологии")
    ctx.setdefault("title", ctx.get("topic", "Статья"))
    ctx.setdefault(
        "analysis_context",
        json.dumps(
            {
                k: article_data[k]
                for k in ("main_angle", "key_questions", "key_facts", "tone")
                if k in article_data
            },
            ensure_ascii=False,
        ),
    )
    ctx.setdefault(
        "outline_context",
        json.dumps(article_data.get("sections", []), ensure_ascii=False)[:1200],
    )
    ctx.setdefault(
        "sections_context",
        json.dumps(
            [
                {"name": s.get("name", ""), "key_points": s.get("key_points", [])}
                for s in article_data.get("sections", [])
            ],
            ensure_ascii=False,
        )[:1200],
    )
    ctx.setdefault("intro_text", article_data.get("intro_text", "")[:600])
    ctx.setdefault(
        "body_summary",
        json.dumps(
            [
                {"name": s.get("name", ""), "snippet": s.get("text", "")[:80]}
                for s in article_data.get("sections", [])
            ],
            ensure_ascii=False,
        )[:800],
    )
    ctx.setdefault("full_article", _build_full_article(article_data)[:2500])
    ctx.setdefault(
        "quality_scores",
        json.dumps(article_data.get("scores", {}), ensure_ascii=False),
    )

    result = template
    for key, value in ctx.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


class ArticleFSMAgent:
    """
    Агент написания статей с конечным автоматом.

    Состояния: planning → execution → validation → done

    Запуск нового задания:
        agent.run(topic="Тема на выбор агента")

    Продолжение после паузы:
        agent.run(task_id="<id из предыдущего запуска>")

    Пауза поддерживается:
        - вручную через Ctrl+C (перехватывается как graceful pause)
        - программно через agent.pause(state)
        - автоматически при ошибке шага
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "zai-org/GLM-4.7",
        storage: Optional[TaskStateStorage] = None,
        verbose: bool = True,
        max_completion_tokens: int = 3000,
        temperature: float = 0.6,
    ) -> None:
        self.client = client
        self.model = model
        self.storage = storage or TaskStateStorage()
        self.verbose = verbose
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self._interrupted = False

    def run(
        self,
        state: Optional[TaskState] = None,
        task_id: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> TaskState:
        """
        Запускает или продолжает задачу.

        Приоритет аргументов:
          1. state  — готовый объект TaskState
          2. task_id — загрузить состояние из хранилища
          3. (новая задача) — topic опционален; если не задан — агент выберет сам

        Returns:
            Финальное состояние задачи после завершения или паузы.
        """
        if state is None and task_id:
            state = self.storage.load(task_id)
            if state is None:
                raise ValueError(f"Задача '{task_id}' не найдена в хранилище")

        is_resume = state is not None
        if state is None:
            state = TaskState.new(topic=topic)
        else:
            state.resume()

        self._interrupted = False
        self._setup_interrupt_handler(state)

        if self.verbose:
            self._print_run_header(state, is_resume)

        while state.phase != TaskPhase.DONE and not self._interrupted:
            step_info = state.get_current_step_info()
            if step_info is None:
                break

            if self.verbose:
                self._print_step_header(state, step_info)

            result = self._execute_step(state, step_info["name"])

            if result is None:
                if self.verbose:
                    print(f"\n  ❌ Шаг {step_info['name']} не удался — пауза")
                state.pause(f"Ошибка на шаге {step_info['name']}")
                self.storage.save(state)
                break

            state.record_step_result(step_info["name"], result)
            self.storage.save(state)

            if self.verbose:
                self._print_step_done(step_info["name"], result, state)

            state.advance_step()
            self.storage.save(state)

        if state.phase == TaskPhase.DONE and not self._interrupted:
            self._on_done(state)
        elif self._interrupted:
            state.pause("Прервано пользователем (Ctrl+C)")
            self.storage.save(state)
            if self.verbose:
                print(f"\n⏸  Задача приостановлена: {state.task_id}")
                print(f"   Для продолжения запустите с --resume {state.task_id}")
                print(state.format_phase_progress())

        return state

    def pause(self, state: TaskState, reason: str = "") -> None:
        """Явная пауза задачи из внешнего кода."""
        state.pause(reason or "Пауза по запросу")
        self.storage.save(state)
        self._interrupted = True

    def _execute_step(
        self, state: TaskState, step_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Вызывает LLM для выполнения одного шага.
        Возвращает dict с результатом или None при неустранимой ошибке.
        Если тема уже задана — choose_topic выполняется без LLM.
        """
        if step_name == "choose_topic" and state.topic:
            return {
                "topic": state.topic,
                "rationale": "Тема задана пользователем",
                "target_audience": "Специалисты и интересующиеся AI/ML",
                "article_type": "analysis",
            }

        prompt = _build_step_prompt(step_name, state.article_data)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=self.max_completion_tokens,
                    temperature=self.temperature,
                )
                raw = response.choices[0].message.content or ""
                if step_name == "write_body":
                    return self._parse_body_text(raw)
                return self._parse_json_response(raw)
            except Exception as e:
                if attempt < 2:
                    if self.verbose:
                        print(f"  ⚠️  API ошибка (попытка {attempt + 1}/3): {e}. Повтор через 2с...")
                    time.sleep(2)
                else:
                    if self.verbose:
                        print(f"  ❌ Шаг {step_name} провалился после 3 попыток: {e}")
                    return None

    def _parse_body_text(self, raw: str) -> Dict[str, Any]:
        """
        Парсит plain-text ответ для write_body.
        Разбивает текст по ## заголовкам в список секций.
        Если заголовков нет — кладёт весь текст в одну секцию.
        """
        text = raw.strip()
        if not text:
            return {"sections": [], "total_word_count": 0}

        sections: List[Dict[str, str]] = []
        current_name = ""
        current_lines: List[str] = []

        for line in text.split("\n"):
            if line.startswith("## "):
                if current_lines:
                    sections.append({
                        "name": current_name,
                        "text": "\n".join(current_lines).strip(),
                    })
                current_name = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append({
                "name": current_name or "Основной текст",
                "text": "\n".join(current_lines).strip(),
            })

        if not sections:
            sections = [{"name": "Основной текст", "text": text}]

        word_count = sum(len(s["text"].split()) for s in sections)
        return {"sections": sections, "total_word_count": word_count}

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """Извлекает JSON из ответа LLM. Поддерживает ```json ... ``` блоки."""
        text = raw.strip()
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_output": raw.strip()}

    def _setup_interrupt_handler(self, state: TaskState) -> None:
        """Перехватывает Ctrl+C для graceful pause."""
        def _handler(sig, frame):
            self._interrupted = True

        try:
            signal.signal(signal.SIGINT, _handler)
        except (OSError, ValueError):
            pass

    def _print_run_header(self, state: TaskState, is_resume: bool) -> None:
        if not self.verbose:
            return
        action = "▶  ПРОДОЛЖЕНИЕ ЗАДАЧИ" if is_resume else "🆕  НОВАЯ ЗАДАЧА"
        print("\n" + "═" * 65)
        print(f"  {action}")
        print("═" * 65)
        print(state.format_status())
        print(state.format_phase_progress())
        print(f"  Модель: {self.model}")
        print("═" * 65)

    def _print_step_header(
        self, state: TaskState, step_info: Dict[str, str]
    ) -> None:
        if not self.verbose:
            return
        phase_steps = PHASE_STEPS.get(state.phase, [])
        total = len(phase_steps)
        step_num = state.current_step + 1
        completed = state.completed_steps_count()
        print(f"\n{'─' * 65}")
        print(
            f"  [{state.phase.value.upper()}]  Шаг {step_num}/{total}  "
            f"(всего {completed}/{TOTAL_STEPS})"
        )
        print(f"  {step_info['name']} — {step_info['description']}")
        print(f"  Ожидаемое действие: {step_info['expected_action']}")
        print(f"{'─' * 65}")

    def _print_step_done(
        self,
        step_name: str,
        result: Dict[str, Any],
        state: TaskState,
    ) -> None:
        if not self.verbose:
            return
        preview = json.dumps(result, ensure_ascii=False)
        if len(preview) > 200:
            preview = preview[:197] + "..."
        print(f"  ✓ Шаг выполнен: {step_name}")
        print(f"    {preview}")

    def _on_done(self, state: TaskState) -> None:
        """Печатает итоговую статью при завершении."""
        if not self.verbose:
            return

        final_article = state.article_data.get(
            "final_article", _build_full_article(state.article_data)
        )
        title = state.article_data.get("title", state.topic or "—")
        overall_score = state.article_data.get("overall_score", "—")
        word_count = state.article_data.get("final_word_count", "—")

        print("\n" + "═" * 65)
        print("  ✅ ЗАДАЧА ЗАВЕРШЕНА  [done]")
        print("═" * 65)
        print(f"  ID задачи:  {state.task_id}")
        print(f"  Тема:       {state.topic or state.article_data.get('topic', '—')}")
        print(f"  Заголовок:  {title}")
        print(f"  Оценка:     {overall_score}/10")
        print(f"  Слов:       {word_count}")
        print(f"  Шагов:      {state.completed_steps_count()}/{TOTAL_STEPS}")
        print(f"\n{'─' * 65}")
        print("  ФИНАЛЬНАЯ СТАТЬЯ:")
        print(f"{'─' * 65}\n")
        for line in final_article.split("\n"):
            print(f"  {line}")
        print(f"\n{'═' * 65}\n")
