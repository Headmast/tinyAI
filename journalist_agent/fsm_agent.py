"""
JournalistFSMAgent — агент-журналист с контролируемым жизненным циклом задачи.

Каждая задача проходит явные состояния (FSM):
  article:       planning → research → drafting → editing → validation → published
  news_research: topic_selection → source_gathering → analysis → drafting → fact_check → published
  review:        subject_definition → criteria_setting → examination → drafting → validation → published
  note:          idea_capture → drafting → published
  blog_analysis: blog_selection → metrics_collection → content_analysis → insights_extraction
                 → drafting → validation → published

Гарантии:
  - нельзя перепрыгнуть шаг: переход проверяется до вызова LLM
  - нельзя выйти из паузы прямым переходом — только через resume()
  - published — терминальное состояние, дальше нельзя
  - при любой ошибке шага задача уходит на паузу (не теряется)

Персистентность: каждое состояние сохраняется в JournalistTaskStorage.
"""

from __future__ import annotations

import json
import signal
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from journalist_agent.workflow import (
    ContentType,
    JournalistTask,
    JournalistTaskStorage,
    STATE_PAUSED,
    STATE_PUBLISHED,
    StepConfig,
    TransitionError,
    WORKFLOWS,
)
from journalist_agent.step_prompts import SYSTEM_PROMPT, build_step_prompt
from journalist_agent.invariants import InvariantStore


# ─────────────────────────────────────────────────────────────
# JournalistFSMAgent
# ─────────────────────────────────────────────────────────────

class JournalistFSMAgent:
    """
    Агент-журналист с конечным автоматом и контролем переходов.

    Запуск новой задачи:
        task = agent.start(ContentType.ARTICLE, topic="...")
        task = agent.run_all(task)

    Пошаговое управление:
        task = agent.start(ContentType.REVIEW, topic="...")
        task = agent.advance(task)           # авто-переход к следующему шагу
        task = agent.advance(task, "editing")  # явный целевой шаг

    Пауза и продолжение:
        task = agent.pause(task, "перерыв")
        task = agent.resume(task)
        task = agent.run_all(task)

    Возобновление по ID:
        task = agent.load_and_resume("task_id_here")
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "zai-org/GLM-4.7",
        storage: Optional[JournalistTaskStorage] = None,
        invariant_store: Optional[InvariantStore] = None,
        verbose: bool = True,
        max_completion_tokens: int = 3000,
        temperature: float = 0.6,
    ) -> None:
        self.client = client
        self.model = model
        self.storage = storage or JournalistTaskStorage()
        self.store = invariant_store or InvariantStore()
        self.verbose = verbose
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self._interrupted = False

    # ─────────────────────────────────────────────────────────
    # Публичный API
    # ─────────────────────────────────────────────────────────

    def start(self, content_type: ContentType, topic: str = "") -> JournalistTask:
        """
        Создаёт новую задачу и возвращает её в начальном состоянии.
        Задача сохраняется в хранилище.
        """
        task = JournalistTask.new(content_type, topic=topic)
        self.storage.save(task)
        if self.verbose:
            self._print_header(task, is_new=True)
        return task

    def advance(
        self,
        task: JournalistTask,
        to_state: Optional[str] = None,
    ) -> JournalistTask:
        """
        Выполняет текущий шаг и переходит к следующему.

        Args:
            task:     текущая задача
            to_state: целевое состояние (None = авто-следующий шаг)

        Returns:
            Обновлённая задача после выполнения шага.

        Raises:
            TransitionError:  если запрошен недопустимый переход
            ValueError:       если задача на паузе или уже завершена
        """
        if task.is_paused:
            raise ValueError(
                f"Задача {task.task_id!r} на паузе. "
                "Используйте resume() для продолжения."
            )
        if task.is_done:
            raise ValueError(
                f"Задача {task.task_id!r} уже завершена (published). "
                "Нельзя продолжать выполнение."
            )

        workflow = task.workflow
        current = task.state

        # Определяем целевое состояние
        target = to_state or workflow.auto_next(current)
        if target is None:
            raise TransitionError(current, "(нет)", [], task.content_type)

        # Проверяем допустимость перехода
        if not workflow.can_transition(current, target):
            allowed = workflow.allowed_next(current)
            raise TransitionError(current, target, allowed, task.content_type)

        # Проверяем инварианты если topic передаётся впервые
        if task.topic:
            violation = self.store.pre_check(task.topic)
            if violation.violated:
                if self.verbose:
                    print(f"\n  ⛔ Инвариант нарушен [{violation.invariant.id}]: {violation.explanation}")
                task = self._do_pause(task, f"Нарушение инварианта {violation.invariant.id}")
                self.storage.save(task)
                return task

        # Выполняем шаг
        step = workflow.get_step(current)
        if step is None:
            raise ValueError(f"Шаг {current!r} не найден в рабочем процессе")

        if self.verbose:
            self._print_step_header(task, step, target)

        result = self._execute_step(task, step)

        if result is None:
            if self.verbose:
                print(f"\n  ❌ Шаг {step.label!r} не удался — задача на паузе")
            task = self._do_pause(task, f"Ошибка на шаге {step.state}")
            self.storage.save(task)
            return task

        # Применяем результат и переходим
        task.step_results.update(result)
        task.state = target
        self.storage.save(task)

        if self.verbose:
            self._print_step_done(step, result, task)

        return task

    def pause(self, task: JournalistTask, reason: str = "") -> JournalistTask:
        """Явная пауза из внешнего кода."""
        if task.is_done:
            raise ValueError("Нельзя поставить на паузу завершённую задачу.")
        if task.is_paused:
            return task  # уже на паузе
        task = self._do_pause(task, reason or "Пауза по запросу")
        self.storage.save(task)
        if self.verbose:
            print(f"\n⏸  Задача приостановлена: {task.task_id}")
            print(f"   Причина: {task.pause_reason}")
            print(f"   Для продолжения: agent.resume(task) или --resume {task.task_id}")
        return task

    def resume(self, task: JournalistTask) -> JournalistTask:
        """
        Возобновляет задачу после паузы.
        Восстанавливает состояние, в котором была задача до паузы.
        """
        if not task.is_paused:
            raise ValueError(f"Задача {task.task_id!r} не на паузе (state={task.state!r}).")

        restored = task.state_before_pause or task.workflow.initial_state
        task.state = restored
        task.state_before_pause = None
        task.pause_reason = ""
        self.storage.save(task)

        if self.verbose:
            print(f"\n▶  Задача возобновлена: {task.task_id}")
            print(f"   Продолжаем с шага: {restored!r}")
            step = task.workflow.get_step(restored)
            if step:
                print(f"   Шаг: {step.label} — {step.description}")
        return task

    def run_all(self, task: JournalistTask) -> JournalistTask:
        """
        Прогоняет задачу до конца (или до паузы/прерывания).

        Поддерживает Ctrl+C для graceful pause.
        """
        self._interrupted = False
        self._setup_interrupt_handler(task)

        if task.is_paused:
            task = self.resume(task)

        if self.verbose:
            self._print_run_header(task)

        while not task.is_done and not self._interrupted:
            try:
                task = self.advance(task)
            except TransitionError as e:
                if self.verbose:
                    print(f"\n  🚫 FSM ошибка: {e}")
                break
            except ValueError as e:
                if self.verbose:
                    print(f"\n  ⚠️  {e}")
                break

        if self._interrupted:
            task = self.pause(task, "Прервано пользователем (Ctrl+C)")

        if task.is_done and self.verbose:
            self._print_final(task)

        return task

    def load_and_resume(self, task_id: str) -> JournalistTask:
        """Загружает задачу из хранилища и возобновляет её выполнение."""
        task = self.storage.load(task_id)
        if task is None:
            raise ValueError(f"Задача {task_id!r} не найдена в хранилище")
        return self.run_all(task)

    def try_transition(
        self,
        task: JournalistTask,
        to_state: str,
    ) -> Tuple[bool, str]:
        """
        Проверяет, возможен ли переход (без выполнения шага).

        Returns:
            (allowed: bool, reason: str)
        """
        workflow = task.workflow
        if task.is_paused:
            return False, f"Задача на паузе. Текущее состояние до паузы: {task.state_before_pause!r}"
        if task.is_done:
            return False, "Задача завершена (published)"
        if workflow.can_transition(task.state, to_state):
            return True, f"Переход {task.state!r} → {to_state!r} разрешён"
        allowed = workflow.allowed_next(task.state)
        skip_info = self._detect_skip(task, to_state)
        reason = f"Переход {task.state!r} → {to_state!r} запрещён. Разрешено: {allowed}"
        if skip_info:
            reason += f". {skip_info}"
        return False, reason

    # ─────────────────────────────────────────────────────────
    # Внутренние методы
    # ─────────────────────────────────────────────────────────

    def _execute_step(
        self,
        task: JournalistTask,
        step: StepConfig,
    ) -> Optional[Dict[str, Any]]:
        """
        Вызывает LLM для выполнения одного шага.
        Возвращает dict с результатом или None при неустранимой ошибке.
        """
        if step.state == STATE_PUBLISHED:
            return self._assemble_final(task)

        prompt = build_step_prompt(task, step.state)
        messages = [
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

                if step.output_format == "text":
                    return self._parse_text_response(raw, step.state)
                return self._parse_json_response(raw)

            except Exception as e:
                if attempt < 2:
                    if self.verbose:
                        print(f"    ⚠️  API ошибка (попытка {attempt + 1}/3): {e}. Повтор через 2с...")
                    time.sleep(2)
                else:
                    if self.verbose:
                        print(f"    ❌ Шаг провалился после 3 попыток: {e}")
                    return None

    def _assemble_final(self, task: JournalistTask) -> Dict[str, Any]:
        """Собирает финальный текст из накопленных step_results."""
        sr = task.step_results
        parts = []

        # Заголовок
        title = sr.get("title", task.topic or "Материал")
        parts.append(f"# {title}")

        # Основной текст (отредактированный > черновик)
        main_text = sr.get("edited_text", sr.get("draft_text", sr.get("note_text", "")))
        if main_text:
            parts.append(main_text)

        final_text = "\n\n".join(p for p in parts if p)
        return {
            "final_text": final_text,
            "word_count": len(final_text.split()),
        }

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """Парсит JSON из ответа LLM с поддержкой ```json``` блоков."""
        text = raw.strip()
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end == -1:
                end = len(text)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end == -1:
                end = len(text)
            text = text[start:end].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_output": raw.strip()}

    def _parse_text_response(self, raw: str, state: str) -> Dict[str, Any]:
        """Оборачивает текстовый ответ в dict."""
        text = raw.strip()
        key_map = {
            "drafting":  "draft_text",
            "note_drafting": "note_text",
        }
        key = key_map.get(state, "draft_text")
        return {key: text, "word_count": len(text.split())}

    def _do_pause(self, task: JournalistTask, reason: str) -> JournalistTask:
        task.state_before_pause = task.state
        task.state = STATE_PAUSED
        task.pause_reason = reason
        return task

    def _detect_skip(self, task: JournalistTask, to_state: str) -> str:
        """Если to_state есть в workflow но не следующий — объясняем пропуск."""
        workflow = task.workflow
        from_idx = workflow.step_index(task.state)
        to_idx = workflow.step_index(to_state)
        if from_idx >= 0 and to_idx > from_idx + 1:
            skipped = [s.label for s in workflow.steps[from_idx + 1:to_idx]]
            return f"Пропускаются шаги: {', '.join(skipped)}"
        return ""

    def _setup_interrupt_handler(self, task: JournalistTask) -> None:
        def _handler(sig, frame):
            self._interrupted = True
        try:
            signal.signal(signal.SIGINT, _handler)
        except (OSError, ValueError):
            pass

    # ─────────────────────────────────────────────────────────
    # Форматированный вывод
    # ─────────────────────────────────────────────────────────

    def _print_header(self, task: JournalistTask, is_new: bool = True) -> None:
        if not self.verbose:
            return
        action = "🆕  НОВАЯ ЗАДАЧА" if is_new else "▶  ПРОДОЛЖЕНИЕ"
        print("\n" + "═" * 65)
        print(f"  {action}  [{task.content_type.label()}]")
        print("═" * 65)
        print(f"  ID:     {task.task_id}")
        print(f"  Тема:   {task.topic or '(будет определена на первом шаге)'}")
        print(f"  Модель: {self.model}")
        print("─" * 65)
        print(task.format_progress())
        print("═" * 65)

    def _print_run_header(self, task: JournalistTask) -> None:
        if not self.verbose:
            return
        print("\n" + "═" * 65)
        print(f"  ЗАПУСК ЗАДАЧИ  [{task.content_type.label()}]")
        print("═" * 65)
        print(f"  ID:     {task.task_id}")
        print(f"  Тема:   {task.topic or '(без темы)'}")
        print(task.format_progress())
        print("═" * 65)

    def _print_step_header(
        self, task: JournalistTask, step: StepConfig, next_state: str
    ) -> None:
        if not self.verbose:
            return
        workflow = task.workflow
        total = len(workflow.steps) - 1  # без published
        current_idx = workflow.step_index(step.state) + 1
        print(f"\n{'─' * 65}")
        print(f"  ШАГ {current_idx}/{total}  [{step.label.upper()}]")
        print(f"  {step.description}")
        print(f"  Ожидаемый результат: {step.expected_output}")
        print(f"  Следующее состояние: {next_state!r}")
        print(f"{'─' * 65}")

    def _print_step_done(
        self,
        step: StepConfig,
        result: Dict[str, Any],
        task: JournalistTask,
    ) -> None:
        if not self.verbose:
            return
        preview = json.dumps(result, ensure_ascii=False)
        if len(preview) > 220:
            preview = preview[:217] + "..."
        print(f"  ✓ Шаг выполнен: {step.label}")
        print(f"    {preview}")
        print(f"  → Новое состояние: {task.state!r}")

    def _print_final(self, task: JournalistTask) -> None:
        if not self.verbose:
            return
        sr = task.step_results
        final = sr.get("final_text", sr.get("edited_text", sr.get("draft_text", "(нет текста)")))
        title = sr.get("title", task.topic or "(без заголовка)")
        wc = sr.get("word_count", "—")

        print("\n" + "═" * 65)
        print(f"  ✅ ЗАДАЧА ЗАВЕРШЕНА  [{task.content_type.label()}]")
        print("═" * 65)
        print(f"  ID:     {task.task_id}")
        print(f"  Тема:   {task.topic}")
        print(f"  Заголовок: {title}")
        print(f"  Слов:   {wc}")
        print(f"\n{'─' * 65}")
        print("  ФИНАЛЬНЫЙ ТЕКСТ:")
        print(f"{'─' * 65}\n")
        for line in str(final).split("\n"):
            print(f"  {line}")
        print(f"\n{'═' * 65}\n")
