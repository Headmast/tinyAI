"""
Конфигурация рабочих процессов для агента-журналиста.

Поддерживаемые типы контента:
  article        — полноценная статья (план → исследование → черновик → правка → валидация)
  news_research  — новостное исследование (тема → источники → анализ → черновик → фактчек)
  review         — ревью (объект → критерии → изучение → черновик → валидация)
  note           — быстрая заметка (идея → черновик)
  blog_analysis  — анализ блога (выбор → метрики → контент → выводы → черновик → валидация)

Специальные состояния (общие для всех типов):
  paused    — задача приостановлена; хранит state_before_pause для возобновления
  published — терминальное состояние; переходов нет

Гарантии FSM:
  - нельзя перепрыгнуть шаг (каждый переход явно разрешён или запрещён)
  - нельзя перейти в published без валидации (где она предусмотрена)
  - paused → любое состояние запрещено (нужен resume)
  - published → любое состояние запрещено
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# Типы контента
# ─────────────────────────────────────────────────────────────

class ContentType(Enum):
    ARTICLE       = "article"
    NEWS_RESEARCH = "news_research"
    REVIEW        = "review"
    NOTE          = "note"
    BLOG_ANALYSIS = "blog_analysis"

    @classmethod
    def from_str(cls, s: str) -> "ContentType":
        try:
            return cls(s.lower())
        except ValueError:
            valid = [e.value for e in cls]
            raise ValueError(f"Неизвестный тип контента: {s!r}. Допустимые: {valid}")

    def label(self) -> str:
        return {
            "article":       "Статья",
            "news_research": "Новостное исследование",
            "review":        "Ревью",
            "note":          "Заметка",
            "blog_analysis": "Анализ блога",
        }[self.value]


# ─────────────────────────────────────────────────────────────
# Специальные состояния
# ─────────────────────────────────────────────────────────────

STATE_PAUSED    = "paused"
STATE_PUBLISHED = "published"


# ─────────────────────────────────────────────────────────────
# Конфигурация шагов
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StepConfig:
    """Описание одного шага в рабочем процессе."""
    state: str               # идентификатор состояния
    label: str               # читаемое название
    description: str         # что делается на этом шаге
    expected_output: str     # что должен вернуть LLM (для отображения)
    output_format: str = "json"  # "json" | "text"


# ─────────────────────────────────────────────────────────────
# WorkflowConfig — конфигурация одного типа контента
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorkflowConfig:
    """
    Конфигурация рабочего процесса для одного типа контента.

    steps:       упорядоченный список шагов
    transitions: допустимые переходы state → [next_states]
    """
    content_type: ContentType
    steps: tuple          # Tuple[StepConfig, ...]
    transitions: Dict[str, List[str]]

    @property
    def initial_state(self) -> str:
        return self.steps[0].state

    def get_step(self, state: str) -> Optional[StepConfig]:
        return next((s for s in self.steps if s.state == state), None)

    def step_index(self, state: str) -> int:
        """Индекс шага в списке, -1 если не найден."""
        for i, s in enumerate(self.steps):
            if s.state == state:
                return i
        return -1

    def allowed_next(self, current_state: str) -> List[str]:
        """Список допустимых следующих состояний."""
        if current_state in (STATE_PAUSED, STATE_PUBLISHED):
            return []
        return list(self.transitions.get(current_state, []))

    def can_transition(self, from_state: str, to_state: str) -> bool:
        """True если переход from_state → to_state допустим."""
        if to_state == STATE_PAUSED:
            return from_state not in (STATE_PAUSED, STATE_PUBLISHED)
        if from_state in (STATE_PAUSED, STATE_PUBLISHED):
            return False
        return to_state in self.allowed_next(from_state)

    def auto_next(self, current_state: str) -> Optional[str]:
        """Следующее состояние по умолчанию (первый в списке разрешённых)."""
        nexts = self.allowed_next(current_state)
        return nexts[0] if nexts else None


# ─────────────────────────────────────────────────────────────
# Определения рабочих процессов
# ─────────────────────────────────────────────────────────────

WORKFLOWS: Dict[ContentType, WorkflowConfig] = {

    ContentType.ARTICLE: WorkflowConfig(
        content_type=ContentType.ARTICLE,
        steps=(
            StepConfig(
                state="planning",
                label="Планирование",
                description="Составление структуры и плана статьи",
                expected_output="title, sections[], target_audience, tone, word_count_target",
            ),
            StepConfig(
                state="research",
                label="Исследование",
                description="Сбор ключевых фактов, данных и углов подачи",
                expected_output="key_facts[], data_points[], angles[], sources[]",
            ),
            StepConfig(
                state="drafting",
                label="Написание черновика",
                description="Написание полного текста статьи по плану и исследованию",
                expected_output="draft_text (полный текст статьи)",
                output_format="text",
            ),
            StepConfig(
                state="editing",
                label="Редактирование",
                description="Стилистическая и структурная правка черновика",
                expected_output="edited_text, changes[]",
            ),
            StepConfig(
                state="validation",
                label="Валидация",
                description="Проверка фактов, качества и соответствия теме",
                expected_output="quality_score (1-10), issues[], approved (bool)",
            ),
            StepConfig(
                state=STATE_PUBLISHED,
                label="Опубликовано",
                description="Финальная статья готова к публикации",
                expected_output="(финальный текст)",
                output_format="text",
            ),
        ),
        transitions={
            "planning":   ["research"],
            "research":   ["drafting"],
            "drafting":   ["editing"],
            "editing":    ["validation"],
            "validation": [STATE_PUBLISHED],
        },
    ),

    ContentType.NEWS_RESEARCH: WorkflowConfig(
        content_type=ContentType.NEWS_RESEARCH,
        steps=(
            StepConfig(
                state="topic_selection",
                label="Выбор темы",
                description="Определение темы, актуальности и целевой аудитории",
                expected_output="topic, relevance, audience, news_angle",
            ),
            StepConfig(
                state="source_gathering",
                label="Сбор источников",
                description="Поиск и оценка источников по теме",
                expected_output="sources[], primary_sources[], credibility_notes",
            ),
            StepConfig(
                state="analysis",
                label="Анализ данных",
                description="Анализ собранных данных, выявление паттернов и выводов",
                expected_output="findings[], key_insights[], contradictions[]",
            ),
            StepConfig(
                state="drafting",
                label="Написание черновика",
                description="Оформление исследования в читаемый текст",
                expected_output="draft_text",
                output_format="text",
            ),
            StepConfig(
                state="fact_check",
                label="Проверка фактов",
                description="Верификация каждого утверждения в черновике",
                expected_output="verified_claims[], disputed_claims[], fact_check_score",
            ),
            StepConfig(
                state=STATE_PUBLISHED,
                label="Опубликовано",
                description="Исследование завершено и опубликовано",
                expected_output="(финальный текст)",
                output_format="text",
            ),
        ),
        transitions={
            "topic_selection":  ["source_gathering"],
            "source_gathering": ["analysis"],
            "analysis":         ["drafting"],
            "drafting":         ["fact_check"],
            "fact_check":       [STATE_PUBLISHED],
        },
    ),

    ContentType.REVIEW: WorkflowConfig(
        content_type=ContentType.REVIEW,
        steps=(
            StepConfig(
                state="subject_definition",
                label="Определение объекта",
                description="Точное определение того, что будет рецензироваться",
                expected_output="subject_name, subject_type, context, review_scope",
            ),
            StepConfig(
                state="criteria_setting",
                label="Установка критериев",
                description="Формирование системы оценки и критериев",
                expected_output="criteria[], weights{}, scoring_scale",
            ),
            StepConfig(
                state="examination",
                label="Изучение объекта",
                description="Детальный анализ по установленным критериям",
                expected_output="scores{criterion: score}, notes{}, strengths[], weaknesses[]",
            ),
            StepConfig(
                state="drafting",
                label="Написание черновика",
                description="Написание ревью на основе анализа",
                expected_output="draft_text",
                output_format="text",
            ),
            StepConfig(
                state="validation",
                label="Валидация",
                description="Проверка объективности, полноты и корректности оценки",
                expected_output="overall_score, verdict, approved (bool)",
            ),
            StepConfig(
                state=STATE_PUBLISHED,
                label="Опубликовано",
                description="Ревью опубликовано",
                expected_output="(финальный текст)",
                output_format="text",
            ),
        ),
        transitions={
            "subject_definition": ["criteria_setting"],
            "criteria_setting":   ["examination"],
            "examination":        ["drafting"],
            "drafting":           ["validation"],
            "validation":         [STATE_PUBLISHED],
        },
    ),

    ContentType.NOTE: WorkflowConfig(
        content_type=ContentType.NOTE,
        steps=(
            StepConfig(
                state="idea_capture",
                label="Захват идеи",
                description="Фиксация основной мысли, контекста и тезисов",
                expected_output="core_idea, context, key_points[], tone",
            ),
            StepConfig(
                state="drafting",
                label="Написание заметки",
                description="Написание короткого связного текста по идее",
                expected_output="note_text",
                output_format="text",
            ),
            StepConfig(
                state=STATE_PUBLISHED,
                label="Опубликовано",
                description="Заметка опубликована",
                expected_output="(финальный текст)",
                output_format="text",
            ),
        ),
        transitions={
            "idea_capture": ["drafting"],
            "drafting":     [STATE_PUBLISHED],
        },
    ),

    ContentType.BLOG_ANALYSIS: WorkflowConfig(
        content_type=ContentType.BLOG_ANALYSIS,
        steps=(
            StepConfig(
                state="blog_selection",
                label="Выбор блога",
                description="Определение блога для анализа и целей анализа",
                expected_output="blog_name, url, focus_areas[], analysis_goals",
            ),
            StepConfig(
                state="metrics_collection",
                label="Сбор метрик",
                description="Анализ количественных показателей блога",
                expected_output="post_frequency, avg_length, topics[], engagement_signals[]",
            ),
            StepConfig(
                state="content_analysis",
                label="Анализ контента",
                description="Качественный анализ стиля, структуры и тематики",
                expected_output="writing_style, content_patterns[], niche_positioning, audience_fit",
            ),
            StepConfig(
                state="insights_extraction",
                label="Извлечение выводов",
                description="Формулировка инсайтов и применимых уроков",
                expected_output="insights[], lessons[], what_works[], what_doesnt[]",
            ),
            StepConfig(
                state="drafting",
                label="Написание отчёта",
                description="Оформление анализа в структурированный текст",
                expected_output="draft_text",
                output_format="text",
            ),
            StepConfig(
                state="validation",
                label="Валидация",
                description="Проверка обоснованности выводов и полноты анализа",
                expected_output="completeness_score, bias_check, approved (bool)",
            ),
            StepConfig(
                state=STATE_PUBLISHED,
                label="Опубликовано",
                description="Анализ опубликован",
                expected_output="(финальный текст)",
                output_format="text",
            ),
        ),
        transitions={
            "blog_selection":      ["metrics_collection"],
            "metrics_collection":  ["content_analysis"],
            "content_analysis":    ["insights_extraction"],
            "insights_extraction": ["drafting"],
            "drafting":            ["validation"],
            "validation":          [STATE_PUBLISHED],
        },
    ),
}


# ─────────────────────────────────────────────────────────────
# Исключение при недопустимом переходе
# ─────────────────────────────────────────────────────────────

class TransitionError(Exception):
    """Попытка недопустимого перехода между состояниями."""

    def __init__(
        self,
        from_state: str,
        to_state: str,
        allowed: List[str],
        content_type: Optional[ContentType] = None,
    ) -> None:
        self.from_state = from_state
        self.to_state = to_state
        self.allowed = allowed
        self.content_type = content_type

        ctype_label = f" [{content_type.label()}]" if content_type else ""
        if not allowed:
            why = "(нет допустимых переходов — состояние терминальное или задача на паузе)"
        else:
            why = "Разрешено: " + ", ".join(allowed)

        super().__init__(
            f"Недопустимый переход{ctype_label}: "
            f"«{from_state}» → «{to_state}». {why}"
        )


# ─────────────────────────────────────────────────────────────
# JournalistTask — состояние задачи с персистентностью
# ─────────────────────────────────────────────────────────────

@dataclass
class JournalistTask:
    """
    Полное состояние одной журналистской задачи.

    Персистируется в JSON после каждого шага.
    task_id используется для возобновления после паузы.
    """

    task_id: str
    content_type: ContentType
    topic: str
    state: str
    state_before_pause: Optional[str]
    step_results: Dict[str, Any]
    created_at: str
    updated_at: str
    pause_reason: str = ""

    @classmethod
    def new(cls, content_type: ContentType, topic: str = "") -> "JournalistTask":
        workflow = WORKFLOWS[content_type]
        now = datetime.now().isoformat()
        return cls(
            task_id=uuid.uuid4().hex[:10],
            content_type=content_type,
            topic=topic,
            state=workflow.initial_state,
            state_before_pause=None,
            step_results={},
            created_at=now,
            updated_at=now,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "content_type": self.content_type.value,
            "topic": self.topic,
            "state": self.state,
            "state_before_pause": self.state_before_pause,
            "step_results": self.step_results,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "pause_reason": self.pause_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JournalistTask":
        return cls(
            task_id=data["task_id"],
            content_type=ContentType(data["content_type"]),
            topic=data.get("topic", ""),
            state=data["state"],
            state_before_pause=data.get("state_before_pause"),
            step_results=data.get("step_results", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            pause_reason=data.get("pause_reason", ""),
        )

    @property
    def is_paused(self) -> bool:
        return self.state == STATE_PAUSED

    @property
    def is_done(self) -> bool:
        return self.state == STATE_PUBLISHED

    @property
    def workflow(self) -> WorkflowConfig:
        return WORKFLOWS[self.content_type]

    @property
    def current_step(self) -> Optional[StepConfig]:
        return self.workflow.get_step(self.state)

    @property
    def active_state(self) -> str:
        """Активное рабочее состояние (state_before_pause при паузе, иначе state)."""
        if self.is_paused and self.state_before_pause:
            return self.state_before_pause
        return self.state

    def format_status(self) -> str:
        """Однострочный статус задачи."""
        step = self.current_step
        step_label = step.label if step else self.state
        status = "⏸ ПАУЗА" if self.is_paused else ("✅ ГОТОВО" if self.is_done else "▶ В работе")
        return (
            f"[{self.content_type.label()}] {self.topic or '(без темы)'} | "
            f"Состояние: {step_label} | {status}"
        )

    def format_progress(self) -> str:
        """Строка прогресса по шагам."""
        workflow = self.workflow
        lines = []
        active = self.active_state
        for step in workflow.steps:
            idx = workflow.step_index(step.state)
            active_idx = workflow.step_index(active)
            if step.state == STATE_PUBLISHED:
                marker = "✅" if self.is_done else "○"
            elif idx < active_idx:
                marker = "✓"
            elif step.state == active:
                marker = "▶" if not self.is_paused else "⏸"
            else:
                marker = "○"
            lines.append(f"  {marker} {step.label}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# JournalistTaskStorage — файловое хранилище задач
# ─────────────────────────────────────────────────────────────

class JournalistTaskStorage:
    """Хранит задачи в директории base_dir/<task_id>.json."""

    def __init__(self, base_dir: str = "journalist_tasks") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        return self.base_dir / f"{task_id}.json"

    def save(self, task: JournalistTask) -> None:
        task.updated_at = datetime.now().isoformat()
        data = task.to_dict()
        tmp = self._path(task.task_id).with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path(task.task_id))

    def load(self, task_id: str) -> Optional[JournalistTask]:
        p = self._path(task_id)
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return JournalistTask.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def list_tasks(self, n: int = 20) -> List[Dict[str, Any]]:
        items = []
        for p in sorted(self.base_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                items.append({
                    "task_id": d.get("task_id", p.stem),
                    "content_type": d.get("content_type", ""),
                    "topic": d.get("topic", ""),
                    "state": d.get("state", ""),
                    "updated_at": d.get("updated_at", ""),
                })
            except Exception:
                continue
            if len(items) >= n:
                break
        return items
