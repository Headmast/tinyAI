"""
FSM (Finite State Machine) для управления состоянием задачи написания статьи.

Состояния: planning → execution → validation → done

Каждое состояние фиксирует:
  - текущий этап (phase)
  - текущий шаг (current_step)
  - ожидаемое действие (expected_action)

Поддерживает паузу и продолжение без повторных объяснений.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class TaskPhase(str, Enum):
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"


PHASE_TRANSITIONS: Dict[str, Optional[str]] = {
    TaskPhase.PLANNING: TaskPhase.EXECUTION,
    TaskPhase.EXECUTION: TaskPhase.VALIDATION,
    TaskPhase.VALIDATION: TaskPhase.DONE,
    TaskPhase.DONE: None,
}

PHASE_STEPS: Dict[TaskPhase, List[Dict[str, str]]] = {
    TaskPhase.PLANNING: [
        {
            "name": "choose_topic",
            "description": "Выбор темы для статьи",
            "expected_action": "Агент предлагает тему на основе актуальных трендов в AI/технологиях",
        },
        {
            "name": "analyze_topic",
            "description": "Анализ выбранной темы",
            "expected_action": "Агент анализирует аудиторию, угол подачи и ключевые тезисы",
        },
        {
            "name": "create_outline",
            "description": "Создание структуры статьи",
            "expected_action": "Агент строит детальный план: заголовок, разделы, объём",
        },
    ],
    TaskPhase.EXECUTION: [
        {
            "name": "write_intro",
            "description": "Написание вступления",
            "expected_action": "Агент пишет лид-абзац: крючок + суть + контекст",
        },
        {
            "name": "write_body",
            "description": "Написание основного текста",
            "expected_action": "Агент разворачивает каждый раздел плана в связный текст",
        },
        {
            "name": "write_conclusion",
            "description": "Написание заключения",
            "expected_action": "Агент подводит итоги и формулирует главный вывод",
        },
    ],
    TaskPhase.VALIDATION: [
        {
            "name": "check_structure",
            "description": "Проверка структуры статьи",
            "expected_action": "Агент проверяет: все разделы на месте, логика соблюдена",
        },
        {
            "name": "score_quality",
            "description": "Оценка качества текста",
            "expected_action": "Агент выставляет оценки по критериям: ясность, точность, вовлечённость",
        },
        {
            "name": "final_edit",
            "description": "Финальное редактирование",
            "expected_action": "Агент вносит правки и финализирует текст для публикации",
        },
    ],
    TaskPhase.DONE: [],
}

TOTAL_STEPS = sum(len(v) for v in PHASE_STEPS.values())


@dataclass
class StepResult:
    step_name: str
    phase: str
    output: Dict[str, Any]
    completed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "phase": self.phase,
            "output": self.output,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepResult":
        return cls(
            step_name=data["step_name"],
            phase=data["phase"],
            output=data.get("output", {}),
            completed_at=data.get("completed_at", datetime.now().isoformat()),
        )


@dataclass
class TaskState:
    task_id: str
    phase: TaskPhase
    current_step: int
    expected_action: str
    topic: Optional[str]
    paused: bool
    pause_reason: str
    article_data: Dict[str, Any]
    step_results: List[StepResult]
    messages: List[Dict[str, Any]]
    created_at: str
    updated_at: str

    @classmethod
    def new(cls, topic: Optional[str] = None) -> "TaskState":
        """Создаёт новое состояние задачи — начало с planning."""
        first_step = PHASE_STEPS[TaskPhase.PLANNING][0]
        return cls(
            task_id=str(uuid.uuid4())[:8],
            phase=TaskPhase.PLANNING,
            current_step=0,
            expected_action=first_step["expected_action"],
            topic=topic,
            paused=False,
            pause_reason="",
            article_data={},
            step_results=[],
            messages=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

    def get_current_step_info(self) -> Optional[Dict[str, str]]:
        """Возвращает метаданные текущего шага или None если фаза завершена."""
        steps = PHASE_STEPS.get(self.phase, [])
        if self.current_step < len(steps):
            return steps[self.current_step]
        return None

    def advance_step(self) -> bool:
        """
        Переходит к следующему шагу внутри фазы.
        Если шаги в фазе закончились — переходит к следующей фазе.
        Возвращает True если переход выполнен, False если уже DONE.
        """
        if self.phase == TaskPhase.DONE:
            return False

        steps = PHASE_STEPS[self.phase]
        next_step = self.current_step + 1

        if next_step < len(steps):
            self.current_step = next_step
            self.expected_action = steps[next_step]["expected_action"]
        else:
            next_phase = PHASE_TRANSITIONS.get(self.phase)
            if next_phase is None or next_phase == TaskPhase.DONE:
                self.phase = TaskPhase.DONE
                self.current_step = 0
                self.expected_action = "Задача завершена"
            else:
                self.phase = TaskPhase(next_phase)
                self.current_step = 0
                next_steps = PHASE_STEPS[self.phase]
                self.expected_action = (
                    next_steps[0]["expected_action"] if next_steps else "Задача завершена"
                )

        self.updated_at = datetime.now().isoformat()
        return True

    def record_step_result(self, step_name: str, output: Dict[str, Any]) -> None:
        """Записывает результат выполнения шага и обновляет article_data."""
        result = StepResult(
            step_name=step_name,
            phase=self.phase.value,
            output=output,
        )
        self.step_results.append(result)
        self.article_data.update(output)
        self.updated_at = datetime.now().isoformat()

    def pause(self, reason: str = "") -> None:
        """Приостанавливает выполнение задачи."""
        self.paused = True
        self.pause_reason = reason or "по запросу"
        self.updated_at = datetime.now().isoformat()

    def resume(self) -> None:
        """Снимает паузу и продолжает выполнение."""
        self.paused = False
        self.pause_reason = ""
        self.updated_at = datetime.now().isoformat()

    def completed_steps_count(self) -> int:
        return len(self.step_results)

    def format_status(self) -> str:
        """Возвращает форматированный статус задачи для вывода в консоль."""
        step_info = self.get_current_step_info()
        phase_steps = PHASE_STEPS.get(self.phase, [])
        total_in_phase = len(phase_steps)
        completed = self.completed_steps_count()

        lines = [
            "╔" + "═" * 57 + "╗",
            f"  Задача:    {self.task_id}",
            f"  Тема:      {(self.topic or '(будет выбрана)') [:50]}",
            f"  Этап:      [{self.phase.value.upper()}]  "
            f"шаг {self.current_step + 1}/{total_in_phase or 1}",
        ]

        if step_info:
            lines.append(f"  Шаг:       {step_info['name']} — {step_info['description']}")
            lines.append(f"  Ожидание:  {step_info['expected_action']}")

        status_icon = "⏸ ПАУЗА" if self.paused else "▶ Выполняется"
        lines.append(f"  Статус:    {status_icon}")
        if self.paused and self.pause_reason:
            lines.append(f"  Причина:   {self.pause_reason}")

        lines.append(f"  Прогресс:  {completed}/{TOTAL_STEPS} шагов выполнено")
        lines.append("╚" + "═" * 57 + "╝")
        return "\n".join(lines)

    def format_phase_progress(self) -> str:
        """Показывает прогресс по всем фазам."""
        phases = [TaskPhase.PLANNING, TaskPhase.EXECUTION, TaskPhase.VALIDATION, TaskPhase.DONE]
        done_phases = []
        current_idx = phases.index(self.phase) if self.phase in phases else 0
        for i, ph in enumerate(phases[:-1]):
            if i < current_idx:
                done_phases.append(f"[✓ {ph.value}]")
            elif i == current_idx:
                done_phases.append(f"[▶ {ph.value}]")
            else:
                done_phases.append(f"[  {ph.value}]")
        return " → ".join(done_phases) + (" → [✓ done]" if self.phase == TaskPhase.DONE else " → [  done]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "phase": self.phase.value,
            "current_step": self.current_step,
            "expected_action": self.expected_action,
            "topic": self.topic,
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "article_data": self.article_data,
            "step_results": [r.to_dict() for r in self.step_results],
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskState":
        return cls(
            task_id=data["task_id"],
            phase=TaskPhase(data["phase"]),
            current_step=data["current_step"],
            expected_action=data.get("expected_action", ""),
            topic=data.get("topic"),
            paused=data.get("paused", False),
            pause_reason=data.get("pause_reason", ""),
            article_data=data.get("article_data", {}),
            step_results=[StepResult.from_dict(r) for r in data.get("step_results", [])],
            messages=data.get("messages", []),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )

    def __repr__(self) -> str:
        return (
            f"TaskState(id={self.task_id!r}, phase={self.phase.value!r}, "
            f"step={self.current_step}, paused={self.paused})"
        )


class TaskStateStorage:
    """
    Персистентное хранилище состояний задач.

    Структура:
        tasks/
            index.json        — индекс всех задач (метаданные)
            <task_id>.json    — полное состояние задачи

    Каждое сохранение автоматически обновляет index.json.
    """

    def __init__(self, base_dir: str = "tasks") -> None:
        self.base_dir = Path(base_dir)
        self.index_file = self.base_dir / "index.json"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_file.exists():
            self._write_index([])

    def _read_index(self) -> List[Dict[str, Any]]:
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_index(self, index: List[Dict[str, Any]]) -> None:
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _make_index_entry(self, state: TaskState) -> Dict[str, Any]:
        return {
            "task_id": state.task_id,
            "phase": state.phase.value,
            "current_step": state.current_step,
            "topic": state.topic or state.article_data.get("topic"),
            "paused": state.paused,
            "completed_steps": state.completed_steps_count(),
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }

    def save(self, state: TaskState) -> None:
        """Сохраняет состояние задачи на диск и обновляет индекс."""
        task_file = self.base_dir / f"{state.task_id}.json"
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

        index = self._read_index()
        existing_idx = next(
            (i for i, e in enumerate(index) if e["task_id"] == state.task_id),
            None,
        )
        entry = self._make_index_entry(state)
        if existing_idx is not None:
            index[existing_idx] = entry
        else:
            index.insert(0, entry)
        self._write_index(index)

    def load(self, task_id: str) -> Optional[TaskState]:
        """Загружает состояние задачи по ID. Возвращает None если не найдено."""
        task_file = self.base_dir / f"{task_id}.json"
        if not task_file.exists():
            return None
        try:
            with open(task_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TaskState.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def list_tasks(self, n: int = 20, paused_only: bool = False) -> List[Dict[str, Any]]:
        """Возвращает список задач из индекса."""
        index = self._read_index()
        if paused_only:
            index = [e for e in index if e.get("paused")]
        return index[:n]

    def find_latest_paused(self) -> Optional[str]:
        """Возвращает ID последней приостановленной незавершённой задачи."""
        for entry in self._read_index():
            if entry.get("paused") and entry.get("phase") != TaskPhase.DONE.value:
                return entry["task_id"]
        return None

    def find_in_progress(self) -> List[Dict[str, Any]]:
        """Возвращает все незавершённые задачи."""
        return [
            e for e in self._read_index()
            if e.get("phase") != TaskPhase.DONE.value
        ]
