"""
Интерактивный запуск JournalistFSMAgent с тремя задачами.

Темы:
  [1] Техническая  — Квантовые вычисления в 2025  (ARTICLE)
  [2] Финансовая   — DeFi против традиционных банков (NEWS_RESEARCH)
  [3] Искусство    — Нейросети и современное искусство (REVIEW)

Управление в интерактивном режиме:
  Enter / next     — авто-переход к следующему шагу
  <state>          — явный переход (например: editing, published) — можно провоцировать ошибку
  pause            — поставить на паузу
  resume           — возобновить
  status           — показать прогресс
  skip             — показать меню пропуска шагов
  all              — прогнать задачу до конца без остановок
  q / quit         — завершить текущую задачу

Запуск:
  python run_journalist_interactive.py --model zai-org/GLM-4.7
  python run_journalist_interactive.py --model zai-org/GLM-4.7 --task 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from journalist_agent.workflow import (
    ContentType,
    JournalistTask,
    JournalistTaskStorage,
    STATE_PAUSED,
    STATE_PUBLISHED,
    TransitionError,
    WORKFLOWS,
)
from journalist_agent.fsm_agent import JournalistFSMAgent
from journalist_agent.invariants import InvariantStore

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Предопределённые задачи
# ─────────────────────────────────────────────────────────────

TASKS = [
    {
        "label":        "🔬 Техническая",
        "content_type": ContentType.ARTICLE,
        "topic":        "Квантовые вычисления в 2025: от лабораторных экспериментов к практическим применениям",
    },
    {
        "label":        "💰 Финансовая",
        "content_type": ContentType.NEWS_RESEARCH,
        "topic":        "DeFi против традиционных банков: кто побеждает в 2025 и что ждёт инвесторов",
    },
    {
        "label":        "🎨 Искусство",
        "content_type": ContentType.REVIEW,
        "topic":        "Нейросети и современное искусство: угроза творчеству или новый инструмент художника",
    },
]

# ─────────────────────────────────────────────────────────────
# Утилиты вывода
# ─────────────────────────────────────────────────────────────

def hr(char: str = "─", width: int = 70) -> str:
    return char * width


def print_hr(char: str = "─", width: int = 70) -> None:
    print(char * width)


def print_state_panel(task: JournalistTask) -> None:
    """Показывает текущее состояние задачи и разрешённые переходы."""
    wf = task.workflow
    print()
    print_hr("═")
    print(f"  ЗАДАЧА: {task.content_type.label().upper()}")
    print(f"  Тема:   {task.topic}")
    print(f"  ID:     {task.task_id}")
    print_hr()
    print(task.format_progress())
    print_hr()

    if task.is_done:
        print("  ✅ ЗАДАЧА ЗАВЕРШЕНА")
        print_hr("═")
        return

    if task.is_paused:
        print(f"  ⏸  НА ПАУЗЕ  (состояние до паузы: {task.state_before_pause!r})")
        print(f"  Причина: {task.pause_reason or '—'}")
        print_hr()
        print("  Команды: resume | status | q")
        print_hr("═")
        return

    step = task.current_step
    if step:
        print(f"  ▶ ТЕКУЩИЙ ШАГ: [{step.state}] {step.label}")
        print(f"    {step.description}")
        print(f"    Ожидаемый вывод: {step.expected_output}")

    allowed = wf.allowed_next(task.state)
    all_states = [s.state for s in wf.steps]
    forbidden = [s for s in all_states if s not in allowed and s != task.state]

    print_hr()
    print(f"  ✅ Разрешённые переходы:  {', '.join(allowed) or '(нет)'}")
    print(f"  🚫 Запрещённые переходы: {', '.join(forbidden[:5]) or '(нет)'}")
    print_hr()
    print("  Команды: next | <state> | pause | skip | all | status | q")
    print_hr("═")


def print_welcome() -> None:
    print()
    print_hr("═")
    print("  JOURNALIST FSM AGENT — ИНТЕРАКТИВНОЕ ДЕМО")
    print_hr("═")
    print()
    for i, t in enumerate(TASKS, 1):
        ct = t["content_type"]
        wf = WORKFLOWS[ct]
        steps_str = " → ".join(s.label for s in wf.steps)
        print(f"  [{i}] {t['label']}")
        print(f"      Тема: {t['topic']}")
        print(f"      Тип:  {ct.label()}  ({len(wf.steps)} шагов)")
        print(f"      Путь: {steps_str}")
        print()
    print("  [A] Запустить все три задачи последовательно")
    print()
    print_hr("═")


def prompt_input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return "q"


# ─────────────────────────────────────────────────────────────
# Интерактивный цикл для одной задачи
# ─────────────────────────────────────────────────────────────

def run_interactive(
    agent: JournalistFSMAgent,
    task: JournalistTask,
) -> JournalistTask:
    """Интерактивный REPL для управления одной задачей."""

    print_state_panel(task)

    while True:
        if task.is_done:
            _show_result(task)
            break

        cmd = prompt_input("\n  > ").lower().strip()

        if cmd in ("q", "quit", "exit"):
            print("\n  Выход из текущей задачи.")
            break

        elif cmd in ("", "next", "n"):
            task = _do_advance(agent, task, to_state=None)
            print_state_panel(task)

        elif cmd == "all":
            print("\n  Запускаем до конца...")
            task = agent.run_all(task)
            print_state_panel(task)
            if task.is_done:
                _show_result(task)
                break

        elif cmd == "pause":
            reason = prompt_input("  Причина паузы (Enter — без причины): ")
            try:
                task = agent.pause(task, reason or "Пауза пользователя")
            except ValueError as e:
                print(f"\n  ⚠️  {e}")
            print_state_panel(task)

        elif cmd == "resume":
            try:
                task = agent.resume(task)
            except ValueError as e:
                print(f"\n  ⚠️  {e}")
            print_state_panel(task)

        elif cmd == "status":
            print_state_panel(task)

        elif cmd == "skip":
            task = _interactive_skip(agent, task)
            print_state_panel(task)

        elif cmd == "help":
            _print_help(task)

        else:
            # Пользователь ввёл название состояния — пробуем перейти
            task = _do_advance(agent, task, to_state=cmd)
            print_state_panel(task)

    return task


def _do_advance(
    agent: JournalistFSMAgent,
    task: JournalistTask,
    to_state: Optional[str],
) -> JournalistTask:
    """Вызывает advance() с обработкой всех ошибок."""

    target_label = to_state or "(авто)"
    print(f"\n  → Пытаемся: {task.state!r} → {target_label!r}...")

    # Сначала проверяем без выполнения
    if to_state:
        ok, reason = agent.try_transition(task, to_state)
        if not ok:
            print(f"\n  🚫 ПЕРЕХОД ЗАБЛОКИРОВАН FSM")
            print(f"     {reason}")
            wf = task.workflow
            allowed = wf.allowed_next(task.state)
            if allowed:
                print(f"\n  ℹ️  Разрешённые переходы из «{task.state}»: {', '.join(allowed)}")
            # Объясняем что было бы пропущено
            if to_state in [s.state for s in wf.steps]:
                from_idx = wf.step_index(task.state)
                to_idx   = wf.step_index(to_state)
                if to_idx > from_idx + 1:
                    skipped = [wf.steps[i].label for i in range(from_idx + 1, to_idx)]
                    print(f"  ℹ️  Пропускаемые шаги: {', '.join(skipped)}")
            return task

    try:
        t0 = time.time()
        task = agent.advance(task, to_state=to_state)
        elapsed = time.time() - t0
        print(f"  ✅ Шаг выполнен за {elapsed:.1f}с → новое состояние: {task.state!r}")
    except TransitionError as e:
        print(f"\n  🚫 TransitionError: {e}")
    except ValueError as e:
        print(f"\n  ⚠️  ValueError: {e}")
    except Exception as e:
        print(f"\n  ❌ Ошибка: {type(e).__name__}: {e}")

    return task


def _interactive_skip(
    agent: JournalistFSMAgent,
    task: JournalistTask,
) -> JournalistTask:
    """Показывает меню пропуска шагов — удобно для провокации ошибок."""

    wf = task.workflow
    print()
    print_hr()
    print("  ПОПЫТКА ПРОПУСТИТЬ ШАГ — выберите целевое состояние:")
    print_hr()
    for i, step in enumerate(wf.steps):
        ok = wf.can_transition(task.state, step.state)
        if step.state == task.state:
            marker = "▶ (текущее)"
        elif ok:
            marker = "✅ разрешено"
        else:
            marker = "🚫 ЗАПРЕЩЕНО"
        print(f"  [{i}] {step.state:<30s} {step.label:<25s} {marker}")
    print_hr()

    choice = prompt_input("  Номер или название состояния (Enter — отмена): ")
    if not choice:
        return task

    # Попытка определить состояние по номеру или имени
    target = None
    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(wf.steps):
            target = wf.steps[idx].state
    else:
        target = choice.lower()

    if not target:
        print("  Некорректный выбор.")
        return task

    print(f"\n  Пробуем перейти: {task.state!r} → {target!r}")
    return _do_advance(agent, task, to_state=target)


def _show_result(task: JournalistTask) -> None:
    sr = task.step_results
    final = sr.get("final_text", sr.get("edited_text", sr.get("draft_text", "(нет текста)")))
    title = sr.get("title", task.topic)

    print()
    print_hr("═")
    print(f"  ✅ ФИНАЛЬНЫЙ РЕЗУЛЬТАТ — {task.content_type.label()}")
    print_hr("═")
    print(f"  Заголовок: {title}")
    print(f"  Слов: {sr.get('word_count', '—')}")
    print_hr()
    lines = str(final).split("\n")
    # Показываем первые 40 строк
    for line in lines[:40]:
        print(f"  {line}")
    if len(lines) > 40:
        print(f"  ... [ещё {len(lines) - 40} строк]")
    print_hr("═")


def _print_help(task: JournalistTask) -> None:
    wf = task.workflow
    print()
    print_hr()
    print("  КОМАНДЫ:")
    print("  Enter / next     — выполнить текущий шаг и перейти к следующему")
    print("  <state>          — явный переход (провоцирует ошибку при недопустимом)")
    print("  pause            — поставить задачу на паузу")
    print("  resume           — возобновить задачу с паузы")
    print("  skip             — меню выбора состояния для прыжка")
    print("  all              — запустить до конца без остановок")
    print("  status           — показать прогресс")
    print("  q                — выйти из задачи")
    print_hr()
    print("  СОСТОЯНИЯ этой задачи:")
    for step in wf.steps:
        print(f"    {step.state:<30s} — {step.label}")
    print_hr()


# ─────────────────────────────────────────────────────────────
# Главный цикл
# ─────────────────────────────────────────────────────────────

def run_task_menu(agent: JournalistFSMAgent, task_idx: int) -> None:
    """Запускает одну задачу из TASKS по индексу (0-based)."""
    t = TASKS[task_idx]
    print(f"\n  Запускаем: {t['label']}")
    print(f"  Тема: {t['topic']}")
    print(f"  Тип:  {t['content_type'].label()}")

    task = agent.start(t["content_type"], topic=t["topic"])
    run_interactive(agent, task)


def main() -> None:
    parser = argparse.ArgumentParser(description="Интерактивный JournalistFSMAgent")
    parser.add_argument("--model", default="zai-org/GLM-4.7")
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3],
        help="Номер задачи (1=техника, 2=финансы, 3=искусство)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2500,
        help="Максимум токенов на шаг",
    )
    parser.add_argument(
        "--storage-dir", default="journalist_tasks",
        help="Директория для хранения задач",
    )
    parser.add_argument(
        "--resume", metavar="TASK_ID",
        help="Возобновить задачу по ID",
    )
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.getenv("CLOUD_API_KEY"),
        base_url=os.getenv("CLOUD_BASE_URL") or "https://foundation-models.api.cloud.ru/v1",
    )
    storage = JournalistTaskStorage(base_dir=args.storage_dir)
    agent = JournalistFSMAgent(
        client=client,
        model=args.model,
        storage=storage,
        verbose=False,   # verbose=False: вывод контролируется нашим REPL
        max_completion_tokens=args.max_tokens,
        temperature=0.6,
    )

    print(f"\n  Модель: {args.model}  |  max_tokens: {args.max_tokens}")

    # Возобновление существующей задачи
    if args.resume:
        task = storage.load(args.resume)
        if task is None:
            print(f"  ❌ Задача {args.resume!r} не найдена")
            sys.exit(1)
        print(f"\n  Возобновляем задачу {args.resume!r}")
        run_interactive(agent, task)
        return

    print_welcome()

    if args.task:
        run_task_menu(agent, args.task - 1)
        return

    # Главное меню
    while True:
        choice = prompt_input("\n  Выберите задачу [1/2/3/A] или q для выхода: ").upper()

        if choice in ("Q", ""):
            print("  До свидания.")
            break
        elif choice == "1":
            run_task_menu(agent, 0)
        elif choice == "2":
            run_task_menu(agent, 1)
        elif choice == "3":
            run_task_menu(agent, 2)
        elif choice == "A":
            for i in range(3):
                run_task_menu(agent, i)
        else:
            print("  Неверный выбор.")


if __name__ == "__main__":
    main()
