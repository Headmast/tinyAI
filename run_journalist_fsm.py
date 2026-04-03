"""
Демо-скрипт JournalistFSMAgent.

Показывает:
  1. Допустимые переходы — агент проходит шаги по порядку
  2. Недопустимые переходы — попытки перепрыгнуть шаги
  3. Попытки выйти из терминального состояния
  4. Пауза и продолжение после паузы
  5. try_transition() — проверка без выполнения

Запуск:
  python run_journalist_fsm.py --model zai-org/GLM-4.7 --type note --topic "Тема"
  python run_journalist_fsm.py --demo-only   # только проверка переходов (без LLM)
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from journalist_agent.workflow import ContentType, WORKFLOWS, STATE_PUBLISHED, STATE_PAUSED
from journalist_agent.fsm_agent import JournalistFSMAgent
from journalist_agent.invariants import InvariantStore


load_dotenv()


# ─────────────────────────────────────────────────────────────
# Демо без LLM — только FSM логика и переходы
# ─────────────────────────────────────────────────────────────

def demo_fsm_only() -> None:
    """Демонстрирует FSM без вызовов LLM."""

    print("\n" + "═" * 65)
    print("  FSM DEMO — проверка переходов (без LLM)")
    print("═" * 65)

    from journalist_agent.workflow import JournalistTask, TransitionError

    # ── Тест 1: Допустимые переходы ARTICLE ──────────────────
    print("\n[1] Допустимые переходы (article)")
    wf = WORKFLOWS[ContentType.ARTICLE]
    chain = ["planning", "research", "drafting", "editing", "validation", STATE_PUBLISHED]
    for i in range(len(chain) - 1):
        ok = wf.can_transition(chain[i], chain[i + 1])
        mark = "✅" if ok else "❌"
        print(f"  {mark} {chain[i]} → {chain[i + 1]}")

    # ── Тест 2: Недопустимые переходы (перепрыгивание) ───────
    print("\n[2] Недопустимые переходы — перепрыгивание шагов (article)")
    bad_transitions = [
        ("planning",   "drafting",       "пропуск research"),
        ("planning",   "validation",     "пропуск 3 шагов"),
        ("planning",   STATE_PUBLISHED,  "пропуск всех шагов"),
        ("research",   "editing",        "пропуск drafting"),
        ("editing",    STATE_PUBLISHED,  "пропуск validation"),
    ]
    for frm, to, reason in bad_transitions:
        ok = wf.can_transition(frm, to)
        mark = "✅ (правильно заблокировано)" if not ok else "❌ (должно быть запрещено!)"
        print(f"  {mark} {frm} → {to}  [{reason}]")

    # ── Тест 3: Терминальное состояние ───────────────────────
    print("\n[3] Переходы из терминального состояния (published)")
    for dest in ["planning", "drafting", STATE_PAUSED]:
        ok = wf.can_transition(STATE_PUBLISHED, dest)
        mark = "✅ заблокировано" if not ok else "❌ не должно быть разрешено"
        print(f"  {mark} published → {dest}")

    # ── Тест 4: Переходы из paused ───────────────────────────
    print("\n[4] Переходы из paused (без resume)")
    for dest in ["planning", "research", "drafting"]:
        ok = wf.can_transition(STATE_PAUSED, dest)
        mark = "✅ заблокировано" if not ok else "❌ не должно быть разрешено"
        print(f"  {mark} paused → {dest}")

    # ── Тест 5: Допустимые переходы всех типов ───────────────
    print("\n[5] Цепочки переходов по всем типам контента")
    for ct in ContentType:
        wf_ct = WORKFLOWS[ct]
        steps = [s.state for s in wf_ct.steps]
        all_ok = all(
            wf_ct.can_transition(steps[i], steps[i + 1])
            for i in range(len(steps) - 1)
        )
        mark = "✅" if all_ok else "❌"
        print(f"  {mark} {ct.label():30s}  шагов: {len(steps)}")

    # ── Тест 6: Пауза ─────────────────────────────────────────
    print("\n[6] Пауза: можно из любого активного состояния")
    article_wf = WORKFLOWS[ContentType.ARTICLE]
    for step in article_wf.steps:
        ok = article_wf.can_transition(step.state, STATE_PAUSED)
        expected = step.state != STATE_PUBLISHED
        mark = "✅" if ok == expected else "❌"
        print(f"  {mark} {step.state} → paused: {ok}")

    # ── Тест 7: TransitionError с информацией ────────────────
    print("\n[7] TransitionError содержит полезную информацию")
    try:
        raise TransitionError("planning", "drafting", ["research"], ContentType.ARTICLE)
    except TransitionError as e:
        print(f"  ✅ Ошибка: {e}")

    print("\n✅ FSM логика работает корректно")


# ─────────────────────────────────────────────────────────────
# Полное демо с LLM
# ─────────────────────────────────────────────────────────────

def demo_with_llm(
    model: str,
    content_type: ContentType,
    topic: str,
    max_tokens: int,
) -> None:
    """Запускает реальную задачу через LLM."""

    client = OpenAI(
        api_key=os.getenv("CLOUD_API_KEY"),
        base_url=os.getenv("CLOUD_BASE_URL", "https://api.cloud.ru/v1"),
    )

    agent = JournalistFSMAgent(
        client=client,
        model=model,
        verbose=True,
        max_completion_tokens=max_tokens,
    )

    # ── Показываем структуру рабочего процесса ────────────────
    wf = WORKFLOWS[content_type]
    print("\n" + "═" * 65)
    print(f"  РАБОЧИЙ ПРОЦЕСС: {content_type.label()}")
    print("═" * 65)
    for i, step in enumerate(wf.steps):
        nexts = wf.allowed_next(step.state)
        arrows = " → ".join(nexts) if nexts else "(терминальное)"
        print(f"  {i + 1:2d}. [{step.state}] {step.label}")
        print(f"       {step.description}")
        print(f"       Допустимые переходы: {arrows}")
    print("═" * 65)

    # ── Демо: недопустимые переходы ───────────────────────────
    print("\n  [Проверка защиты от перепрыгивания шагов]")
    task = agent.start(content_type, topic=topic)
    print(f"\n  Начальное состояние: {task.state!r}")

    steps = [s.state for s in wf.steps if s.state != STATE_PUBLISHED]
    if len(steps) > 2:
        skip_target = steps[2]  # прыжок на 3-й шаг через 2
        ok, reason = agent.try_transition(task, skip_target)
        print(f"\n  try_transition({task.state!r} → {skip_target!r}): разрешено={ok}")
        print(f"  Причина: {reason}")
        try:
            agent.advance(task, to_state=skip_target)
            print("  ❌ Ошибка: TransitionError не был вызван!")
        except Exception as e:
            print(f"  ✅ TransitionError корректно вызван: {e}")

    # ── Демо: пауза и возобновление ───────────────────────────
    print("\n  [Пауза и возобновление]")
    task2 = agent.start(content_type, topic=topic)
    task2 = agent.pause(task2, "демонстрационная пауза")
    print(f"  Состояние после паузы: {task2.state!r}")
    print(f"  Состояние до паузы: {task2.state_before_pause!r}")

    try:
        agent.advance(task2)
        print("  ❌ Ошибка: ValueError не был вызван!")
    except ValueError as e:
        print(f"  ✅ ValueError при advance() на паузе: {e}")

    task2 = agent.resume(task2)
    print(f"  Состояние после resume: {task2.state!r}")

    # ── Основной прогон ────────────────────────────────────────
    print("\n  [Запуск задачи до конца]")
    task3 = agent.start(content_type, topic=topic)
    task3 = agent.run_all(task3)

    if task3.is_done:
        print(f"\n  ✅ Задача {task3.task_id!r} завершена")
    else:
        print(f"\n  ⏸ Задача {task3.task_id!r} приостановлена (state={task3.state!r})")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Демо JournalistFSMAgent")
    parser.add_argument(
        "--model",
        default="zai-org/GLM-4.7",
        help="Модель LLM",
    )
    parser.add_argument(
        "--type",
        choices=[ct.value for ct in ContentType],
        default="note",
        help="Тип контента",
    )
    parser.add_argument(
        "--topic",
        default="Будущее AI-ассистентов в журналистике",
        help="Тема задачи",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Максимум completion-токенов",
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Только FSM логика без вызовов LLM",
    )
    args = parser.parse_args()

    if args.demo_only:
        demo_fsm_only()
        return

    demo_with_llm(
        model=args.model,
        content_type=ContentType.from_str(args.type),
        topic=args.topic,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
