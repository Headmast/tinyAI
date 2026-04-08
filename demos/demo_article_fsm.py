"""
Демонстрация FSM-агента написания статей (День 13).

Агент проходит состояния: planning → execution → validation → done

Возможности:
  1. Старт новой задачи (агент сам выбирает тему)
  2. Старт с заданной темой
  3. Пауза на любом этапе (Ctrl+C)
  4. Продолжение с точного места остановки (--resume <task_id>)

Запуск:
  python3 demo_article_fsm.py                        # новая задача, тема на выбор агента
  python3 demo_article_fsm.py --topic "Тема"         # новая задача с конкретной темой
  python3 demo_article_fsm.py --resume <task_id>     # продолжить после паузы
  python3 demo_article_fsm.py --list                 # показать все задачи
  python3 demo_article_fsm.py --pause-after planning # новая задача с паузой после planning
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from news_agent.fsm_state import TaskPhase, TaskState, TaskStateStorage, PHASE_STEPS
from news_agent.fsm_agent import ArticleFSMAgent

load_dotenv()


MODEL = "zai-org/GLM-4.7"
BASE_URL = os.getenv("BASE_URL", "https://foundation-models.api.cloud.ru/v1")
API_KEY = os.getenv("CLOUD_API_KEY", "")
TASKS_DIR = "tasks"


def create_client() -> OpenAI:
    if not API_KEY:
        print("❌ Не задан CLOUD_API_KEY в .env")
        sys.exit(1)
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def cmd_list(storage: TaskStateStorage) -> None:
    """Выводит список всех задач."""
    tasks = storage.list_tasks(n=20)
    if not tasks:
        print("  (нет сохранённых задач)")
        return

    print(f"\n  {'ID':>8}  {'Фаза':<12}  {'Шаг':>4}  {'Пауза':>6}  {'Тема':<35}  {'Обновлено'}")
    print(f"  {'─' * 8}  {'─' * 12}  {'─' * 4}  {'─' * 6}  {'─' * 35}  {'─' * 19}")
    for t in tasks:
        topic = (t.get("topic") or "(нет)")[:33]
        paused = "⏸ ДА" if t.get("paused") else "▶ нет"
        updated = t.get("updated_at", "")[:19].replace("T", " ")
        print(
            f"  {t['task_id']:>8}  {t['phase']:<12}  {t['current_step']:>4}  "
            f"{paused:>6}  {topic:<35}  {updated}"
        )
    print()


def cmd_new(
    client: OpenAI,
    storage: TaskStateStorage,
    topic: str = None,
    pause_after: str = None,
) -> None:
    """Запускает новую задачу."""
    agent = ArticleFSMAgent(
        client=client,
        model=MODEL,
        storage=storage,
        verbose=True,
    )

    if pause_after:
        print(f"\n  ℹ️  Пауза будет выставлена после фазы: {pause_after}")
        state = TaskState.new(topic=topic)

        target_phase = TaskPhase(pause_after)
        while state.phase != TaskPhase.DONE:
            step_info = state.get_current_step_info()
            if step_info is None:
                break

            if state.phase == target_phase and state.current_step == len(PHASE_STEPS[target_phase]) - 1:
                result = agent._execute_step(state, step_info["name"])
                if result is not None:
                    state.record_step_result(step_info["name"], result)
                    storage.save(state)
                    state.advance_step()
                    storage.save(state)

                agent.pause(state, reason=f"Пауза по запросу после фазы {pause_after}")
                print(f"\n⏸  Задача приостановлена после фазы [{pause_after}]")
                print(f"   task_id: {state.task_id}")
                print(f"   Продолжить: python3 demo_article_fsm.py --resume {state.task_id}")
                print(state.format_phase_progress())
                return

            result = agent._execute_step(state, step_info["name"])
            if result is None:
                state.pause(f"Ошибка шага {step_info['name']}")
                storage.save(state)
                break
            state.record_step_result(step_info["name"], result)
            storage.save(state)
            agent._print_step_done(step_info["name"], result, state)
            state.advance_step()
            storage.save(state)
    else:
        agent.run(topic=topic)


def cmd_resume(client: OpenAI, storage: TaskStateStorage, task_id: str) -> None:
    """Продолжает задачу с последнего сохранённого состояния."""
    state = storage.load(task_id)
    if state is None:
        print(f"❌ Задача '{task_id}' не найдена")
        sys.exit(1)

    print(f"\n  Загружена задача: {task_id}")
    print(f"  Фаза: {state.phase.value}  Шаг: {state.current_step}")
    print(f"  Тема: {state.topic or state.article_data.get('topic', '(нет)')}")
    if state.paused:
        print(f"  Была на паузе: {state.pause_reason}")
    print(state.format_phase_progress())

    agent = ArticleFSMAgent(
        client=client,
        model=MODEL,
        storage=storage,
        verbose=True,
    )
    agent.run(state=state)


def cmd_status(storage: TaskStateStorage, task_id: str) -> None:
    """Показывает детальный статус задачи."""
    state = storage.load(task_id)
    if state is None:
        print(f"❌ Задача '{task_id}' не найдена")
        sys.exit(1)
    print()
    print(state.format_status())
    print(state.format_phase_progress())
    print(f"\n  Выполненные шаги:")
    for r in state.step_results:
        keys = list(r.output.keys())[:4]
        print(f"    [{r.phase}] {r.step_name} → {keys}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="FSM-агент написания статей (День 13)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--topic", type=str, default=None, help="Тема статьи (опционально)")
    parser.add_argument("--resume", type=str, default=None, metavar="TASK_ID", help="Продолжить задачу по ID")
    parser.add_argument("--list", action="store_true", help="Список всех задач")
    parser.add_argument("--status", type=str, default=None, metavar="TASK_ID", help="Статус конкретной задачи")
    parser.add_argument(
        "--pause-after",
        type=str,
        default=None,
        choices=["planning", "execution", "validation"],
        metavar="PHASE",
        help="Поставить паузу после указанной фазы",
    )
    parser.add_argument("--tasks-dir", type=str, default=TASKS_DIR, help="Директория для хранения задач")

    args = parser.parse_args()

    storage = TaskStateStorage(base_dir=args.tasks_dir)

    print("\n" + "═" * 65)
    print("  ДЕНЬ 13: FSM-АГЕНТ — НАПИСАНИЕ СТАТЕЙ")
    print(f"  Модель: {MODEL}")
    print("  Состояния: planning → execution → validation → done")
    print("═" * 65)

    if args.list:
        cmd_list(storage)
        return

    if args.status:
        cmd_status(storage, args.status)
        return

    client = create_client()

    if args.resume:
        cmd_resume(client, storage, args.resume)
    else:
        cmd_new(client, storage, topic=args.topic, pause_after=args.pause_after)


if __name__ == "__main__":
    main()
