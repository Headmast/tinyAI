"""
Автоматический прогон трёх тем с намеренными провокациями ошибок FSM.

Темы:
  [1] Техника  — Квантовые вычисления 2025     → ARTICLE
  [2] Финансы  — DeFi vs банки 2025             → NEWS_RESEARCH
  [3] Искусство — Нейросети и современное арт  → REVIEW

Для каждой задачи:
  - показывает цепочку состояний
  - пытается нелегальные переходы (перепрыгнуть шаги)
  - прогоняет задачу по порядку через LLM
  - показывает время каждого шага
"""

import datetime
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

from journalist_agent.workflow import (
    ContentType, JournalistTask, JournalistTaskStorage,
    STATE_PAUSED, STATE_PUBLISHED, TransitionError, WORKFLOWS,
)
from journalist_agent.fsm_agent import JournalistFSMAgent

load_dotenv()


class Tee:
    """Дублирует весь stdout одновременно в терминал и в log-файл."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> None:
        for s in self._streams:
            try:
                s.write(text)
                s.flush()
            except Exception:
                pass

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def fileno(self):
        return self._streams[0].fileno()

    def isatty(self):
        return getattr(self._streams[0], "isatty", lambda: False)()


TASKS = [
    {
        "label":        "🔬 ТЕХНИКА",
        "content_type": ContentType.ARTICLE,
        "topic":        "Квантовые вычисления в 2025: от лаборатории к реальным продуктам",
    },
    {
        "label":        "💰 ФИНАНСЫ",
        "content_type": ContentType.NEWS_RESEARCH,
        "topic":        "DeFi против традиционных банков: кто победит в 2025 году",
    },
    {
        "label":        "🎨 ИСКУССТВО",
        "content_type": ContentType.REVIEW,
        "topic":        "Midjourney 7 и нейросетевое искусство: революция или угроза творчеству",
    },
]

WIDTH = 68


def hr(c="─"): print(c * WIDTH)
def section(title): print(f"\n{'═' * WIDTH}\n  {title}\n{'═' * WIDTH}")


def show_workflow_map(ct: ContentType) -> None:
    wf = WORKFLOWS[ct]
    hr()
    print(f"  РАБОЧИЙ ПРОЦЕСС: {ct.label()}")
    hr()
    for i, step in enumerate(wf.steps):
        nexts = wf.allowed_next(step.state)
        arrow = " → ".join(nexts) if nexts else "✅ (финал)"
        print(f"  {i + 1:2d}. [{step.state:<22s}] {step.label:<25s}  → {arrow}")
    hr()


def demo_forbidden_transitions(agent: JournalistFSMAgent, task: JournalistTask) -> None:
    """Намеренно пробуем недопустимые переходы."""
    wf = task.workflow
    steps = [s.state for s in wf.steps]
    current = task.state

    print(f"\n  --- Проверка защиты FSM (текущее состояние: «{current}») ---")
    # Берём 3 запрещённых варианта: через 1 шаг, через 2, и published
    current_idx = wf.step_index(current)
    bad_targets = []
    if current_idx + 2 < len(steps):
        bad_targets.append(steps[current_idx + 2])
    if current_idx + 3 < len(steps):
        bad_targets.append(steps[current_idx + 3])
    bad_targets.append(STATE_PUBLISHED)

    for target in bad_targets:
        ok, reason = agent.try_transition(task, target)
        if not ok:
            from_idx = wf.step_index(current)
            to_idx   = wf.step_index(target)
            if from_idx >= 0 and to_idx > from_idx + 1:
                skipped = [wf.steps[i].label for i in range(from_idx + 1, to_idx)]
                skip_note = f"  пропуск: {', '.join(skipped)}"
            else:
                skip_note = ""
            print(f"\n  🚫 Попытка: «{current}» → «{target}»")
            if skip_note:
                print(f"     {skip_note}")
            try:
                agent.advance(task, to_state=target)
            except TransitionError as e:
                print(f"     ✅ TransitionError поймана: {e}")
            except ValueError as e:
                print(f"     ✅ ValueError: {e}")

    # Попытка перейти в несуществующее состояние
    print(f"\n  🚫 Попытка: «{current}» → «несуществующее_состояние»")
    try:
        agent.advance(task, to_state="несуществующее_состояние")
    except TransitionError as e:
        print(f"     ✅ TransitionError поймана: {e}")
    except ValueError as e:
        print(f"     ✅ ValueError: {e}")


def demo_pause_resume(agent: JournalistFSMAgent, task: JournalistTask) -> JournalistTask:
    """Демонстрирует паузу и возобновление."""
    print(f"\n  --- Пауза/возобновление (состояние: «{task.state}») ---")

    task = agent.pause(task, "Демонстрационная пауза")
    print(f"  ⏸  Состояние после паузы: {task.state!r}")
    print(f"     Сохранено: state_before_pause = {task.state_before_pause!r}")

    # Попытка advance() на паузе
    print(f"\n  🚫 Попытка advance() на задаче, стоящей на паузе:")
    try:
        agent.advance(task)
    except ValueError as e:
        print(f"     ✅ ValueError поймана: {e}")

    # Попытка paused → published напрямую (без resume)
    wf = task.workflow
    print(f"\n  🚫 Попытка перейти из paused → published напрямую:")
    if not wf.can_transition(STATE_PAUSED, STATE_PUBLISHED):
        print(f"     ✅ Правильно заблокировано: paused → published запрещён")

    task = agent.resume(task)
    print(f"\n  ▶  Состояние после resume: {task.state!r}")
    return task


def run_task_full(agent: JournalistFSMAgent, cfg: dict) -> JournalistTask:
    """Полный прогон одной задачи с показом времени."""
    section(f"{cfg['label']}  |  {cfg['content_type'].label()}")
    print(f"  Тема: {cfg['topic']}")

    show_workflow_map(cfg["content_type"])

    # Создаём задачу
    task = agent.start(cfg["content_type"], topic=cfg["topic"])

    # ── Шаг 1: Проверяем FSM-защиту ──────────────────────────
    demo_forbidden_transitions(agent, task)

    # ── Шаг 2: Пауза/возобновление ───────────────────────────
    task = demo_pause_resume(agent, task)

    # ── Шаг 3: Прогон по порядку ─────────────────────────────
    wf = task.workflow
    total_steps = len(wf.steps) - 1  # без published
    step_num = 0

    print(f"\n  {'─' * 60}")
    print(f"  ВЫПОЛНЯЕМ ЗАДАЧУ ПО ПОРЯДКУ ({total_steps} шагов + финал)")
    print(f"  {'─' * 60}")

    task_start = time.time()

    while not task.is_done:
        current_step = task.current_step
        step_num += 1

        if current_step and current_step.state != STATE_PUBLISHED:
            label = current_step.label
            state = current_step.state
        else:
            label = "Публикация"
            state = STATE_PUBLISHED

        print(f"\n  ⟳  ШАГ {step_num}/{total_steps}  [{state}]  {label}")
        print(f"     Вызываем LLM ({agent.model})...", end="", flush=True)

        t0 = time.time()
        try:
            prev_state = task.state
            task = agent.advance(task)
            elapsed = time.time() - t0

            if task.is_paused:
                print(f" ОШИБКА ШАГ УПАЛ → пауза за {elapsed:.1f}с")
                print(f"     ⏸  Причина паузы: {task.pause_reason!r}")
                print(f"     Возможно ошибка API или пустой ответ LLM")
                break
            else:
                print(f" готово за {elapsed:.1f}с")

            # Краткий превью результата
            if task.step_results:
                last_key = list(task.step_results.keys())[-1]
                last_val = task.step_results[last_key]
                if isinstance(last_val, str):
                    preview = last_val[:120].replace("\n", " ")
                elif isinstance(last_val, list):
                    preview = str(last_val[:2])[:120]
                else:
                    preview = str(last_val)[:120]
                print(f"     Результат [{last_key}]: {preview}...")

            print(f"     → Состояние: {prev_state!r} → {task.state!r}")

        except TransitionError as e:
            elapsed = time.time() - t0
            print(f" ошибка ({elapsed:.1f}с)")
            print(f"     ❌ TransitionError: {e}")
            break
        except Exception as e:
            elapsed = time.time() - t0
            print(f" ошибка ({elapsed:.1f}с)")
            print(f"     ❌ {type(e).__name__}: {e}")
            break

    total_elapsed = time.time() - task_start
    hr("═")
    if task.is_done:
        sr = task.step_results
        print(f"  ✅ ЗАДАЧА ЗАВЕРШЕНА за {total_elapsed:.1f}с")
        print(f"  ID:       {task.task_id}")
        title = sr.get("title", cfg["topic"])
        print(f"  Заголовок: {title}")
        wc = sr.get("word_count", "—")
        print(f"  Слов:     {wc}")
        hr()
        final = sr.get("final_text", sr.get("edited_text", sr.get("draft_text", "")))
        if final:
            lines = final.split("\n")
            print(f"  ТЕКСТ (первые 30 строк):")
            hr()
            for line in lines[:30]:
                print(f"  {line}")
            if len(lines) > 30:
                print(f"  ... [ещё {len(lines) - 30} строк]")
    else:
        print(f"  ⏸  Задача не завершена за {total_elapsed:.1f}с")
        print(f"     Состояние: {task.state!r}")
    hr("═")

    return task


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="zai-org/GLM-4.7")
    p.add_argument("--max-tokens", type=int, default=2500)
    p.add_argument("--task", type=int, choices=[1, 2, 3],
                   help="Запустить только одну задачу")
    p.add_argument("--storage-dir", default="journalist_tasks_demo")
    p.add_argument("--no-log", action="store_true",
                   help="Не записывать лог в файл")
    args = p.parse_args()

    os.makedirs(args.storage_dir, exist_ok=True)
    log_file = None
    if not args.no_log:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        task_tag = f"_task{args.task}" if args.task else "_all"
        log_path = os.path.join(args.storage_dir, f"run{task_tag}_{ts}.log")
        log_file = open(log_path, "w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, log_file)
        print(f"  📝 Лог записывается в: {log_path}", flush=True)

    client = OpenAI(
        api_key=os.getenv("CLOUD_API_KEY"),
        base_url=os.getenv("CLOUD_BASE_URL") or "https://foundation-models.api.cloud.ru/v1",
    )

    storage = JournalistTaskStorage(base_dir=args.storage_dir)
    agent = JournalistFSMAgent(
        client=client,
        model=args.model,
        storage=storage,
        verbose=True,
        stream_tokens=True,
        max_completion_tokens=args.max_tokens,
        temperature=0.6,
    )

    print("\n" + "═" * WIDTH)
    print(f"  JOURNALIST FSM — 3 ТЕМЫ, МОДЕЛЬ: {args.model}")
    print(f"  max_tokens={args.max_tokens}  storage={args.storage_dir}")
    print("═" * WIDTH)

    tasks_to_run = [TASKS[args.task - 1]] if args.task else TASKS

    for cfg in tasks_to_run:
        run_task_full(agent, cfg)
        if len(tasks_to_run) > 1:
            print("\n  Пауза 2с перед следующей задачей...")
            time.sleep(2)

    print(f"\n  Задачи сохранены в: {args.storage_dir}/")
    all_tasks = storage.list_tasks()
    print(f"  Всего задач в хранилище: {len(all_tasks)}")
    for t in all_tasks[:5]:
        print(f"    {t['task_id']}  {t['content_type']:<15s}  {t['state']:<20s}  {t['topic'][:40]}")

    if log_file is not None:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"  📝 Лог сохранён: {log_path}")


if __name__ == "__main__":
    main()
