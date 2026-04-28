#!/usr/bin/env python3
"""
RAG Chat CLI — мини-чат с RAG-поиском, источниками и памятью задачи.

Команды:
  /help <вопрос>   — спросить ассистента о проекте (RAG + Git)
  /goal <текст>    — явно задать цель диалога
  /memory          — показать состояние рабочей памяти
  /history         — показать последние сообщения диалога
  /reset           — сбросить память задачи (новая задача)
  /stats           — показать статистику
  /sources         — показать источники последнего ответа
  /quit, /exit, q  — выйти
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Project root setup
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from rag.rag_chat_agent import RagChatAgent
from dev_assistant import DevAssistant

load_dotenv()

# ── Visual constants ─────────────────────────────────────────────────────────
SEP = "═" * 70
SUB = "─" * 70
PROMPT_PREFIX = "💬 Вы: "
BANNER = """
╔══════════════════════════════════════════════════════════╗
║           🤖 RAG Chat — Умный чат с RAG-поиском          ║
║          📚 Источники • 💾 Память • ✏️ Цитаты            ║
╚══════════════════════════════════════════════════════════╝
"""


class RagChatCLI:
    """Интерактивный CLI для RAG-чата с памятью задачи."""

    def __init__(
        self,
        agent: Optional[RagChatAgent] = None,
        verbose: bool = False,
    ) -> None:
        self.agent = agent or RagChatAgent(verbose=verbose)
        self.verbose = verbose
        self._last_result: Optional[Dict] = None
        self._running = True
        self._dev_assistant: Optional[DevAssistant] = None

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Запустить основной цикл чата."""
        print(BANNER)
        print(f"  Модель: {self.agent.model}")
        print(f"  Индекс: {self.agent.index_dir}")
        print(f"  Память: {self.agent.memory._path}")
        self._print_memory_status()
        print(f"\n  Введите вопрос или команду (начинается с /).")
        print(f"  Наберите {SUB} для разделения блоков.\n")

        while self._running:
            try:
                user_input = input(f"\n{PROMPT_PREFIX}").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  👋 До свидания!")
                self._running = False
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            self._process_turn(user_input)

    # ── Command handlers ──────────────────────────────────────────────────────

    def _handle_command(self, cmd: str) -> None:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        HANDLERS = {
            "/help": self._cmd_help,
            "/goal": self._cmd_goal,
            "/memory": self._cmd_memory,
            "/history": self._cmd_history,
            "/reset": self._cmd_reset,
            "/stats": self._cmd_stats,
            "/sources": self._cmd_sources,
            "/quit": self._quit,
            "/exit": self._quit,
            "q": self._quit,
        }

        handler = HANDLERS.get(command)
        if handler:
            handler(arg)
        else:
            print(f"\n  ❌ Неизвестная команда: {command}")
            print(f"  Доступные команды:")
            print(f"    /help <вопрос> — спросить ассистента о проекте")
            print(f"    /goal <текст>  — задать цель диалога")
            print(f"    /memory        — показать память задачи")
            print(f"    /history       — последние сообщения")
            print(f"    /reset         — сбросить память")
            print(f"    /stats         — статистика")
            print(f"    /sources       — источники последнего ответа")
            print(f"    /quit          — выйти")

    def _cmd_help(self, arg: str) -> None:
        """Ассистент разработчика: отвечает на вопросы о проекте."""
        if not arg:
            print(f"\n  ℹ️  Использование: /help <ваш вопрос о проекте>")
            print(f"  Примеры:")
            print(f"    /help Как устроена архитектура проекта?")
            print(f"    /help Какие MCP-серверы есть в проекте?")
            print(f"    /help На какой я ветке?")
            return

        # Lazy init
        if self._dev_assistant is None:
            print(f"\n  ⏳ Инициализация ассистента...")
            self._dev_assistant = DevAssistant(verbose=self.verbose)

        print(f"\n  ⏳ Ищу ответ в документации проекта...")
        try:
            answer = self._dev_assistant.answer_help(arg)
        except Exception as e:
            print(f"\n  ❌ Ошибка: {e}")
            return

        print(f"\n{SEP}")
        print(f"  🛠️  Developer Assistant")
        print(f"{SUB}")
        print(f"\n{answer}")
        print(f"\n{SEP}")

    def _cmd_goal(self, arg: str) -> None:
        if not arg:
            current = self.agent.memory.goal or "(не задана)"
            print(f"\n  🎯 Текущая цель: {current}")
            return
        self.agent.set_goal(arg)
        print(f"\n  🎯 Цель установлена: {arg}")

    def _cmd_memory(self, arg: str) -> None:
        print(f"\n{SEP}")
        print(f"  🧠 Состояние памяти задачи:")
        print(f"{SUB}")
        self._print_memory_status()
        print(f"{SEP}")

    def _cmd_history(self, arg: str) -> None:
        try:
            n = int(arg) if arg else 5
        except ValueError:
            n = 5

        print(f"\n{SEP}")
        print(f"  📜 Последние {n} обменов:")
        print(f"{SUB}")

        exchanges = self.agent.memory.get_last_n_exchanges(n)
        if not exchanges:
            print(f"  (история пуста)")
        else:
            for ex in exchanges:
                if ex["role"] == "user":
                    print(f"  👤 Вы: {ex['content'][:120]}")
                else:
                    preview = ex["content"][:120].replace("\n", " ")
                    print(f"  🤖 Ассистент: {preview}")
        print(f"{SEP}")

    def _cmd_reset(self, arg: str) -> None:
        confirm = input("\n  ⚠️  Сбросить память задачи? (y/N): ").strip().lower()
        if confirm in ("y", "yes", "да"):
            self.agent.reset_memory()
            print("\n  🧹 Память задачи сброшена.")
            self._last_result = None
        else:
            print("\n  Отмена.")

    def _cmd_stats(self, arg: str) -> None:
        stats = self.agent.get_stats()
        print(f"\n{SEP}")
        print(f"  📊 Статистика:")
        print(f"{SUB}")
        print(f"  Поклонов (turns):      {stats['turns']}")
        print(f"  Токенов запросов:      {stats['prompt_tokens']}")
        print(f"  Токенов ответов:       {stats['completion_tokens']}")
        print(f"  Всего токенов:         {stats['total_tokens']}")
        print(f"  Общее время:           {stats['total_elapsed_ms'] / 1000:.1f}s")
        print(f"  Цель задана:           {'✅' if stats['goal_set'] else '❌'}")
        print(f"{SEP}")

    def _cmd_sources(self, arg: str) -> None:
        if not self._last_result:
            print("\n  (нет предыдущих ответов)")
            return
        print(f"\n{SEP}")
        print(f"  📚 Источники последнего ответа:")
        print(f"{SUB}")
        self._print_sources(self._last_result)
        print(f"{SEP}")

    def _quit(self, arg: str) -> None:
        self._running = False
        print("\n  👋 До свидания!")

    # ── Turn processing ───────────────────────────────────────────────────────

    def _process_turn(self, user_input: str) -> None:
        """Обработать один вопрос пользователя."""
        print(f"\n  ⏳ Думаю...")

        start = time.monotonic()

        try:
            result = self.agent.chat_turn(user_input)
        except Exception as e:
            print(f"\n  ❌ Ошибка: {e}")
            return

        elapsed = time.monotonic() - start
        self._last_result = result

        # Print answer
        self._print_answer(result, elapsed)

    # ── Output formatting ─────────────────────────────────────────────────────

    def _print_answer(self, result: Dict, elapsed: float) -> None:
        """Форматированный вывод ответа."""
        # Separator
        print(f"\n{SEP}")

        # Status
        if result.get("is_uncertain"):
            print(f"  ⚠️  Ответ с низкой уверенностью (confidence: {result['confidence']:.2f})")
        else:
            print(f"  ✅ Уверенный ответ")
        print(f"  ⏱️  {elapsed:.1f}s")
        print(f"{SUB}")

        # Main answer
        answer = result.get("answer", "")
        print(f"\n{answer}\n")

        # Sources
        sources = result.get("sources", [])
        if sources:
            print(f"  📚 ИСТОЧНИКИ:")
            for s in sources:
                section = s.get("section", "")
                section_str = f" / {section}" if section else ""
                print(f"     [{s['index']}] {s['source']}{section_str}  (score: {s['score']:.4f})")

        # Quotes
        quotes = result.get("quotes", [])
        if quotes:
            print(f"\n  💬 ЦИТАТЫ:")
            for i, q in enumerate(quotes, start=1):
                verified = "✓" if q.get("verified") else "?"
                print(f"     [{i}] {verified} {q['text'][:150]}...")

        # Memory update
        goal = result.get("goal", "")
        if goal:
            print(f"\n  🎯 Цель: {goal}")

        print(f"{SEP}")

    def _print_memory_status(self) -> None:
        """Краткий статус памяти."""
        mem = self.agent.get_memory_state()
        goal = mem.get("goal", "") or "(не задана)"
        clarifications = mem.get("clarifications", [])
        constraints = mem.get("constraints", [])
        turns = mem.get("turn_count", 0)

        print(f"\n  🧠 Память задачи:")
        print(f"     Цель:          {goal}")
        print(f"     Уточнения:     {len(clarifications)}")
        print(f"     Ограничения:   {len(constraints)}")
        print(f"     Сообщений:     {turns}")

    def _print_sources(self, result: Dict) -> None:
        sources = result.get("sources", [])
        if not sources:
            print(f"  (нет источников)")
            return
        for s in sources:
            section = s.get("section", "")
            section_str = f" / {section}" if section else ""
            print(f"  [{s['index']}] {s['source']}{section_str}  (score: {s['score']:.4f})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAG Chat CLI")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Подробный вывод"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="LLM модель"
    )
    parser.add_argument(
        "--index-dir", type=str, default=None, help="Директория с RAG-индексом"
    )
    args = parser.parse_args()

    agent_kwargs = {"verbose": args.verbose}
    if args.model:
        agent_kwargs["model"] = args.model
    if args.index_dir:
        agent_kwargs["index_dir"] = args.index_dir

    cli = RagChatCLI(**agent_kwargs)
    cli.run()


if __name__ == "__main__":
    main()
