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
from code_reviewer import CodeReviewer
from dev_assistant import DevAssistant
from support_assistant import SupportAssistant

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
        self._code_reviewer: Optional[CodeReviewer] = None
        self._support_assistant: Optional[SupportAssistant] = None

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
            "/support": self._cmd_support,
            "/review": self._cmd_review,
            "/search": self._cmd_search,
            "/analyze": self._cmd_analyze,
            "/fix": self._cmd_fix,
            "/refactor": self._cmd_refactor,
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
            print(f"  ❌ Неизвестная команда: {command}")
            print(f"  Доступные команды:")
            print(f"    /help <вопрос> — спросить ассистента о проекте")
            print(f"    /support <вопрос> — спросить службу поддержки")
            print(f"    /review [base] — AI-ревью кода (по умолчанию working dir)")
            print(f"    /search <паттерн> — поиск по коду + AI-анализ")
            print(f"    /analyze <файл> — анализ файла на проблемы")
            print(f"    /fix <файл> [описание] — найти и исправить проблемы")
            print(f"    /refactor <файл> [цель] — предложить рефакторинг")
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

    def _cmd_support(self, arg: str) -> None:
        """Ассистент поддержки: отвечает на вопросы пользователей о продукте."""
        if not arg:
            print(f"\n  ℹ️  Использование: /support <вопрос>")
            print(f"  Опции:")
            print(f"    /support <вопрос>                — ответ из FAQ")
            print(f"    /support --user USER_ID <вопрос> — с контекстом пользователя")
            print(f"    /support --ticket TICKET_ID <вопрос> — с контекстом тикета")
            print(f"  Примеры:")
            print(f"    /support Почему не работает авторизация?")
            print(f"    /support --user user_1 Почему не работает авторизация?")
            print(f"    /support --ticket ticket_1 Помогите с моей проблемой")
            return

        # Parse --user and --ticket flags
        user_id = None
        ticket_id = None
        parts = arg.split()
        question_parts = []
        i = 0
        while i < len(parts):
            if parts[i] == "--user" and i + 1 < len(parts):
                user_id = parts[i + 1]
                i += 2
            elif parts[i] == "--ticket" and i + 1 < len(parts):
                ticket_id = parts[i + 1]
                i += 2
            else:
                question_parts.append(parts[i])
                i += 1
        question = " ".join(question_parts)

        if not question:
            print(f"\n  ❌ Укажите вопрос после флагов.")
            return

        # Lazy init
        if self._support_assistant is None:
            print(f"\n  ⏳ Инициализация ассистента поддержки...")
            self._support_assistant = SupportAssistant(verbose=self.verbose)

        print(f"\n  ⏳ Ищу ответ в FAQ...")
        try:
            answer = self._support_assistant.answer(
                question, user_id=user_id, ticket_id=ticket_id
            )
        except Exception as e:
            print(f"\n  ❌ Ошибка: {e}")
            return

        print(f"\n{SEP}")
        print(f"  🎧  Support Assistant")
        print(f"{SUB}")
        print(f"\n{answer}")
        print(f"\n{SEP}")

    def _cmd_review(self, arg: str) -> None:
        """AI-ревью кода: анализ diff с GPT-5-nano."""
        # Lazy init
        if self._code_reviewer is None:
            print(f"\n  ⏳ Инициализация code reviewer...")
            self._code_reviewer = CodeReviewer(verbose=self.verbose)

        if arg.strip() == "staged":
            print(f"\n  ⏳ Анализирую staged изменения...")
            try:
                report = self._code_reviewer.review_staged()
            except Exception as e:
                print(f"\n  ❌ Ошибка: {e}")
                return
        elif arg.strip():
            print(f"\n  ⏳ Анализирую PR: текущая ветка vs {arg.strip()}...")
            try:
                report = self._code_reviewer.review_pr(base_branch=arg.strip())
            except Exception as e:
                print(f"\n  ❌ Ошибка: {e}")
                return
        else:
            print(f"\n  ⏳ Анализирую незакоммиченные изменения...")
            try:
                report = self._code_reviewer.review_working()
            except Exception as e:
                print(f"\n  ❌ Ошибка: {e}")
                return

        print(f"\n{SEP}")
        print(f"  🔍 AI Code Review")
        print(f"{SUB}")
        print(f"\n{report}")
        print(f"\n{SEP}")

    def _cmd_search(self, arg: str) -> None:
        """AI-поиск по коду: находит паттерн и анализирует."""
        if not arg:
            print(f"\n  ℹ️  Использование: /search <паттерн>")
            print(f"  Примеры:")
            print(f"    /search eval(")
            print(f"    /search TODO")
            print(f"    /search subprocess.run")
            return

        if self._code_reviewer is None:
            print(f"\n  ⏳ Инициализация code reviewer...")
            self._code_reviewer = CodeReviewer(verbose=self.verbose)

        print(f"\n  ⏳ Ищу '{arg}' в коде...")
        try:
            report = self._code_reviewer.search_code(arg)
        except Exception as e:
            print(f"\n  ❌ Ошибка: {e}")
            return

        print(f"\n{SEP}")
        print(f"  🔎 AI Code Search")
        print(f"{SUB}")
        print(f"\n{report}")
        print(f"\n{SEP}")

    def _cmd_analyze(self, arg: str) -> None:
        """Анализ файла: баги, безопасность, качество."""
        if not arg:
            print(f"\n  ℹ️  Использование: /analyze <файл>")
            print(f"  Примеры:")
            print(f"    /analyze core/config.py")
            print(f"    /analyze mcp_server.py")
            return

        if self._code_reviewer is None:
            print(f"\n  ⏳ Инициализация code reviewer...")
            self._code_reviewer = CodeReviewer(verbose=self.verbose)

        print(f"\n  ⏳ Анализирую {arg.strip()}...")
        try:
            report = self._code_reviewer.analyze_file(arg.strip())
        except Exception as e:
            print(f"\n  ❌ Ошибка: {e}")
            return

        print(f"\n{SEP}")
        print(f"  📊 AI Code Analysis")
        print(f"{SUB}")
        print(f"\n{report}")
        print(f"\n{SEP}")

    def _cmd_fix(self, arg: str) -> None:
        """Найти и исправить проблемы в файле."""
        if not arg:
            print(f"\n  ℹ️  Использование: /fix <файл> [описание проблемы]")
            print(f"  Примеры:")
            print(f"    /fix mcp_server.py")
            print(f"    /fix core/config.py ошибка при отсутствии API-ключа")
            return

        if self._code_reviewer is None:
            print(f"\n  ⏳ Инициализация code reviewer...")
            self._code_reviewer = CodeReviewer(verbose=self.verbose)

        parts = arg.strip().split(maxsplit=1)
        file_path = parts[0]
        issue = parts[1] if len(parts) > 1 else ""

        desc = f" ({issue})" if issue else ""
        print(f"\n  ⏳ Ищу исправления для {file_path}{desc}...")
        try:
            report = self._code_reviewer.fix_code(file_path, issue)
        except Exception as e:
            print(f"\n  ❌ Ошибка: {e}")
            return

        print(f"\n{SEP}")
        print(f"  🔧 AI Code Fix")
        print(f"{SUB}")
        print(f"\n{report}")
        print(f"\n{SEP}")

    def _cmd_refactor(self, arg: str) -> None:
        """Предложить рефакторинг файла."""
        if not arg:
            print(f"\n  ℹ️  Использование: /refactor <файл> [цель]")
            print(f"  Примеры:")
            print(f"    /refactor mcp_server.py")
            print(f"    /refactor mcp_server.py разделить на модули")
            print(f"    /refactor core/config.py улучшить читаемость")
            return

        if self._code_reviewer is None:
            print(f"\n  ⏳ Инициализация code reviewer...")
            self._code_reviewer = CodeReviewer(verbose=self.verbose)

        parts = arg.strip().split(maxsplit=1)
        file_path = parts[0]
        goal = parts[1] if len(parts) > 1 else ""

        desc = f" ({goal})" if goal else ""
        print(f"\n  ⏳ Готовлю рефакторинг {file_path}{desc}...")
        try:
            report = self._code_reviewer.refactor(file_path, goal)
        except Exception as e:
            print(f"\n  ❌ Ошибка: {e}")
            return

        print(f"\n{SEP}")
        print(f"  ♻️ AI Refactoring")
        print(f"{SUB}")
        print(f"\n{report}")
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
