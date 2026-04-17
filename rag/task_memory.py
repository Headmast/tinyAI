"""
PersistentTaskMemory — персистентная рабочая память для диалогового чата.

Хранит состояние задачи:
  - goal:        цель диалога
  - clarifications:  что пользователь уже уточнил
  - constraints:     зафиксированные ограничения/термины
  - user_messages:   история сообщений пользователя (ограниченная)
  - assistant_messages: история ответов ассистента (ограниченная)

Персистентность: JSON-файл, загружается/сохраняется после каждого turn.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class PersistentTaskMemory:
    """
    Рабочая память для одного диалогового сеанса.

    Автоматически извлекает goal, clarifications и constraints
    из сообщений пользователя. Сохраняется в JSON.
    """

    def __init__(
        self,
        storage_path: str = "rag_data/task_memory.json",
        max_history: int = 20,
    ) -> None:
        self._path = Path(storage_path)
        self.max_history = max_history

        # Core state
        self.goal: str = ""
        self.clarifications: List[str] = []
        self.constraints: List[str] = []
        self.user_messages: List[Dict[str, str]] = []
        self.assistant_messages: List[Dict[str, str]] = []
        self.turn_count: int = 0

        # Timestamps
        self.created_at: str = ""
        self.updated_at: str = ""

        # Internal tracking
        self._seen_clarifications: set[str] = set()
        self._seen_constraints: set[str] = set()

        self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_goal(self, goal: str) -> None:
        """Явно установить цель диалога."""
        goal = goal.strip()
        if goal and goal != self.goal:
            self.goal = goal
            self._save()

    def add_clarification(self, clarification: str) -> None:
        """Добавить уточнение пользователя."""
        clarification = clarification.strip()
        if clarification and clarification not in self._seen_clarifications:
            self.clarifications.append(clarification)
            self._seen_clarifications.add(clarification)
            self.updated_at = datetime.now().isoformat()
            self._save()

    def add_constraint(self, constraint: str) -> None:
        """Добавить ограничение или термин."""
        constraint = constraint.strip()
        if constraint and constraint not in self._seen_constraints:
            self.constraints.append(constraint)
            self._seen_constraints.add(constraint)
            self.updated_at = datetime.now().isoformat()
            self._save()

    def add_user_message(self, text: str) -> int:
        """Добавить сообщение пользователя. Возвращает номер turn."""
        self.user_messages.append({
            "text": text,
            "timestamp": datetime.now().isoformat(),
        })
        # Trim history
        if len(self.user_messages) > self.max_history:
            self.user_messages = self.user_messages[-self.max_history:]
        self.updated_at = datetime.now().isoformat()
        self.turn_count = len(self.user_messages)
        self._save()
        return self.turn_count

    def add_assistant_message(self, text: str) -> None:
        """Добавить сообщение ассистента."""
        self.assistant_messages.append({
            "text": text,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.assistant_messages) > self.max_history:
            self.assistant_messages = self.assistant_messages[-self.max_history:]
        self.updated_at = datetime.now().isoformat()
        self._save()

    def get_last_n_user_messages(self, n: int = 5) -> List[Dict[str, str]]:
        """Получить последние N сообщений пользователя."""
        return self.user_messages[-n:] if self.user_messages else []

    def get_last_n_exchanges(self, n: int = 5) -> List[Dict[str, str]]:
        """
        Получить последние N обменов (user + assistant pair)
        в формате чередования.
        """
        exchanges: List[Dict[str, str]] = []
        max_len = min(len(self.user_messages), len(self.assistant_messages))
        start = max(0, max_len - n)
        for i in range(start, max_len):
            exchanges.append({
                "role": "user",
                "content": self.user_messages[i]["text"],
            })
            if i < len(self.assistant_messages):
                exchanges.append({
                    "role": "assistant",
                    "content": self.assistant_messages[i]["text"],
                })
        return exchanges

    def get_summary(self) -> str:
        """Текстовая сводка состояния задачи."""
        parts: List[str] = []

        if self.goal:
            parts.append(f"Цель: {self.goal}")
        if self.clarifications:
            parts.append("Уточнения:\n" + "\n".join(
                f"  • {c}" for c in self.clarifications
            ))
        if self.constraints:
            parts.append("Ограничения:\n" + "\n".join(
                f"  • {c}" for c in self.constraints
            ))
        if not parts:
            parts.append("(состояние задачи пусто)")
        return "\n\n".join(parts)

    def get_memory_context_block(self) -> str:
        """
        Формировать блок состояния задачи для вставки в system prompt.
        """
        block = "### Контекст задачи:\n"

        if self.goal:
            block += f"- Цель диалога: {self.goal}\n"
        else:
            block += "- Цель диалога: не определена\n"

        if self.clarifications:
            block += "- Уже уточнено:\n"
            for c in self.clarifications[-5:]:  # последние 5
                block += f"    • {c}\n"

        if self.constraints:
            block += "- Ограничения:\n"
            for c in self.constraints[-5:]:  # последние 5
                block += f"    • {c}\n"

        block += f"- Количество сообщений: {self.turn_count}\n"
        return block

    def reset(self) -> None:
        """Сбросить все поля (новая задача)."""
        self.goal = ""
        self.clarifications = []
        self.constraints = []
        self.user_messages = []
        self.assistant_messages = []
        self.turn_count = 0
        self._seen_clarifications.clear()
        self._seen_constraints.clear()
        self.updated_at = datetime.now().isoformat()
        self._save()

    def auto_extract_user_input(self, text: str) -> None:
        """
        Автоматически извлечь goal, clarifications, constraints
        из ввода пользователя по эвристикам.
        """
        text_lower = text.lower().strip()

        # ——— Goal detection ———
        goal_patterns = [
            r"(?:моя цель|цель|хочу узнать|хочу понять|мне нужно|задача)\s*[:\-]?\s*(.+)",
            r"(?:давай обсудим|расскажи мне|объясни мне|помоги мне)\s+[:\-]?\s*(.+)",
            r"(?:расскажи|объясни|покажи)\s+(?:как|что|почему|зачем)\s+(.+?)(?:\s*$|\s*[.?!,])",
        ]
        if not self.goal:
            for pattern in goal_patterns:
                m = re.search(pattern, text_lower)
                if m:
                    self.set_goal(m.group(1).strip())
                    break

            # Если первый вопрос — это первый ввод, цель = краткое содержание
            if not self.goal and self.turn_count == 0:
                # Сохраняем первый вопрос как "seed" для будущей цели
                seed = text[:80] if len(text) > 80 else text
                self.set_goal(f"Ответить на вопрос: {seed}")

        # ——— Clarification detection ———
        clarification_patterns = [
            r"(?:а что насчёт|а как|а почему|а можно|подскажи|уточни|объясни подробнее|расскажи больше|подробнее)\s*(.+)",
            r"(?:я имею в виду|я имел в виду|то есть|имеется в виду)\s+(.+)",
            r"(?:а как насчёт|а если)\s+(.+)",
            r"(?:давай (?:обсудим|разберём|поговорим))\s+(.+)",
        ]
        for pattern in clarification_patterns:
            m = re.search(pattern, text_lower)
            if m and len(text) > 10:
                self.add_clarification(text.strip())
                return  # Это уточнение, не нужно проверять дальше

        # Если текст длинный (>20 слов) и содержит вопросительные слова — может быть уточнением
        if self.user_messages:  # Уже был предыдущий ввод
            words = text.split()
            if len(words) > 20:
                # Длинный ввод после первого — может быть новым аспектом
                if any(w in text_lower for w in
                       ["а ещё", "а также", "кроме того", "также хочу", "ещё один"]):
                    self.add_clarification(text.strip()[:100] + "...")

        # ——— Constraint detection ———
        constraint_patterns = [
            r"(?:только|лишь)\s+(.+?)(?:\s*[.!?,]\s*|$)",
            r"(?:не используй|не используй|без|исключая)\s+(.+?)(?:\s*[.!?,]\s*|$)",
            r"(?:ограничение|конкретно|именно)\s*[:\-]?\s*(.+)",
            r"(?:используй|применяй|опирайся)\s+(?:только\s+)?(?:на\s+)?(.+)",
        ]
        for pattern in constraint_patterns:
            m = re.search(pattern, text_lower)
            if m:
                candidate = m.group(1).strip()
                if 3 < len(candidate) < 150:
                    self.add_constraint(text.strip())
                    break

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Загрузить состояние из JSON-файла."""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.goal = data.get("goal", "")
                self.clarifications = data.get("clarifications", [])
                self.constraints = data.get("constraints", [])
                self.user_messages = data.get("user_messages", [])
                self.assistant_messages = data.get("assistant_messages", [])
                self.turn_count = data.get("turn_count", 0)
                self.created_at = data.get("created_at", "")
                self.updated_at = data.get("updated_at", "")
                # Rehydrate seen sets
                self._seen_clarifications = set(self.clarifications)
                self._seen_constraints = set(self.constraints)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  ⚠️  Не удалось загрузить память: {e}")
                self._init_blank()
        else:
            self._init_blank()

    def _init_blank(self) -> None:
        """Инициализировать пустое состояние."""
        now = datetime.now().isoformat()
        self.created_at = now
        self.updated_at = now

    def _save(self) -> None:
        """Сохранить состояние в JSON-файл."""
        self.updated_at = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = self.updated_at

        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "goal": self.goal,
            "clarifications": self.clarifications,
            "constraints": self.constraints,
            "user_messages": self.user_messages,
            "assistant_messages": self.assistant_messages,
            "turn_count": self.turn_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._path)

    def dump(self) -> Dict[str, Any]:
        """Возвращает полный дамп состояния."""
        return {
            "goal": self.goal,
            "clarifications": list(self.clarifications),
            "constraints": list(self.constraints),
            "turn_count": self.turn_count,
            "user_message_count": len(self.user_messages),
            "assistant_message_count": len(self.assistant_messages),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
