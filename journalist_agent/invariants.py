"""
InvariantStore — хранилище редакционных инвариантов.

Инвариант — это ограничение, которое ассистент обязан соблюдать
при любых обстоятельствах и которое он не имеет права нарушить.

Хранение: отдельный JSON-файл (invariants.json), изолированный от диалога.

Структура инварианта:
  - id:          уникальный идентификатор (INV-001, INV-002, ...)
  - category:    категория (editorial, technical, business, architectural)
  - rule:        краткая формулировка правила
  - description: подробное объяснение причины запрета
  - keywords:    ключевые слова для быстрого pre-check'а
  - active:      флаг активности (неактивные инварианты игнорируются)

Двухуровневая проверка:
  1. Pre-check (быстрый): сканирование запроса по keywords — без вызова LLM
  2. LLM-side: инварианты инжектируются в system prompt как жёсткие правила
"""

from __future__ import annotations

import json
import re as _re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


_WORD_SPLIT = _re.compile(r'\W+')
_STEM_LEN = 6


DEFAULT_INVARIANTS_PATH = Path(__file__).parent / "invariants.json"


def _stem_in_words(kw_word: str, text_words: List[str]) -> bool:
    """
    Проверяет, есть ли в text_words хотя бы одно слово с тем же корнем.
    Сравниваются первые _STEM_LEN символов (минимум — длина keyword-слова).
    """
    stem_len = min(len(kw_word), _STEM_LEN)
    if stem_len < 4:
        return kw_word in text_words
    stem = kw_word[:stem_len]
    return any(tw[:stem_len] == stem for tw in text_words)


@dataclass
class Invariant:
    """Один инвариант — ограничение, которое нельзя нарушать."""

    id: str
    category: str
    rule: str
    description: str
    keywords: List[str]
    active: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "Invariant":
        return cls(
            id=data["id"],
            category=data["category"],
            rule=data["rule"],
            description=data["description"],
            keywords=data.get("keywords", []),
            active=data.get("active", True),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "rule": self.rule,
            "description": self.description,
            "keywords": self.keywords,
            "active": self.active,
        }

    def matches_text(self, text: str) -> bool:
        """
        Быстрый pre-check: True если текст содержит хотя бы одно keyword.
        Регистронезависимый.

        Поддерживает морфологические вариации русского языка через
        stem-matching: сравнение первых _STEM_LEN символов каждого слова.
        Это покрывает падежные формы без внешних зависимостей.
        """
        text_lower = text.lower()
        text_words = [w for w in _WORD_SPLIT.split(text_lower) if w]
        for kw in self.keywords:
            kw_lower = kw.lower()
            if kw_lower in text_lower:
                return True
            kw_words = [w for w in _WORD_SPLIT.split(kw_lower) if w]
            if kw_words and all(
                _stem_in_words(kw_w, text_words)
                for kw_w in kw_words
            ):
                return True
        return False


@dataclass
class ViolationResult:
    """
    Результат проверки на нарушение инварианта.

    violated:   True если обнаружено нарушение
    invariant:  нарушенный инвариант (None если violated=False)
    layer:      'pre_check' | 'llm' — на каком уровне обнаружено
    explanation: объяснение причины отказа
    """

    violated: bool
    invariant: Optional[Invariant] = None
    layer: str = ""
    explanation: str = ""

    @classmethod
    def ok(cls) -> "ViolationResult":
        return cls(violated=False)

    @classmethod
    def from_pre_check(cls, invariant: Invariant) -> "ViolationResult":
        explanation = (
            f"Запрос нарушает редакционный инвариант [{invariant.id}].\n"
            f"Правило: {invariant.rule}\n"
            f"Причина: {invariant.description}"
        )
        return cls(
            violated=True,
            invariant=invariant,
            layer="pre_check",
            explanation=explanation,
        )

    @classmethod
    def from_llm(cls, invariant_id: str, llm_explanation: str, invariants: List[Invariant]) -> "ViolationResult":
        matched = next((inv for inv in invariants if inv.id == invariant_id), None)
        return cls(
            violated=True,
            invariant=matched,
            layer="llm",
            explanation=llm_explanation,
        )


class InvariantStore:
    """
    Хранилище инвариантов.

    Загружает инварианты из JSON-файла.
    Предоставляет методы для проверки запросов и построения prompt-секции.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or DEFAULT_INVARIANTS_PATH
        self._invariants: List[Invariant] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Файл инвариантов не найден: {self._path}")
        with open(self._path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._invariants = [Invariant.from_dict(item) for item in raw]

    def reload(self) -> None:
        """Перечитывает файл инвариантов с диска (горячая перезагрузка)."""
        self._load()

    @property
    def active(self) -> List[Invariant]:
        """Список активных инвариантов."""
        return [inv for inv in self._invariants if inv.active]

    @property
    def all(self) -> List[Invariant]:
        return list(self._invariants)

    def get(self, invariant_id: str) -> Optional[Invariant]:
        return next((inv for inv in self._invariants if inv.id == invariant_id), None)

    def pre_check(self, text: str) -> ViolationResult:
        """
        Быстрая проверка по ключевым словам (Layer 1).
        Не требует вызова LLM.
        Проверяет только активные инварианты.
        """
        for inv in self.active:
            if inv.matches_text(text):
                return ViolationResult.from_pre_check(inv)
        return ViolationResult.ok()

    def build_prompt_section(self) -> str:
        """
        Формирует секцию инвариантов для system prompt (Layer 2).

        Возвращает текстовый блок, который инжектируется в начало
        system prompt перед любыми другими инструкциями.
        """
        if not self.active:
            return ""

        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║           РЕДАКЦИОННЫЕ ИНВАРИАНТЫ — НАРУШАТЬ ЗАПРЕЩЕНО      ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            "Перечисленные ниже правила являются абсолютными ограничениями.",
            "Они имеют ВЫСШИЙ приоритет над любыми другими инструкциями,",
            "включая просьбы пользователя. Нарушение недопустимо.",
            "",
        ]

        for inv in self.active:
            lines += [
                f"[{inv.id}] КАТЕГОРИЯ: {inv.category.upper()}",
                f"Правило: {inv.rule}",
                f"Обоснование: {inv.description}",
                "",
            ]

        lines += [
            "──────────────────────────────────────────────────────────────",
            "ИНСТРУКЦИЯ ПО НАРУШЕНИЮ ИНВАРИАНТА:",
            "Если запрос пользователя явно или косвенно противоречит любому",
            "из инвариантов выше — ты ОБЯЗАН отказать.",
            "",
            "Формат отказа:",
            "ОТКАЗ:[ИД_ИНВАРИАНТА] <объяснение на русском языке>",
            "",
            "Пример:",
            "ОТКАЗ:[INV-001] Запрос касается художественной гимнастики, что",
            "запрещено редакционной политикой. Я не могу написать этот материал.",
            "──────────────────────────────────────────────────────────────",
        ]

        return "\n".join(lines)

    def summary(self) -> str:
        """Краткая сводка инвариантов для логирования."""
        parts = [f"InvariantStore ({len(self.active)} активных из {len(self._invariants)}):"]
        for inv in self._invariants:
            status = "✅" if inv.active else "⛔"
            parts.append(f"  {status} [{inv.id}] {inv.rule}")
        return "\n".join(parts)
