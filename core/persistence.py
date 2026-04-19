"""
Утилиты для работы с JSON-файлами.

Единые функции чтения/записи JSON, устраняющие дублирование
open() → json.load/dump паттерна по всему проекту.

Использование:
    from core.persistence import load_json, save_json

    data = load_json("memory_data/facts.json", default=[])
    save_json("memory_data/facts.json", data)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union


def load_json(path: Union[str, Path], default: Any = None) -> Any:
    """Загружает данные из JSON-файла.

    Args:
        path: Путь к JSON-файлу.
        default: Значение по умолчанию, если файл не найден.

    Returns:
        Десериализованные данные или default.
    """
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Сохраняет данные в JSON-файл.

    Args:
        path: Путь к JSON-файлу.
        data: Данные для сериализации.
        indent: Отступ для форматирования (по умолчанию 2).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
