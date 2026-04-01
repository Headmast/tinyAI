"""
UserProfile — профиль пользователя для персонализации ассистента.

Профиль описывает предпочтения пользователя:
  - стиль ответов (формальный, живой, технический...)
  - предпочитаемый формат (списки, таблицы, нарратив...)
  - ограничения (длина, терминология, стандарты...)
  - контекст роли (род деятельности, экспертиза...)

ProfileManager хранит предустановленные и пользовательские профили,
обеспечивает сериализацию и загрузку из JSON.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class UserProfile:
    """
    Профиль пользователя.

    Содержит все параметры персонализации, которые инжектируются
    в system prompt при каждом запросе к LLM.
    """

    name: str
    role: str
    description: str

    # Стиль общения
    response_style: str = ""
    tone: str = ""
    language_level: str = ""

    # Формат ответов
    preferred_format: str = ""
    structure_preferences: List[str] = field(default_factory=list)

    # Ограничения
    constraints: List[str] = field(default_factory=list)

    # Области экспертизы / интересов
    expertise_areas: List[str] = field(default_factory=list)

    # Произвольные предпочтения (ключ-значение)
    custom_preferences: Dict[str, str] = field(default_factory=dict)

    # Метаданные
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def build_prompt_section(self) -> str:
        """
        Формирует секцию для system prompt на основе профиля.

        Возвращает отформатированный текст, готовый для вставки
        в системное сообщение.
        """
        parts: List[str] = []

        parts.append(f"Имя пользователя: {self.name}")
        parts.append(f"Роль: {self.role}")

        if self.description:
            parts.append(f"О пользователе: {self.description}")

        if self.response_style:
            parts.append(f"Стиль ответов: {self.response_style}")

        if self.tone:
            parts.append(f"Тон общения: {self.tone}")

        if self.language_level:
            parts.append(f"Уровень языка: {self.language_level}")

        if self.preferred_format:
            parts.append(f"Предпочитаемый формат: {self.preferred_format}")

        if self.structure_preferences:
            parts.append(
                "Структура ответов:\n"
                + "\n".join(f"  - {s}" for s in self.structure_preferences)
            )

        if self.constraints:
            parts.append(
                "Ограничения и правила:\n"
                + "\n".join(f"  - {c}" for c in self.constraints)
            )

        if self.expertise_areas:
            parts.append(
                "Области экспертизы пользователя: "
                + ", ".join(self.expertise_areas)
            )

        if self.custom_preferences:
            pref_lines = [f"  {k}: {v}" for k, v in self.custom_preferences.items()]
            parts.append("Дополнительные предпочтения:\n" + "\n".join(pref_lines))

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализует профиль в словарь."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Десериализует профиль из словаря."""
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })

    def update(self, **kwargs) -> None:
        """Обновляет поля профиля."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now().isoformat()

    def __repr__(self) -> str:
        return f"UserProfile(name={self.name!r}, role={self.role!r})"


# ─────────────────────────────────────────────────────────────
# Предустановленные профили
# ─────────────────────────────────────────────────────────────

BUILTIN_PROFILES: Dict[str, UserProfile] = {

    "system_analyst": UserProfile(
        name="Системный аналитик",
        role="Системный аналитик",
        description=(
            "Опытный системный аналитик, работающий с требованиями к ПО, "
            "бизнес-процессами и техническими спецификациями. "
            "Ценит точность, структурированность и соответствие стандартам."
        ),
        response_style=(
            "Формальный, структурированный, технически точный. "
            "Ответы должны быть логически выстроены с чёткой иерархией."
        ),
        tone="деловой, нейтральный, аналитический",
        language_level="профессиональный, с использованием технической терминологии",
        preferred_format=(
            "Структурированные списки, таблицы, нумерованные пункты. "
            "Диаграммы описывать текстом (UML, BPMN нотация). "
            "Каждый тезис — с обоснованием."
        ),
        structure_preferences=[
            "Начинать с краткого резюме (executive summary)",
            "Декомпозировать задачу на подзадачи",
            "Использовать нумерованные списки для последовательностей",
            "Выделять риски и зависимости",
            "Заканчивать выводами и рекомендациями",
        ],
        constraints=[
            "Избегать расплывчатых формулировок — только конкретика",
            "Каждое утверждение подкреплять фактами или логикой",
            "Придерживаться стандартов: ГОСТ, IEEE, UML где уместно",
            "Не использовать эмодзи и неформальную лексику",
            "Разделять функциональные и нефункциональные требования",
        ],
        expertise_areas=[
            "требования к ПО", "бизнес-анализ", "UML/BPMN",
            "проектирование систем", "интеграции", "базы данных",
            "API-проектирование", "тестирование",
        ],
        custom_preferences={
            "глубина_анализа": "максимальная — все edge-case и альтернативы",
            "формат_требований": "User Story + Acceptance Criteria",
            "приоритизация": "MoSCoW (Must/Should/Could/Won't)",
            "метрики": "всегда включать количественные оценки где возможно",
        },
    ),

    "journalist": UserProfile(
        name="Журналист",
        role="Журналист",
        description=(
            "Журналист-универсал с опытом в деловых и технологических СМИ. "
            "Ценит живой слог, точность фактов и доступность для широкой аудитории."
        ),
        response_style=(
            "Живой, яркий, увлекательный. Использовать storytelling, "
            "конкретные примеры и аналогии. Писать так, чтобы было интересно "
            "читать неподготовленному человеку."
        ),
        tone="энергичный, дружелюбный, но профессиональный",
        language_level="доступный широкой аудитории, без сложного жаргона",
        preferred_format=(
            "Нарративный текст с подзаголовками. Короткие абзацы (2-3 предложения). "
            "Цитаты и факты вплетены в текст органично. "
            "Перевёрнутая пирамида: главное — в начале."
        ),
        structure_preferences=[
            "Цепляющий заголовок без кликбейта",
            "Лид-абзац: Кто? Что? Когда? Где? Почему?",
            "Тело: факты по убыванию важности",
            "Цитаты и статистика для усиления",
            "Заключение: вывод или открытый вопрос",
        ],
        constraints=[
            "Объективность — показывать разные точки зрения",
            "Проверка фактов — не утверждать недоказанное",
            "Активный залог предпочтительнее пассивного",
            "Без канцелярита и воды",
            "Конкретные детали вместо общих слов",
            "Эмодзи допустимы в заголовках для соцсетей",
        ],
        expertise_areas=[
            "новостная журналистика", "интервью", "расследования",
            "копирайтинг", "SMM", "контент-стратегия",
            "технологические тренды", "деловые новости",
        ],
        custom_preferences={
            "целевая_аудитория": "широкая — от студентов до топ-менеджеров",
            "длина_абзаца": "2-4 предложения максимум",
            "стиль_заголовков": "конкретный, информативный, с числами где уместно",
            "call_to_action": "завершать вопросом или призывом к обсуждению",
        },
    ),
}


class ProfileManager:
    """
    Управление профилями пользователей.

    Хранит предустановленные и пользовательские профили.
    Обеспечивает сохранение/загрузку из JSON-файла.
    """

    def __init__(self, storage_path: str = "memory_data/profiles.json") -> None:
        self._path = Path(storage_path)
        self._profiles: Dict[str, UserProfile] = {}
        self._active_profile: Optional[str] = None
        self._load()

    def _load(self) -> None:
        """Загружает пользовательские профили из файла."""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key, profile_data in data.get("profiles", {}).items():
                    self._profiles[key] = UserProfile.from_dict(profile_data)
                self._active_profile = data.get("active_profile")
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self) -> None:
        """Сохраняет пользовательские профили в файл."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "profiles": {k: v.to_dict() for k, v in self._profiles.items()},
            "active_profile": self._active_profile,
            "updated_at": datetime.now().isoformat(),
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_profile(self, name: str) -> Optional[UserProfile]:
        """
        Возвращает профиль по имени.
        Сначала ищет в пользовательских, затем в предустановленных.
        """
        if name in self._profiles:
            return self._profiles[name]
        if name in BUILTIN_PROFILES:
            return BUILTIN_PROFILES[name]
        return None

    def get_active_profile(self) -> Optional[UserProfile]:
        """Возвращает текущий активный профиль."""
        if self._active_profile:
            return self.get_profile(self._active_profile)
        return None

    def set_active(self, name: str) -> bool:
        """Устанавливает активный профиль. Возвращает True если профиль найден."""
        profile = self.get_profile(name)
        if profile is None:
            return False
        self._active_profile = name
        self._save()
        return True

    def add_profile(self, key: str, profile: UserProfile) -> None:
        """Добавляет или обновляет пользовательский профиль."""
        self._profiles[key] = profile
        self._save()

    def remove_profile(self, key: str) -> bool:
        """Удаляет пользовательский профиль. Встроенные удалить нельзя."""
        if key in self._profiles:
            del self._profiles[key]
            if self._active_profile == key:
                self._active_profile = None
            self._save()
            return True
        return False

    def list_profiles(self) -> Dict[str, Dict[str, str]]:
        """Возвращает список всех доступных профилей (встроенных + пользовательских)."""
        result: Dict[str, Dict[str, str]] = {}
        for key, profile in BUILTIN_PROFILES.items():
            result[key] = {
                "name": profile.name,
                "role": profile.role,
                "type": "builtin",
                "active": key == self._active_profile,
            }
        for key, profile in self._profiles.items():
            result[key] = {
                "name": profile.name,
                "role": profile.role,
                "type": "custom",
                "active": key == self._active_profile,
            }
        return result

    def clear_active(self) -> None:
        """Снимает активный профиль."""
        self._active_profile = None
        self._save()

    @property
    def active_profile_name(self) -> Optional[str]:
        return self._active_profile
