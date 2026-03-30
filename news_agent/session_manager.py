"""
SessionManager — управление диалоговыми сессиями.

Ключевая концепция: в отличие от веб-чата (ChatGPT, Claude), где сервер
хранит историю автоматически, API полностью stateless. Каждый запрос
независим. Чтобы модель «помнила» диалог — нужно вручную передавать
полный массив messages[] при каждом вызове. Именно это делает данный модуль.

С версии 5.0 поддерживается компрессия контекста: длинные истории автоматически
сжимаются через LLM-суммаризацию, что экономит токены при длинных диалогах.

Классы:
    ConversationSession  — одна диалоговая сессия с историей сообщений
    SessionStorage       — файловое хранилище сессий (sessions/*.json)
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from news_agent.token_counter import TokenCounter
from news_agent.context_compressor import ContextCompressor, CompressionStats
from news_agent.context_strategies import (
    ContextStrategy, create_strategy, strategy_from_dict,
    SlidingWindowStrategy, StickyFactsStrategy, BranchingStrategy,
)


MODEL_CONTEXT_SIZES: Dict[str, int] = {
    "zai-org/GLM-4.7-Flash": 128_000,
    "zai-org/GLM-4.7": 128_000,
    "gpt-5-nano": 128_000,
    "gpt-5.4": 128_000,
    "gpt-5.4-mini": 128_000,
}
DEFAULT_CONTEXT_SIZE = 128_000

CONTEXT_WARN_THRESHOLD = 0.80


class ConversationSession:
    """
    Диалоговая сессия.

    Хранит полный список сообщений (messages[]), который при каждом
    обращении к API передаётся целиком — это и есть механизм «памяти» модели.

    Атрибуты:
        session_id          — уникальный идентификатор (8 символов)
        name                — человекочитаемое имя сессии
        model               — модель, используемая в сессии
        system_prompt       — системный промпт (если задан)
        messages            — история сообщений [{"role": ..., "content": ...}]
        created_at          — ISO-timestamp создания
        updated_at          — ISO-timestamp последнего изменения
        status              — "active" | "closed"
        token_usage         — накопленная статистика токенов
        compression_enabled — True если включена компрессия истории
        compressor          — экземпляр ContextCompressor (или None)
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        name: str = "",
        model: str = "zai-org/GLM-4.7-Flash",
        system_prompt: str = "",
    ) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())[:8]
        self.name: str = name or f"Сессия {datetime.now().strftime('%d.%m %H:%M')}"
        self.model: str = model
        self.system_prompt: str = system_prompt
        self.messages: List[Dict[str, Any]] = []
        self.created_at: str = datetime.now().isoformat()
        self.updated_at: str = self.created_at
        self.status: str = "active"
        self.token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.compression_enabled: bool = False
        self.compressor: Optional[ContextCompressor] = None
        self.context_strategy: Optional[ContextStrategy] = None

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    @property
    def max_context_tokens(self) -> int:
        """Максимальный размер контекстного окна для текущей модели."""
        return MODEL_CONTEXT_SIZES.get(self.model, DEFAULT_CONTEXT_SIZE)

    def estimate_tokens(self) -> int:
        """
        Подсчёт токенов в истории через tiktoken (если доступен)
        или эвристику chars÷4 как fallback.
        Включает overhead сообщений (~4 токена на сообщение).
        """
        counter = TokenCounter(model=self.model)
        result = counter.count_messages(self.messages)
        return result["total"]

    def add_user_message(self, content: str) -> None:
        """Добавляет сообщение пользователя в историю."""
        self.messages.append({"role": "user", "content": content})
        self.updated_at = datetime.now().isoformat()

    def add_assistant_message(self, content: str) -> None:
        """Добавляет ответ ассистента в историю."""
        self.messages.append({"role": "assistant", "content": content})
        self.updated_at = datetime.now().isoformat()

    def update_token_usage(self, prompt: int, completion: int) -> None:
        """Обновляет накопленную статистику токенов."""
        self.token_usage["prompt_tokens"] += prompt
        self.token_usage["completion_tokens"] += completion
        self.token_usage["total_tokens"] += prompt + completion

    def get_context_info(self) -> Dict[str, Any]:
        """
        Возвращает сводку использования контекстного окна.

        Возвращает dict с ключами:
            used_tokens     — оценочное число использованных токенов
            max_tokens      — максимум для данной модели
            percentage      — процент заполнения (0–100)
            messages_count  — число сообщений в истории
            warning         — True если использовано > CONTEXT_WARN_THRESHOLD
        """
        estimated = self.estimate_tokens()
        max_tokens = self.max_context_tokens
        percentage = (estimated / max_tokens * 100) if max_tokens > 0 else 0.0
        return {
            "used_tokens": estimated,
            "max_tokens": max_tokens,
            "percentage": round(percentage, 1),
            "messages_count": len(self.messages),
            "warning": percentage >= CONTEXT_WARN_THRESHOLD * 100,
        }

    def format_context_bar(self) -> str:
        """
        Возвращает строку с визуальным индикатором заполнения контекста.

        Пример вывода:
            🟢 Контекст: 1,234 / 128,000 токенов [████░░░░░░░░░░░░░░░░] 0.96%
        """
        info = self.get_context_info()
        used = info["used_tokens"]
        max_t = info["max_tokens"]
        pct = info["percentage"]

        bar_len = 20
        filled = min(int(bar_len * pct / 100), bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        if pct >= 80:
            icon = "🔴"
        elif pct >= 50:
            icon = "🟡"
        else:
            icon = "🟢"

        return f"{icon} Контекст: {used:,} / {max_t:,} токенов [{bar}] {pct:.1f}%"

    def close(self) -> None:
        """Закрывает сессию (статус 'closed'). Загрузка и просмотр остаются доступны."""
        self.status = "closed"
        self.updated_at = datetime.now().isoformat()

    def get_messages_for_api(self) -> List[Dict[str, Any]]:
        """
        Возвращает список сообщений для передачи в API.

        Приоритет: context_strategy > compression > полная история.
        """
        if self.context_strategy is not None:
            return self.context_strategy.get_messages_for_api(self.messages)
        if self.compression_enabled and self.compressor and self.compressor.summaries:
            return self.compressor.get_compressed_messages(self.messages)
        return list(self.messages)

    # ─────────────────────────────────────────────────────────────
    # Управление стратегиями контекста
    # ─────────────────────────────────────────────────────────────

    def set_strategy(self, strategy_name: str, **kwargs) -> str:
        """
        Устанавливает стратегию управления контекстом.

        Args:
            strategy_name: имя стратегии (sliding_window, sticky_facts, branching)
            **kwargs: параметры стратегии (window_size и т.д.)

        Returns:
            Строка-подтверждение
        """
        self.context_strategy = create_strategy(strategy_name, **kwargs)
        return f"Стратегия '{strategy_name}' активирована"

    def clear_strategy(self) -> str:
        """Отключает стратегию контекста (возврат к полной истории)."""
        old_name = self.context_strategy.name if self.context_strategy else "none"
        self.context_strategy = None
        return f"Стратегия '{old_name}' отключена, используется полная история"

    def get_strategy_info(self) -> Dict[str, Any]:
        """Возвращает информацию о текущей стратегии."""
        if self.context_strategy is None:
            return {"strategy": "none", "description": "Полная история (без стратегии)"}
        return self.context_strategy.get_stats()

    # ─────────────────────────────────────────────────────────────
    # Управление компрессией
    # ─────────────────────────────────────────────────────────────

    def enable_compression(
        self,
        summarize_every: int = 10,
        keep_last_n: int = 6,
    ) -> None:
        """
        Включает компрессию истории для этой сессии.

        Если компрессор уже был создан ранее — переиспользует его
        (сохраняет накопленные summaries).

        Args:
            summarize_every — суммаризировать каждые N сообщений (default 10)
            keep_last_n     — хранить последние N сообщений без сжатия (default 6)
        """
        self.compression_enabled = True
        if self.compressor is None:
            self.compressor = ContextCompressor(
                summarize_every=summarize_every,
                keep_last_n=keep_last_n,
                model=self.model,
            )

    def disable_compression(self) -> None:
        """Отключает компрессию. Накопленные summaries сохраняются."""
        self.compression_enabled = False

    def maybe_compress(self, client: Any) -> Optional[CompressionStats]:
        """
        Проверяет необходимость компрессии и при необходимости запускает её.

        Вызывать после каждого обмена (пользователь + ассистент), чтобы
        своевременно сжимать накопленную историю.

        Args:
            client: openai.OpenAI клиент для вызова LLM-суммаризатора

        Returns:
            CompressionStats если сжатие было выполнено, иначе None
        """
        if not self.compression_enabled or self.compressor is None:
            return None
        return self.compressor.compress(client, self.messages)

    def get_compression_info(self) -> Dict[str, Any]:
        """
        Возвращает сводку по состоянию компрессии.

        Всегда возвращает dict с ключами:
            enabled          — включена ли компрессия
            compression_count — число выполненных сжатий
            messages_summarized — число суммаризированных сообщений
            total_tokens_saved  — накопленная оценка экономии токенов
            summarize_every  — параметр порога (или None)
            keep_last_n      — параметр «живого хвоста» (или None)
        """
        if not self.compressor:
            return {
                "enabled": self.compression_enabled,
                "compression_count": 0,
                "messages_summarized": 0,
                "total_tokens_saved": 0,
                "summarize_every": None,
                "keep_last_n": None,
            }
        return {
            "enabled": self.compression_enabled,
            "compression_count": self.compressor.compression_count,
            "messages_summarized": self.compressor.summary_covers_up_to,
            "total_tokens_saved": self.compressor.total_tokens_saved,
            "summarize_every": self.compressor.summarize_every,
            "keep_last_n": self.compressor.keep_last_n,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Сериализует сессию в словарь для сохранения в JSON."""
        data: Dict[str, Any] = {
            "session_id": self.session_id,
            "name": self.name,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "token_usage": self.token_usage,
            "compression_enabled": self.compression_enabled,
        }
        if self.compressor is not None:
            data["compressor"] = self.compressor.to_dict()
        if self.context_strategy is not None:
            data["context_strategy"] = self.context_strategy.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """Десериализует сессию из словаря."""
        session = cls.__new__(cls)
        session.session_id = data["session_id"]
        session.name = data.get("name", "Без имени")
        session.model = data.get("model", "zai-org/GLM-4.7-Flash")
        session.system_prompt = data.get("system_prompt", "")
        session.messages = data.get("messages", [])
        session.created_at = data.get("created_at", datetime.now().isoformat())
        session.updated_at = data.get("updated_at", datetime.now().isoformat())
        session.status = data.get("status", "active")
        session.token_usage = data.get(
            "token_usage",
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        session.compression_enabled = data.get("compression_enabled", False)
        compressor_data = data.get("compressor")
        if compressor_data:
            session.compressor = ContextCompressor.from_dict(compressor_data)
        else:
            session.compressor = None
        strategy_data = data.get("context_strategy")
        if strategy_data:
            session.context_strategy = strategy_from_dict(strategy_data)
        else:
            session.context_strategy = None
        return session

    def __repr__(self) -> str:
        return (
            f"ConversationSession(id={self.session_id!r}, "
            f"name={self.name!r}, "
            f"model={self.model!r}, "
            f"messages={len(self.messages)}, "
            f"status={self.status!r})"
        )


class SessionStorage:
    """
    Файловое хранилище диалоговых сессий.

    Структура:
        sessions/
            index.json          — индекс всех сессий (метаданные)
            <session_id>.json   — полная сессия с историей сообщений

    Каждое изменение автоматически обновляет index.json.
    """

    def __init__(self, base_dir: str = "sessions") -> None:
        self.base_dir = Path(base_dir)
        self.index_file = self.base_dir / "index.json"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_file.exists():
            self._write_index([])

    def _read_index(self) -> List[Dict[str, Any]]:
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_index(self, index: List[Dict[str, Any]]) -> None:
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _make_index_entry(self, session: "ConversationSession") -> Dict[str, Any]:
        return {
            "session_id": session.session_id,
            "name": session.name,
            "model": session.model,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "status": session.status,
            "messages_count": len(session.messages),
        }

    def save(self, session: "ConversationSession") -> None:
        """Сохраняет сессию на диск и обновляет индекс."""
        session_file = self.base_dir / f"{session.session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

        index = self._read_index()
        existing_idx = next(
            (i for i, e in enumerate(index) if e["session_id"] == session.session_id),
            None,
        )
        entry = self._make_index_entry(session)
        if existing_idx is not None:
            index[existing_idx] = entry
        else:
            index.insert(0, entry)
        self._write_index(index)

    def load(self, session_id: str) -> Optional["ConversationSession"]:
        """
        Загружает сессию по ID.
        Возвращает None если сессия не найдена или повреждена.
        """
        session_file = self.base_dir / f"{session_id}.json"
        if not session_file.exists():
            return None
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ConversationSession.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def list_sessions(
        self,
        n: int = 20,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает список сессий из индекса (только метаданные).

        Args:
            n:      максимальное число записей
            status: фильтр по статусу ("active" / "closed" / None = все)
        """
        index = self._read_index()
        if status:
            index = [e for e in index if e.get("status") == status]
        return index[:n]

    def delete(self, session_id: str) -> bool:
        """Удаляет сессию с диска и из индекса."""
        session_file = self.base_dir / f"{session_id}.json"
        if not session_file.exists():
            return False
        session_file.unlink()
        index = [e for e in self._read_index() if e["session_id"] != session_id]
        self._write_index(index)
        return True

    def find_active(self) -> List[Dict[str, Any]]:
        """Возвращает все активные (незакрытые) сессии."""
        return self.list_sessions(status="active")

    def find_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Находит первую сессию с указанным именем (регистронезависимо)."""
        for entry in self._read_index():
            if entry.get("name", "").lower() == name.lower():
                return entry
        return None
