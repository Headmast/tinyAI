"""
ContextCompressor — управление контекстом через сжатие истории диалога.

Ключевая концепция:
    Проблема длинных диалогов: с ростом истории каждый API-запрос
    содержит всё больше токенов, расход растёт линейно.

    Решение — скользящая компрессия:
      ┌──────────────────────────────────────────────────────────────────┐
      │  [SYSTEM]  [SUMMARY_1 + SUMMARY_2 + ...]  [msg_N-5] ... [msg_N] │
      └──────────────────────────────────────────────────────────────────┘
      Старые сообщения заменяются компактным summary, последние keep_last_n
      сообщений сохраняются «как есть» — для поддержания живого контекста.

Параметры по умолчанию:
    summarize_every = 10  — суммаризировать каждые 10 сообщений
    keep_last_n     = 6   — хранить последние 6 сообщений без сжатия

Классы:
    CompressionStats   — статистика одного сжатия
    ContextCompressor  — основной механизм компрессии
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from news_agent.token_counter import TokenCounter


# ─────────────────────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────────────────────

SUMMARY_SYSTEM_PROMPT = (
    "Ты ведёшь краткое изложение диалога. "
    "Создай сжатое summary следующего фрагмента переписки, "
    "сохранив ключевые факты, решения, вопросы и важный контекст. "
    "Пиши на том же языке, что и переписка. "
    "Отвечай только текстом summary, без лишних слов."
)

SUMMARY_USER_TEMPLATE = (
    "Сделай краткое изложение этого фрагмента диалога:\n\n{history}"
)

SUMMARY_PREFIX = "📋 Краткое изложение предыдущей части диалога:\n\n"

DEFAULT_SUMMARIZE_EVERY = 10
DEFAULT_KEEP_LAST_N = 6
DEFAULT_SUMMARY_MAX_TOKENS = 600

_MODELS_NO_TEMPERATURE: frozenset = frozenset({"gpt-5-nano", "o1", "o3", "o3-mini", "o4-mini"})


# ─────────────────────────────────────────────────────────────
# CompressionStats
# ─────────────────────────────────────────────────────────────

@dataclass
class CompressionStats:
    """
    Статистика одного акта сжатия истории.

    Атрибуты:
        compression_index    — порядковый номер сжатия (1, 2, ...)
        messages_compressed  — сколько сообщений было сжато
        tokens_before        — токены в контексте до сжатия
        tokens_after         — токены в контексте после сжатия
        summary_length       — длина сгенерированного summary в символах
    """

    compression_index: int
    messages_compressed: int
    tokens_before: int
    tokens_after: int
    summary_length: int

    @property
    def tokens_saved(self) -> int:
        """Количество сэкономленных токенов."""
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def compression_ratio(self) -> float:
        """Коэффициент сжатия (меньше 1 — лучше)."""
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before

    def format(self) -> str:
        """Возвращает строку с описанием результата сжатия."""
        ratio_pct = (1.0 - self.compression_ratio) * 100
        return (
            f"Сжатие #{self.compression_index}: "
            f"{self.messages_compressed} сообщ. → 1 summary  |  "
            f"~{self.tokens_saved:,} токенов сэкономлено  |  "
            f"сжатие {ratio_pct:.0f}%"
        )


# ─────────────────────────────────────────────────────────────
# ContextCompressor
# ─────────────────────────────────────────────────────────────

class ContextCompressor:
    """
    Механизм скользящей компрессии истории диалога.

    Алгоритм:
        1. Накапливается история сообщений (user + assistant).
        2. Когда накопилось >= summarize_every необработанных сообщений
           (за пределами последних keep_last_n), запускается суммаризация.
        3. LLM генерирует краткое изложение, которое сохраняется в summaries[].
        4. При следующем get_compressed_messages() возвращается:
               [system_msgs] + [summary_message] + [последние keep_last_n сообщений]
           вместо полной истории.

    Атрибуты:
        summarize_every      — порог новых сообщений для запуска суммаризации
        keep_last_n          — число «живых» сообщений в хвосте
        summaries            — накопленные тексты summary
        summary_covers_up_to — число chat-сообщений, уже включённых в summaries
        compression_count    — число выполненных суммаризаций
        total_tokens_saved   — накопленная оценка сэкономленных токенов

    Пример:
        compressor = ContextCompressor(summarize_every=10, keep_last_n=6)

        # В _run_chat_loop после каждого ответа модели:
        stats = compressor.compress(client, session.messages)
        if stats:
            print(stats.format())

        # При формировании запроса:
        api_messages = compressor.get_compressed_messages(session.messages)
    """

    def __init__(
        self,
        summarize_every: int = DEFAULT_SUMMARIZE_EVERY,
        keep_last_n: int = DEFAULT_KEEP_LAST_N,
        model: str = "zai-org/GLM-4.7-Flash",
    ) -> None:
        self.summarize_every: int = summarize_every
        self.keep_last_n: int = keep_last_n
        self.model: str = model
        self.summaries: List[str] = []
        self.summary_covers_up_to: int = 0
        self.compression_count: int = 0
        self.total_tokens_saved: int = 0
        self._counter = TokenCounter(model=model)

    # ─────────────────────────────────────────────────────────────
    # Публичный API
    # ─────────────────────────────────────────────────────────────

    def needs_compression(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Проверяет, нужно ли запускать суммаризацию прямо сейчас.

        Сжатие нужно, когда число необработанных сообщений (вне «живого хвоста»)
        достигло порога summarize_every.

        Args:
            messages: полный список сообщений ConversationSession.messages
        """
        chat_msgs = self._chat_only(messages)
        # сообщения вне хвоста keep_last_n, ещё не суммаризированные
        compressible = len(chat_msgs) - self.summary_covers_up_to - self.keep_last_n
        return compressible >= self.summarize_every

    def compress(
        self,
        client: Any,
        messages: List[Dict[str, Any]],
    ) -> Optional[CompressionStats]:
        """
        Запускает суммаризацию если она нужна, возвращает статистику.

        Вызывает LLM для генерации краткого изложения фрагмента истории,
        обновляет внутреннее состояние (summaries, summary_covers_up_to).

        Args:
            client:   openai.OpenAI клиент для вызова LLM
            messages: полный список сообщений из ConversationSession.messages

        Returns:
            CompressionStats — если сжатие выполнено
            None             — если сжатие не требовалось
        """
        if not self.needs_compression(messages):
            return None

        chat_msgs = self._chat_only(messages)
        compress_end = len(chat_msgs) - self.keep_last_n
        to_compress = chat_msgs[self.summary_covers_up_to:compress_end]

        if not to_compress:
            return None

        # Токены до сжатия
        tokens_before = self._counter.count_messages(
            self.get_compressed_messages(messages)
        )["total"]

        summary_text = self._call_llm_for_summary(client, to_compress)

        self.summaries.append(summary_text)
        self.summary_covers_up_to += len(to_compress)
        self.compression_count += 1

        # Токены после сжатия
        tokens_after = self._counter.count_messages(
            self.get_compressed_messages(messages)
        )["total"]

        saved = max(0, tokens_before - tokens_after)
        self.total_tokens_saved += saved

        return CompressionStats(
            compression_index=self.compression_count,
            messages_compressed=len(to_compress),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            summary_length=len(summary_text),
        )

    def get_compressed_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Возвращает список сообщений с применённой компрессией для передачи в API.

        Если суммаризация ни разу не выполнялась — возвращает полную историю.
        Иначе:
            [system_msgs] + [summary_message] + [recent chat msgs]

        Args:
            messages: полный список сообщений ConversationSession.messages
        """
        if not self.summaries:
            return list(messages)

        system_msgs = [m for m in messages if m["role"] == "system"]
        chat_msgs = self._chat_only(messages)

        combined_summary = "\n\n---\n\n".join(self.summaries)
        summary_message: Dict[str, Any] = {
            "role": "system",
            "content": SUMMARY_PREFIX + combined_summary,
        }

        recent = chat_msgs[self.summary_covers_up_to:]
        return system_msgs + [summary_message] + recent

    def get_comparison(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Сравнивает использование токенов с компрессией и без.

        Returns dict:
            full_messages_count  — сообщений в полной истории
            full_tokens          — токенов без компрессии
            compressed_messages_count — сообщений в сжатой версии
            compressed_tokens    — токенов со сжатием
            tokens_saved         — токенов сэкономлено
            savings_pct          — процент экономии
            compression_count    — число выполненных суммаризаций
        """
        full_tokens = self._counter.count_messages(messages)["total"]
        compressed = self.get_compressed_messages(messages)
        compressed_tokens = self._counter.count_messages(compressed)["total"]

        saved = max(0, full_tokens - compressed_tokens)
        savings_pct = (saved / full_tokens * 100) if full_tokens > 0 else 0.0

        return {
            "full_messages_count": len(messages),
            "full_tokens": full_tokens,
            "compressed_messages_count": len(compressed),
            "compressed_tokens": compressed_tokens,
            "tokens_saved": saved,
            "savings_pct": round(savings_pct, 1),
            "compression_count": self.compression_count,
        }

    def format_stats(self) -> str:
        """Возвращает многострочную сводку по всем выполненным сжатиям."""
        if self.compression_count == 0:
            return "  Сжатий не выполнялось."
        lines = [
            f"  Выполнено сжатий:       {self.compression_count}",
            f"  Суммаризировано сообщ.: {self.summary_covers_up_to}",
            f"  Токенов сэкономлено:   ~{self.total_tokens_saved:,}",
            f"  Хранится summary:       {len(self.summaries)}",
        ]
        return "\n".join(lines)

    def format_summary_preview(self) -> str:
        """Возвращает превью последнего сгенерированного summary."""
        if not self.summaries:
            return "  Summary отсутствует."
        last = self.summaries[-1]
        preview = last[:300] + ("..." if len(last) > 300 else "")
        return f"  Последнее summary:\n    {preview}"

    # ─────────────────────────────────────────────────────────────
    # Сериализация
    # ─────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Сериализует состояние компрессора в словарь для хранения в JSON."""
        return {
            "summarize_every": self.summarize_every,
            "keep_last_n": self.keep_last_n,
            "model": self.model,
            "summaries": self.summaries,
            "summary_covers_up_to": self.summary_covers_up_to,
            "compression_count": self.compression_count,
            "total_tokens_saved": self.total_tokens_saved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextCompressor":
        """Восстанавливает состояние компрессора из словаря."""
        compressor = cls(
            summarize_every=data.get("summarize_every", DEFAULT_SUMMARIZE_EVERY),
            keep_last_n=data.get("keep_last_n", DEFAULT_KEEP_LAST_N),
            model=data.get("model", "zai-org/GLM-4.7-Flash"),
        )
        compressor.summaries = data.get("summaries", [])
        compressor.summary_covers_up_to = data.get("summary_covers_up_to", 0)
        compressor.compression_count = data.get("compression_count", 0)
        compressor.total_tokens_saved = data.get("total_tokens_saved", 0)
        return compressor

    # ─────────────────────────────────────────────────────────────
    # Приватные методы
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _chat_only(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Фильтрует только user+assistant сообщения (без system и summary)."""
        return [m for m in messages if m["role"] in ("user", "assistant")]

    def _call_llm_for_summary(
        self,
        client: Any,
        to_compress: List[Dict[str, Any]],
    ) -> str:
        """
        Вызывает LLM для генерации summary фрагмента истории.

        Использует низкую температуру (0.3) для стабильного, фактического
        изложения без лишней «творческой» вариативности.
        """
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in to_compress
        )
        summary_messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SUMMARY_USER_TEMPLATE.format(history=history_text),
            },
        ]
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": summary_messages,
            "max_completion_tokens": DEFAULT_SUMMARY_MAX_TOKENS,
            "temperature": 0.3,
        }
        if self.model in _MODELS_NO_TEMPERATURE:
            del params["temperature"]

        response = client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()
