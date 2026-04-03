"""
TokenCounter — точный подсчёт токенов через tiktoken с fallback на эвристику.

Ключевые концепции:
    - tiktoken (OpenAI) считает точно для GPT-семейства
    - Для GLM/других моделей используется cl100k_base как наилучшее приближение
    - При отсутствии tiktoken — fallback: 1 токен ≈ 4 символа
    - Для chat completions добавляется overhead на каждое сообщение (~4 токена)

Классы:
    TokenCounter        — подсчёт токенов текста и массива messages
    DialogTokenTracker  — отслеживание роста токенов по ходу диалога
    TokenBudget         — управление бюджетом контекстного окна
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


_TIKTOKEN_AVAILABLE = False
_tiktoken_module = None

try:
    import tiktoken as _tiktoken_module  # type: ignore
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    pass


_FALLBACK_CHARS_PER_TOKEN = 4

_MODEL_ENCODING_MAP: Dict[str, str] = {
    "gpt-5-nano":     "cl100k_base",
    "gpt-5.4":        "cl100k_base",
    "gpt-5.4-mini":   "cl100k_base",
    "zai-org/GLM-4.7-Flash": "cl100k_base",
    "zai-org/GLM-4.7":       "cl100k_base",
}

MODELS_NO_TEMPERATURE: frozenset = frozenset({
    "gpt-5-nano", "o1", "o1-mini", "o3", "o3-mini", "o4-mini",
})

_TOKENS_PER_MESSAGE = 4
_TOKENS_PER_REPLY_PRIMER = 3


def _get_encoding(model: str):
    """Возвращает tiktoken-кодировщик для модели или None при недоступности."""
    if not _TIKTOKEN_AVAILABLE or _tiktoken_module is None:
        return None
    encoding_name = _MODEL_ENCODING_MAP.get(model, "cl100k_base")
    try:
        return _tiktoken_module.get_encoding(encoding_name)
    except Exception:
        try:
            return _tiktoken_module.get_encoding("cl100k_base")
        except Exception:
            return None


class TokenCounter:
    """
    Точный счётчик токенов с автоматическим fallback.

    При наличии tiktoken использует его; при отсутствии — эвристику
    (1 токен ≈ 4 символа). Флаг is_exact показывает, какой режим активен.

    Пример:
        counter = TokenCounter(model="gpt-5.4")
        n = counter.count_text("Hello, world!")            # 4 токена
        d = counter.count_messages(messages)               # dict с деталями
        print(counter.is_exact)                            # True если tiktoken есть
    """

    def __init__(self, model: str = "zai-org/GLM-4.7-Flash") -> None:
        self.model = model
        self._enc = _get_encoding(model)

    @property
    def is_exact(self) -> bool:
        """True если используется tiktoken (точный подсчёт)."""
        return self._enc is not None

    @property
    def method_label(self) -> str:
        return "tiktoken" if self.is_exact else "~chars÷4"

    def count_text(self, text: str) -> int:
        """Считает токены в произвольном тексте."""
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text))
        return max(1, len(text) // _FALLBACK_CHARS_PER_TOKEN)

    def count_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Считает токены для массива messages (chat completions format).

        Учитывает overhead каждого сообщения (~4 токена) и
        primer ответа ассистента (~3 токена).

        Возвращает dict:
            total          — общее число токенов (prompt + overhead)
            content_tokens — токены только контента сообщений
            overhead       — токены на структуру (роли, разделители)
            per_message    — список int, токены каждого сообщения (без overhead)
            message_count  — число сообщений
            is_exact       — True если использован tiktoken
        """
        per_message: List[int] = []
        content_total = 0

        for msg in messages:
            content = msg.get("content") or ""
            role = msg.get("role", "")
            n = self.count_text(content) + self.count_text(role)
            per_message.append(n)
            content_total += n

        message_overhead = _TOKENS_PER_MESSAGE * len(messages)
        reply_primer = _TOKENS_PER_REPLY_PRIMER
        overhead = message_overhead + reply_primer

        return {
            "total": content_total + overhead,
            "content_tokens": content_total,
            "overhead": overhead,
            "per_message": per_message,
            "message_count": len(messages),
            "is_exact": self.is_exact,
        }

    def count_response(self, text: str) -> int:
        """Считает токены в ответе ассистента (completion tokens)."""
        return self.count_text(text)

    def count_request_breakdown(
        self,
        messages: List[Dict[str, Any]],
        response_text: str = "",
    ) -> Dict[str, Any]:
        """
        Полная разбивка токенов для одного запроса.

        Возвращает dict:
            prompt_tokens      — токены промпта (история + текущий запрос)
            completion_tokens  — токены ответа
            total_tokens       — сумма
            per_message        — токены каждого сообщения
            history_tokens     — токены всех сообщений кроме последнего
            current_msg_tokens — токены последнего (текущего) сообщения
            is_exact           — True если использован tiktoken
        """
        msg_info = self.count_messages(messages)
        prompt_tokens = msg_info["total"]
        completion_tokens = self.count_response(response_text)

        per_msg = msg_info["per_message"]
        history_tokens = sum(per_msg[:-1]) if len(per_msg) > 1 else 0
        current_msg_tokens = per_msg[-1] if per_msg else 0

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "per_message": per_msg,
            "history_tokens": history_tokens,
            "current_msg_tokens": current_msg_tokens,
            "overhead": msg_info["overhead"],
            "is_exact": self.is_exact,
        }


@dataclass
class TurnSnapshot:
    """Снимок состояния токенов после одного хода диалога."""
    turn: int
    role: str
    text_preview: str
    message_tokens: int
    history_tokens: int
    total_tokens: int
    growth: int
    growth_pct: float


@dataclass
class DialogTokenTracker:
    """
    Отслеживает рост токенов по мере диалога.

    Ведёт историю снимков после каждого сообщения и позволяет
    наглядно показать, как стоимость растёт с длиной диалога.

    Пример:
        tracker = DialogTokenTracker(model="gpt-5.4", max_tokens=8000)
        tracker.add_turn("user", "Привет!", messages_so_far)
        tracker.add_turn("assistant", "Привет!", messages_so_far)
        print(tracker.format_growth_table())
    """

    model: str = "zai-org/GLM-4.7-Flash"
    max_tokens: int = 128_000
    snapshots: List[TurnSnapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._counter = TokenCounter(model=self.model)

    def add_turn(
        self,
        role: str,
        text: str,
        messages: List[Dict[str, Any]],
    ) -> TurnSnapshot:
        """
        Записывает снимок после добавления нового сообщения в историю.

        Args:
            role:     "user" | "assistant" | "system"
            text:     текст нового сообщения
            messages: полный массив messages включая новое сообщение
        """
        info = self._counter.count_messages(messages)
        total = info["total"]
        msg_tokens = self._counter.count_text(text)

        prev_total = self.snapshots[-1].total_tokens if self.snapshots else 0
        growth = total - prev_total
        growth_pct = (total / self.max_tokens * 100) if self.max_tokens > 0 else 0.0

        snap = TurnSnapshot(
            turn=len(self.snapshots) + 1,
            role=role,
            text_preview=text[:60].replace("\n", " "),
            message_tokens=msg_tokens,
            history_tokens=total - msg_tokens,
            total_tokens=total,
            growth=growth,
            growth_pct=growth_pct,
        )
        self.snapshots.append(snap)
        return snap

    @property
    def current_tokens(self) -> int:
        """Текущее число токенов в диалоге."""
        return self.snapshots[-1].total_tokens if self.snapshots else 0

    @property
    def remaining_tokens(self) -> int:
        """Свободное место в контекстном окне."""
        return max(0, self.max_tokens - self.current_tokens)

    @property
    def fill_percentage(self) -> float:
        """Процент заполнения контекстного окна (0–100)."""
        if self.max_tokens == 0:
            return 0.0
        return min(100.0, self.current_tokens / self.max_tokens * 100)

    def is_overflowed(self) -> bool:
        """True если токенов больше, чем позволяет контекстное окно."""
        return self.current_tokens > self.max_tokens

    def format_growth_table(self) -> str:
        """
        Возвращает таблицу роста токенов по ходу диалога.

        Пример:
            Turn  Role       Msg tok  +Δ    Total    Window%
            1     user       12       +19   19       0.01%
            2     assistant  45       +52   71       0.06%
        """
        if not self.snapshots:
            return "  (нет данных)"

        lines: List[str] = []
        header = f"  {'Turn':>4}  {'Role':9}  {'MsgTok':>7}  {'Δ':>6}  {'Total':>7}  {'Window':>7}"
        sep = "  " + "─" * (len(header) - 2)
        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        for s in self.snapshots:
            overflow_mark = " ⚠️" if s.total_tokens > self.max_tokens else ""
            lines.append(
                f"  {s.turn:>4}  {s.role:9}  {s.message_tokens:>7,}  "
                f"{'+' if s.growth >= 0 else ''}{s.growth:>5,}  "
                f"{s.total_tokens:>7,}  {s.growth_pct:>6.2f}%{overflow_mark}"
            )

        lines.append(sep)
        lines.append(
            f"  Итого: {self.current_tokens:,} / {self.max_tokens:,} токенов  "
            f"({self.fill_percentage:.2f}%)  "
            f"{'[OVERFLOW]' if self.is_overflowed() else f'осталось: {self.remaining_tokens:,}'}"
        )
        return "\n".join(lines)

    def format_context_bar(self) -> str:
        """Визуальный индикатор заполнения контекста."""
        pct = self.fill_percentage
        bar_len = 24
        filled = min(int(bar_len * pct / 100), bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        if pct >= 90:
            icon = "🔴"
        elif pct >= 70:
            icon = "🟡"
        else:
            icon = "🟢"

        return (
            f"{icon} [{bar}] {pct:.1f}%  "
            f"{self.current_tokens:,} / {self.max_tokens:,} токенов"
        )

    def cost_at_turn(self, turn: int, prompt_price: float, completion_price: float) -> float:
        """
        Оценивает стоимость в USD на конкретном ходу диалога.

        Args:
            turn:             номер хода (1-based)
            prompt_price:     цена за 1K prompt-токенов (USD)
            completion_price: цена за 1K completion-токенов (USD)
        """
        if turn < 1 or turn > len(self.snapshots):
            return 0.0
        snap = self.snapshots[turn - 1]
        prompt_cost = snap.history_tokens * prompt_price / 1000
        completion_cost = snap.message_tokens * completion_price / 1000
        return round(prompt_cost + completion_cost, 8)

    def cumulative_cost(self, prompt_price: float, completion_price: float) -> float:
        """Суммарная накопленная стоимость за все ходы диалога."""
        return sum(
            self.cost_at_turn(i + 1, prompt_price, completion_price)
            for i in range(len(self.snapshots))
        )


@dataclass
class TokenBudget:
    """
    Управляет бюджетом токенов для одного запроса.

    Позволяет:
        - Проверить, поместится ли история в контекстное окно
        - Получить максимально допустимое число completion-токенов
        - Предупредить о приближении к лимиту

    Args:
        context_limit:   максимальный размер контекстного окна модели
        max_completion:  желаемый лимит для ответа модели
        reserved_ratio:  доля окна, зарезервированная для ответа (0–1)
    """

    context_limit: int = 128_000
    max_completion: int = 4_000
    reserved_ratio: float = 0.25

    def effective_max_completion(self, prompt_tokens: int) -> int:
        """
        Возвращает безопасный лимит completion-токенов с учётом prompt.

        Если prompt уже занял много окна, completion ограничивается
        оставшимся местом.
        """
        remaining = self.context_limit - prompt_tokens
        if remaining <= 0:
            return 0
        return min(self.max_completion, remaining)

    def check_overflow(self, prompt_tokens: int) -> Tuple[bool, str]:
        """
        Проверяет, не превышает ли prompt контекстное окно.

        Returns:
            (is_overflow, message)
        """
        reserved = int(self.context_limit * self.reserved_ratio)
        if prompt_tokens >= self.context_limit:
            return True, (
                f"OVERFLOW: prompt {prompt_tokens:,} ≥ лимит {self.context_limit:,} токенов. "
                f"Запрос завершится ошибкой."
            )
        if prompt_tokens >= self.context_limit - reserved:
            pct = prompt_tokens / self.context_limit * 100
            return False, (
                f"WARNING: prompt {prompt_tokens:,} токенов ({pct:.1f}% окна). "
                f"Для ответа осталось лишь {self.context_limit - prompt_tokens:,} токенов."
            )
        return False, ""

    def format_budget_line(self, prompt_tokens: int) -> str:
        """Однострочное резюме бюджета токенов."""
        eff = self.effective_max_completion(prompt_tokens)
        used_pct = prompt_tokens / self.context_limit * 100 if self.context_limit > 0 else 0
        return (
            f"prompt: {prompt_tokens:,} ({used_pct:.1f}%) | "
            f"max_completion: {eff:,} | "
            f"context: {self.context_limit:,}"
        )
