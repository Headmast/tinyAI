"""
UsageTracker — накопительный журнал всех запросов к API.

Единый JSON-журнал (logs/usage_stats.json) для агента, чат-сессий и прямых API-вызовов.

Структура записи (RequestRecord):
    timestamp         — ISO время запроса
    command           — оригинальная команда (chat/api_simple/api_advanced/agent/generate/batch)
    source            — категория источника: agent | session | api | generate
    model             — модель, использованная в запросе
    prompt_tokens     — токены промпта (реальные или оценочные)
    completion_tokens — токены ответа
    total_tokens      — сумма
    cost_usd          — стоимость в долларах
    response_time_ms  — время ответа (мс)
    session_id        — ID диалоговой сессии (если применимо)
    success           — True если запрос выполнен успешно
    error             — текст ошибки (если success=False)
    tokens_estimated  — True если токены посчитаны приближённо
    iterations        — число итераций (для agent-команд)

Агрегации:
    get_summary()                          — всё время
    get_summary_for_period(start, end)     — конкретный диапазон дат
    get_cost_for_period(start, end)        — суммарная стоимость за период
    format_period_report(start, end, lbl)  — форматированный отчёт за период

BY_SOURCE breakdowns: agent / session / api / generate

Period shorthands (для CLI):
    today  week  month  YYYY-MM-DD  YYYY-MM-DD YYYY-MM-DD
"""

import json
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


_SOURCE_MAP: Dict[str, str] = {
    "agent":        "agent",
    "chat":         "session",
    "api_simple":   "api",
    "api_advanced": "api",
    "generate":     "generate",
    "batch":        "generate",
}


def _command_to_source(command: str) -> str:
    return _SOURCE_MAP.get(command, "other")


def _parse_date(d: Union[str, date, None]) -> Optional[date]:
    """Разбирает дату из строки YYYY-MM-DD или объекта date."""
    if d is None:
        return None
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


_EMPTY_SUMMARY_BLOCK: Dict[str, Any] = {
    "requests": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "cost_usd": 0.0,
    "avg_response_time_ms": 0.0,
}


class RequestRecord:
    """Запись об одном API-запросе."""

    __slots__ = (
        "timestamp",
        "command",
        "source",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
        "response_time_ms",
        "session_id",
        "success",
        "error",
        "tokens_estimated",
        "iterations",
    )

    def __init__(
        self,
        command: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        response_time_ms: float = 0.0,
        session_id: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        tokens_estimated: bool = False,
        iterations: Optional[int] = None,
    ) -> None:
        self.timestamp: str = datetime.now().isoformat()
        self.command: str = command
        self.source: str = _command_to_source(command)
        self.model: str = model
        self.prompt_tokens: int = prompt_tokens
        self.completion_tokens: int = completion_tokens
        self.total_tokens: int = prompt_tokens + completion_tokens
        self.cost_usd: float = cost_usd
        self.response_time_ms: float = round(response_time_ms, 1)
        self.session_id: Optional[str] = session_id
        self.success: bool = success
        self.error: Optional[str] = error
        self.tokens_estimated: bool = tokens_estimated
        self.iterations: Optional[int] = iterations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp":         self.timestamp,
            "command":           self.command,
            "source":            self.source,
            "model":             self.model,
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
            "cost_usd":          self.cost_usd,
            "response_time_ms":  self.response_time_ms,
            "session_id":        self.session_id,
            "success":           self.success,
            "error":             self.error,
            "tokens_estimated":  self.tokens_estimated,
            "iterations":        self.iterations,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RequestRecord":
        rec = cls.__new__(cls)
        rec.timestamp = d.get("timestamp", "")
        rec.command = d.get("command", "unknown")
        rec.source = d.get("source") or _command_to_source(rec.command)
        rec.model = d.get("model", "unknown")
        rec.prompt_tokens = d.get("prompt_tokens", 0)
        rec.completion_tokens = d.get("completion_tokens", 0)
        rec.total_tokens = d.get("total_tokens", rec.prompt_tokens + rec.completion_tokens)
        rec.cost_usd = d.get("cost_usd", 0.0)
        rec.response_time_ms = d.get("response_time_ms", 0.0)
        rec.session_id = d.get("session_id")
        rec.success = d.get("success", True)
        rec.error = d.get("error")
        rec.tokens_estimated = d.get("tokens_estimated", False)
        rec.iterations = d.get("iterations")
        return rec


class UsageTracker:
    """
    Накопительный журнал запросов к API.

    Данные хранятся в logs/usage_stats.json и не сбрасываются между сессиями —
    накапливаются за всё время работы приложения.

    Пример:
        tracker = UsageTracker()
        rec = tracker.record(command="chat", model="zai-org/GLM-4.7-Flash",
                             prompt_tokens=150, completion_tokens=80,
                             cost_usd=0.0, response_time_ms=1200)
        print(tracker.format_summary())
    """

    STATS_FILE = "usage_stats.json"

    def __init__(self, logs_dir: str = "logs") -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.stats_file = self.logs_dir / self.STATS_FILE
        self._records: List[RequestRecord] = self._load()

    def _load(self) -> List[RequestRecord]:
        if not self.stats_file.exists():
            return []
        try:
            with open(self.stats_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [RequestRecord.from_dict(d) for d in data]
        except (json.JSONDecodeError, TypeError, KeyError):
            return []

    def _save(self) -> None:
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in self._records],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def record(
        self,
        command: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        response_time_ms: float = 0.0,
        session_id: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        tokens_estimated: bool = False,
        iterations: Optional[int] = None,
    ) -> RequestRecord:
        """
        Записывает один API-запрос в журнал и сохраняет на диск.

        Параметры:
            command           — тип операции (agent/chat/api_simple/api_advanced/generate/batch)
            model             — имя модели
            iterations        — число итераций (для агента)
        Returns:
            RequestRecord — созданная запись
        """
        rec = RequestRecord(
            command=command,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            response_time_ms=response_time_ms,
            session_id=session_id,
            success=success,
            error=error,
            tokens_estimated=tokens_estimated,
            iterations=iterations,
        )
        self._records.append(rec)
        self._save()
        return rec

    def get_all_records(self) -> List[RequestRecord]:
        """Возвращает все записи (копия)."""
        return list(self._records)

    def get_records_for_period(
        self,
        start: Union[str, date, None] = None,
        end: Union[str, date, None] = None,
    ) -> List[RequestRecord]:
        """
        Возвращает записи за указанный период (включительно).

        start / end — YYYY-MM-DD строка или объект date.
        Если start=None — с самого начала.
        Если end=None   — по сегодняшний день включительно.
        """
        s = _parse_date(start)
        e = _parse_date(end) or date.today()
        result = []
        for r in self._records:
            rec_date = _parse_date(r.timestamp[:10])
            if rec_date is None:
                continue
            if s and rec_date < s:
                continue
            if rec_date > e:
                continue
            result.append(r)
        return result

    def get_cost_for_period(
        self,
        start: Union[str, date, None] = None,
        end: Union[str, date, None] = None,
    ) -> float:
        """Суммарная стоимость за период (USD)."""
        return round(sum(r.cost_usd for r in self.get_records_for_period(start, end)), 8)

    def get_tokens_for_period(
        self,
        start: Union[str, date, None] = None,
        end: Union[str, date, None] = None,
    ) -> Dict[str, int]:
        """Суммарные токены за период."""
        recs = self.get_records_for_period(start, end)
        return {
            "prompt":     sum(r.prompt_tokens     for r in recs),
            "completion": sum(r.completion_tokens for r in recs),
            "total":      sum(r.total_tokens      for r in recs),
        }

    @staticmethod
    def parse_period_shorthand(
        token: str,
        token2: Optional[str] = None,
    ) -> Tuple[Optional[date], date]:
        """
        Разбирает строковые сокращения периода.

        Варианты:
            "today"                     — только сегодня
            "week"                      — последние 7 дней
            "month"                     — текущий календарный месяц
            "YYYY-MM-DD"                — один конкретный день
            "YYYY-MM-DD" + "YYYY-MM-DD" — диапазон дат

        Returns:
            (start_date, end_date)
        """
        today = date.today()
        t = (token or "").strip().lower()

        if t == "today":
            return today, today
        if t == "week":
            return today - timedelta(days=6), today
        if t == "month":
            return today.replace(day=1), today
        if token2:
            return _parse_date(token), _parse_date(token2) or today
        try:
            d = _parse_date(token)
            return d, d
        except (ValueError, TypeError):
            return None, today

    def get_summary(
        self,
        records: Optional[List[RequestRecord]] = None,
    ) -> Dict[str, Any]:
        """
        Возвращает агрегированную статистику.

        records — список для агрегации; если None — все записи.

        Ключи результата:
            total_requests, successful_requests, failed_requests
            total_prompt_tokens, total_completion_tokens, total_tokens
            total_cost_usd
            by_model    — {model: {requests, total_tokens, cost_usd, ...}}
            by_command  — {command: ...}
            by_source   — {agent|session|api|generate: ...}
            avg_response_time_ms
            first_request_at, last_request_at
            tokens_today, cost_today_usd, requests_today
        """
        if records is None:
            records = self._records

        today_str = date.today().isoformat()

        total_requests = len(records)
        successful = sum(1 for r in records if r.success)
        total_prompt = sum(r.prompt_tokens for r in records)
        total_completion = sum(r.completion_tokens for r in records)
        total_tokens = sum(r.total_tokens for r in records)
        total_cost = sum(r.cost_usd for r in records)

        resp_times = [r.response_time_ms for r in records if r.response_time_ms > 0]
        avg_rt = sum(resp_times) / len(resp_times) if resp_times else 0.0

        by_model:   Dict[str, Dict[str, Any]] = {}
        by_command: Dict[str, Dict[str, Any]] = {}
        by_source:  Dict[str, Dict[str, Any]] = {}

        for r in records:
            _agg(by_model,   r.model,   r)
            _agg(by_command, r.command, r)
            _agg(by_source,  r.source,  r)

        for group in (by_model, by_command, by_source):
            for key in group:
                rts = group[key].pop("_rts", [])
                group[key]["avg_response_time_ms"] = (
                    round(sum(rts) / len(rts), 1) if rts else 0.0
                )

        today_records = [r for r in records if r.timestamp[:10] == today_str]
        tokens_today = sum(r.total_tokens for r in today_records)
        cost_today = sum(r.cost_usd for r in today_records)
        requests_today = len(today_records)

        return {
            "total_requests":       total_requests,
            "successful_requests":  successful,
            "failed_requests":      total_requests - successful,
            "total_prompt_tokens":  total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens":         total_tokens,
            "total_cost_usd":       round(total_cost, 6),
            "by_model":             by_model,
            "by_command":           by_command,
            "by_source":            by_source,
            "avg_response_time_ms": round(avg_rt, 1),
            "first_request_at":     records[0].timestamp if records else None,
            "last_request_at":      records[-1].timestamp if records else None,
            "tokens_today":         tokens_today,
            "cost_today_usd":       round(cost_today, 6),
            "requests_today":       requests_today,
        }

    def get_summary_for_period(
        self,
        start: Union[str, date, None] = None,
        end: Union[str, date, None] = None,
    ) -> Dict[str, Any]:
        """Агрегированная статистика за конкретный период."""
        return self.get_summary(records=self.get_records_for_period(start, end))

    @staticmethod
    def _format_block(
        s: Dict[str, Any],
        title: str,
        show_today: bool = True,
    ) -> str:
        """Форматирует один сводный блок статистики."""
        lines: List[str] = []
        SEP = "─" * 62

        lines += [SEP, f"  {title}", SEP]

        if s["first_request_at"]:
            first = s["first_request_at"][:16].replace("T", " ")
            last  = s["last_request_at"][:16].replace("T", " ")
            lines.append(f"  Период:            {first}  →  {last}")

        lines += [
            f"  Запросов:          {s['total_requests']:,}  "
            f"(✅ {s['successful_requests']}  ❌ {s['failed_requests']})",
        ]
        if show_today:
            lines.append(f"  Сегодня:           {s['requests_today']} запросов")

        lines += [
            SEP,
            f"  Токены (prompt):   {s['total_prompt_tokens']:>12,}",
            f"  Токены (completion): {s['total_completion_tokens']:>10,}",
            f"  Токены (total):    {s['total_tokens']:>12,}",
        ]
        if show_today:
            lines.append(f"  Токены сегодня:    {s['tokens_today']:,}")

        lines += [
            SEP,
            f"  Стоимость:         ${s['total_cost_usd']:.6f}",
        ]
        if show_today:
            lines.append(f"  Стоимость сегодня: ${s['cost_today_usd']:.6f}")
        lines.append(f"  Ср. время ответа:  {s['avg_response_time_ms']:.0f} мс")
        lines.append(SEP)

        if s.get("by_source"):
            lines.append("  ПО ИСТОЧНИКАМ:")
            source_order = ["agent", "session", "api", "generate", "other"]
            src_items = sorted(
                s["by_source"].items(),
                key=lambda x: (source_order.index(x[0]) if x[0] in source_order else 99),
            )
            for src, v in src_items:
                label = {"agent": "🤖 Агент", "session": "💬 Сессии",
                         "api": "🔌 API", "generate": "📝 Генерация"}.get(src, src)
                lines.append(
                    f"    {label:16}  {v['requests']:>4} запросов  "
                    f"{v['total_tokens']:>8,} токенов  ${v['cost_usd']:.6f}"
                )
            lines.append(SEP)

        if s.get("by_model"):
            lines.append("  ПО МОДЕЛЯМ:")
            for model, m in sorted(s["by_model"].items(), key=lambda x: -x[1]["requests"]):
                short = model.split("/")[-1]
                lines.append(
                    f"    {short:22}  {m['requests']:>4} зап.  "
                    f"{m['total_tokens']:>8,} tok  ${m['cost_usd']:.6f}"
                )
            lines.append(SEP)

        if s.get("by_command"):
            lines.append("  ПО КОМАНДАМ:")
            for cmd, c in sorted(s["by_command"].items(), key=lambda x: -x[1]["requests"]):
                lines.append(
                    f"    {cmd:20}  {c['requests']:>4} зап.  "
                    f"{c['total_tokens']:>8,} tok  ${c['cost_usd']:.6f}"
                )
            lines.append(SEP)

        return "\n".join(lines)

    def format_summary(self) -> str:
        """Статистика за всё время."""
        s = self.get_summary()
        if s["total_requests"] == 0:
            return "  Пока нет данных — выполните хотя бы один запрос."
        return self._format_block(s, "СТАТИСТИКА ИСПОЛЬЗОВАНИЯ API  (всё время)", show_today=True)

    def format_period_report(
        self,
        start: Union[str, date, None] = None,
        end: Union[str, date, None] = None,
        label: Optional[str] = None,
    ) -> str:
        """
        Форматированный отчёт за период.

        start / end  — YYYY-MM-DD строка или date.
        label        — заголовок отчёта (генерируется автоматически если None).
        """
        recs = self.get_records_for_period(start, end)
        if not recs:
            s_str = str(start) if start else "начало"
            e_str = str(end)   if end   else str(date.today())
            return f"  Нет данных за период {s_str} — {e_str}."

        s = self.get_summary(records=recs)

        if label is None:
            s_str = str(_parse_date(start)) if start else recs[0].timestamp[:10]
            e_str = str(_parse_date(end))   if end   else recs[-1].timestamp[:10]
            if s_str == e_str:
                label = f"ОТЧЁТ ЗА {s_str}"
            else:
                label = f"ОТЧЁТ ЗА {s_str}  →  {e_str}"

        return self._format_block(s, label, show_today=False)


def _agg(group: Dict, key: str, r: RequestRecord) -> None:
    """Вспомогательная функция агрегации по ключу."""
    if key not in group:
        group[key] = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "_rts": [],
        }
    g = group[key]
    g["requests"] += 1
    g["prompt_tokens"] += r.prompt_tokens
    g["completion_tokens"] += r.completion_tokens
    g["total_tokens"] += r.total_tokens
    g["cost_usd"] = round(g["cost_usd"] + r.cost_usd, 8)
    if r.response_time_ms > 0:
        g["_rts"].append(r.response_time_ms)


class RequestTimer:
    """
    Контекстный менеджер для измерения времени запроса.

    Использование:
        with RequestTimer() as t:
            ...сделать запрос...
        elapsed_ms = t.elapsed_ms
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "RequestTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = round((time.monotonic() - self._start) * 1000, 1)
