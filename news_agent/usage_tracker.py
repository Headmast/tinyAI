"""
UsageTracker — накопительный журнал всех запросов к API.

Хранит каждый запрос в logs/usage_stats.json и предоставляет
агрегированную статистику за всё время работы.

Структура записи (RequestRecord):
    timestamp       — ISO время запроса
    command         — откуда пришёл запрос (chat/api_simple/api_advanced/generate/agent/batch)
    model           — модель, использованная в запросе
    prompt_tokens   — токены промпта (реальные или оценочные)
    completion_tokens — токены ответа
    total_tokens    — сумма
    cost_usd        — стоимость в долларах
    response_time_ms — время ответа (мс)
    session_id      — ID диалоговой сессии (если применимо)
    success         — True если запрос выполнен успешно
    error           — текст ошибки (если success=False)
    tokens_estimated — True если токены посчитаны приближённо (4 chars = 1 token)

Агрегации (UsageTracker.get_summary):
    total_requests          — всего запросов
    successful_requests     — успешных
    total_tokens            — всего токенов
    total_cost_usd          — всего потрачено
    by_model                — статистика по каждой модели
    by_command              — статистика по типу команды
    avg_response_time_ms    — среднее время ответа
    first_request_at        — первый запрос
    last_request_at         — последний запрос
    tokens_today            — токены за сегодня
    cost_today_usd          — стоимость за сегодня
"""

import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    ) -> None:
        self.timestamp: str = datetime.now().isoformat()
        self.command: str = command
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "command": self.command,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "response_time_ms": self.response_time_ms,
            "session_id": self.session_id,
            "success": self.success,
            "error": self.error,
            "tokens_estimated": self.tokens_estimated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RequestRecord":
        rec = cls.__new__(cls)
        rec.timestamp = d.get("timestamp", "")
        rec.command = d.get("command", "unknown")
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
    ) -> RequestRecord:
        """
        Записывает один API-запрос в журнал и сохраняет на диск.

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
        )
        self._records.append(rec)
        self._save()
        return rec

    def get_all_records(self) -> List[RequestRecord]:
        """Возвращает все записи (копия)."""
        return list(self._records)

    def get_summary(self) -> Dict[str, Any]:
        """
        Возвращает полную агрегированную статистику за всё время.

        Ключи:
            total_requests, successful_requests, failed_requests
            total_prompt_tokens, total_completion_tokens, total_tokens
            total_cost_usd
            by_model        — {model: {requests, total_tokens, cost_usd, ...}}
            by_command      — {command: {requests, total_tokens, cost_usd}}
            avg_response_time_ms
            first_request_at, last_request_at
            tokens_today, cost_today_usd, requests_today
        """
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

        by_model: Dict[str, Dict[str, Any]] = {}
        by_command: Dict[str, Dict[str, Any]] = {}

        for r in records:
            _agg(by_model, r.model, r)
            _agg(by_command, r.command, r)

        for group in (by_model, by_command):
            for key in group:
                rts = group[key].get("_rts", [])
                group[key]["avg_response_time_ms"] = (
                    round(sum(rts) / len(rts), 1) if rts else 0.0
                )
                del group[key]["_rts"]

        today_records = [r for r in records if r.timestamp[:10] == today_str]
        tokens_today = sum(r.total_tokens for r in today_records)
        cost_today = sum(r.cost_usd for r in today_records)
        requests_today = len(today_records)

        return {
            "total_requests": total_requests,
            "successful_requests": successful,
            "failed_requests": total_requests - successful,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "by_model": by_model,
            "by_command": by_command,
            "avg_response_time_ms": round(avg_rt, 1),
            "first_request_at": records[0].timestamp if records else None,
            "last_request_at": records[-1].timestamp if records else None,
            "tokens_today": tokens_today,
            "cost_today_usd": round(cost_today, 6),
            "requests_today": requests_today,
        }

    def format_summary(self) -> str:
        """Возвращает красиво отформатированную строку со статистикой."""
        s = self.get_summary()

        if s["total_requests"] == 0:
            return "  Пока нет данных — выполните хотя бы один запрос."

        lines: List[str] = []
        SEP = "─" * 58

        lines += [
            SEP,
            "  СТАТИСТИКА ИСПОЛЬЗОВАНИЯ API",
            SEP,
        ]

        if s["first_request_at"]:
            first = s["first_request_at"][:16].replace("T", " ")
            last = s["last_request_at"][:16].replace("T", " ")
            lines.append(f"  Период:          {first}  →  {last}")

        lines += [
            f"  Всего запросов:  {s['total_requests']:,}  "
            f"(успешных: {s['successful_requests']}, "
            f"ошибок: {s['failed_requests']})",
            f"  Сегодня:         {s['requests_today']} запросов",
            SEP,
            f"  Токены (всего):",
            f"    prompt:        {s['total_prompt_tokens']:>12,}",
            f"    completion:    {s['total_completion_tokens']:>12,}",
            f"    total:         {s['total_tokens']:>12,}",
            f"  Токены сегодня:  {s['tokens_today']:,}",
            SEP,
            f"  Стоимость всего: ${s['total_cost_usd']:.6f}",
            f"  Стоимость сегодня: ${s['cost_today_usd']:.6f}",
            f"  Ср. время ответа: {s['avg_response_time_ms']:.0f} мс",
            SEP,
        ]

        if s["by_model"]:
            lines.append("  ПО МОДЕЛЯМ:")
            for model, m in sorted(s["by_model"].items(), key=lambda x: -x[1]["requests"]):
                short = model.split("/")[-1]
                lines.append(
                    f"    {short:20}  {m['requests']:>4} запросов  "
                    f"{m['total_tokens']:>8,} токенов  ${m['cost_usd']:.4f}"
                )
            lines.append(SEP)

        if s["by_command"]:
            lines.append("  ПО КОМАНДАМ:")
            for cmd, c in sorted(s["by_command"].items(), key=lambda x: -x[1]["requests"]):
                lines.append(
                    f"    {cmd:18}  {c['requests']:>4} запросов  "
                    f"{c['total_tokens']:>8,} токенов  ${c['cost_usd']:.4f}"
                )
            lines.append(SEP)

        return "\n".join(lines)


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
