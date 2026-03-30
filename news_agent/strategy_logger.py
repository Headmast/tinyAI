"""
strategy_logger — логирование метрик стратегий управления контекстом.

Каждый turn записывается как JSON-строка в .jsonl файл.
Поддерживает:
  - посменное логирование (turn-by-turn)
  - итоговую статистику сессии
  - чтение и анализ логов
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class StrategyLogger:
    """
    Логирует каждый turn диалога в JSONL-файл с метриками стратегии.

    Формат файла: logs/strategy_<name>_<timestamp>.jsonl
    Каждая строка — JSON-объект с данными одного turn.
    """

    def __init__(
        self,
        strategy_name: str,
        model: str,
        log_dir: str = "logs",
        session_id: str = "",
    ) -> None:
        self.strategy_name = strategy_name
        self.model = model
        self.session_id = session_id
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = self._log_dir / f"strategy_{strategy_name}_{ts}.jsonl"

        self._turn_number: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_extraction_tokens: int = 0
        self._start_time: float = time.monotonic()
        self._turns: List[Dict[str, Any]] = []

    @property
    def log_file(self) -> Path:
        return self._log_file

    def log_turn(
        self,
        user_message: str,
        assistant_response: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        messages_sent_to_api: int = 0,
        messages_total_in_session: int = 0,
        response_time_ms: float = 0.0,
        strategy_stats: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Логирует один turn диалога."""
        self._turn_number += 1
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        extraction_tokens = 0
        if strategy_stats:
            extraction_tokens = strategy_stats.get("last_extraction_tokens", 0)
        self._total_extraction_tokens += extraction_tokens

        entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "strategy": self.strategy_name,
            "model": self.model,
            "turn_number": self._turn_number,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "metrics": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "extraction_tokens": extraction_tokens,
                "messages_sent_to_api": messages_sent_to_api,
                "messages_total_in_session": messages_total_in_session,
                "messages_dropped": messages_total_in_session - messages_sent_to_api,
                "response_time_ms": round(response_time_ms, 1),
            },
            "strategy_stats": strategy_stats or {},
        }
        if extra:
            entry["extra"] = extra

        self._turns.append(entry)

        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_summary(self) -> Dict[str, Any]:
        """Возвращает итоговую статистику по всем turns."""
        elapsed = time.monotonic() - self._start_time

        response_times = [t["metrics"]["response_time_ms"] for t in self._turns]
        avg_response = sum(response_times) / len(response_times) if response_times else 0

        return {
            "strategy": self.strategy_name,
            "model": self.model,
            "session_id": self.session_id,
            "total_turns": self._turn_number,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_extraction_tokens": self._total_extraction_tokens,
            "total_all_tokens": (
                self._total_prompt_tokens
                + self._total_completion_tokens
                + self._total_extraction_tokens
            ),
            "avg_response_time_ms": round(avg_response, 1),
            "total_elapsed_s": round(elapsed, 1),
            "log_file": str(self._log_file),
        }

    def save_summary(self) -> Path:
        """Сохраняет итоговую сводку в отдельный JSON-файл."""
        summary = self.get_summary()
        summary_file = self._log_file.with_suffix(".summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary_file


def load_log(log_file: str) -> List[Dict[str, Any]]:
    """Читает JSONL-лог и возвращает список turn-записей."""
    entries = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compare_logs(log_files: List[str]) -> Dict[str, Any]:
    """
    Сравнивает логи нескольких стратегий.
    Возвращает сводную таблицу метрик.
    """
    results: Dict[str, Any] = {}

    for lf in log_files:
        entries = load_log(lf)
        if not entries:
            continue

        strategy = entries[0].get("strategy", "unknown")
        total_prompt = sum(e["metrics"]["prompt_tokens"] for e in entries)
        total_completion = sum(e["metrics"]["completion_tokens"] for e in entries)
        total_extraction = sum(e["metrics"].get("extraction_tokens", 0) for e in entries)
        total_all = total_prompt + total_completion + total_extraction
        avg_response = (
            sum(e["metrics"]["response_time_ms"] for e in entries) / len(entries)
        )

        results[strategy] = {
            "turns": len(entries),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_extraction_tokens": total_extraction,
            "total_all_tokens": total_all,
            "avg_response_time_ms": round(avg_response, 1),
            "avg_prompt_tokens_per_turn": round(total_prompt / len(entries)),
            "avg_completion_tokens_per_turn": round(total_completion / len(entries)),
        }

    return results
