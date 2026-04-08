"""
Тесты для расширенной аналитики UsageTracker.

Покрывают: source-категоризацию, фильтрацию по периоду,
суммарные расходы, форматированные отчёты, AgentLoop.tracker,
CLI parse_period_shorthand.
"""

import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from news_agent.usage_tracker import (
    RequestRecord,
    UsageTracker,
    _command_to_source,
    _parse_date,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_tracker(tmp_path: Path) -> UsageTracker:
    return UsageTracker(logs_dir=str(tmp_path))


def _rec(tracker: UsageTracker, command: str, cost: float = 0.0,
         ts_override: Optional[str] = None, **kwargs) -> RequestRecord:
    rec = tracker.record(command=command, model="test-model", cost_usd=cost, **kwargs)
    if ts_override:
        rec.timestamp = ts_override
        tracker._save()
    return rec


# ─── _command_to_source ───────────────────────────────────────────────────────

class TestCommandToSource:

    def test_agent(self):
        assert _command_to_source("agent") == "agent"

    def test_chat_is_session(self):
        assert _command_to_source("chat") == "session"

    def test_api_simple(self):
        assert _command_to_source("api_simple") == "api"

    def test_api_advanced(self):
        assert _command_to_source("api_advanced") == "api"

    def test_generate(self):
        assert _command_to_source("generate") == "generate"

    def test_batch(self):
        assert _command_to_source("batch") == "generate"

    def test_unknown_returns_other(self):
        assert _command_to_source("unknown_cmd") == "other"


# ─── RequestRecord.source ─────────────────────────────────────────────────────

class TestRequestRecordSource:

    def test_source_set_on_init(self, tmp_path):
        t = _make_tracker(tmp_path)
        rec = _rec(t, "chat")
        assert rec.source == "session"

    def test_source_in_to_dict(self, tmp_path):
        t = _make_tracker(tmp_path)
        rec = _rec(t, "agent")
        assert rec.to_dict()["source"] == "agent"

    def test_source_from_dict_legacy(self):
        d = {"command": "chat", "model": "m", "prompt_tokens": 0,
             "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0,
             "response_time_ms": 0.0, "session_id": None,
             "success": True, "error": None, "tokens_estimated": False}
        rec = RequestRecord.from_dict(d)
        assert rec.source == "session"

    def test_source_from_dict_explicit(self):
        d = {"command": "chat", "source": "api", "model": "m",
             "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
             "cost_usd": 0.0, "response_time_ms": 0.0, "session_id": None,
             "success": True, "error": None, "tokens_estimated": False}
        rec = RequestRecord.from_dict(d)
        assert rec.source == "api"

    def test_iterations_stored(self, tmp_path):
        t = _make_tracker(tmp_path)
        rec = t.record(command="agent", model="m", iterations=5)
        assert rec.iterations == 5
        assert rec.to_dict()["iterations"] == 5

    def test_iterations_roundtrip(self, tmp_path):
        t = _make_tracker(tmp_path)
        t.record(command="agent", model="m", iterations=7)
        t2 = _make_tracker(tmp_path)
        assert t2.get_all_records()[0].iterations == 7


# ─── get_records_for_period ───────────────────────────────────────────────────

class TestPeriodFiltering:

    def _tracker_with_dates(self, tmp_path):
        t = _make_tracker(tmp_path)
        today = date.today()
        dates = [
            str(today - timedelta(days=10)),
            str(today - timedelta(days=5)),
            str(today - timedelta(days=3)),
            str(today),
        ]
        for i, d in enumerate(dates):
            _rec(t, "chat", cost=float(i + 1), ts_override=f"{d}T10:00:00")
        return t, dates

    def test_no_filter_returns_all(self, tmp_path):
        t, _ = self._tracker_with_dates(tmp_path)
        assert len(t.get_records_for_period()) == 4

    def test_start_filter(self, tmp_path):
        t, dates = self._tracker_with_dates(tmp_path)
        recs = t.get_records_for_period(start=dates[2])
        assert len(recs) == 2

    def test_end_filter(self, tmp_path):
        t, dates = self._tracker_with_dates(tmp_path)
        recs = t.get_records_for_period(end=dates[1])
        assert len(recs) == 2

    def test_both_filters(self, tmp_path):
        t, dates = self._tracker_with_dates(tmp_path)
        recs = t.get_records_for_period(start=dates[1], end=dates[2])
        assert len(recs) == 2

    def test_single_day(self, tmp_path):
        t, dates = self._tracker_with_dates(tmp_path)
        recs = t.get_records_for_period(start=dates[0], end=dates[0])
        assert len(recs) == 1

    def test_no_records_in_range(self, tmp_path):
        t, dates = self._tracker_with_dates(tmp_path)
        future = str(date.today() + timedelta(days=30))
        assert t.get_records_for_period(start=future) == []

    def test_get_cost_for_period(self, tmp_path):
        t, dates = self._tracker_with_dates(tmp_path)
        cost = t.get_cost_for_period(start=dates[3], end=dates[3])
        assert cost == pytest.approx(4.0)

    def test_get_tokens_for_period(self, tmp_path):
        t = _make_tracker(tmp_path)
        today = str(date.today())
        t.record(command="chat", model="m", prompt_tokens=100, completion_tokens=50)
        tokens = t.get_tokens_for_period(start=today)
        assert tokens["prompt"] == 100
        assert tokens["completion"] == 50
        assert tokens["total"] == 150


# ─── parse_period_shorthand ───────────────────────────────────────────────────

class TestParsePeriodShorthand:

    def test_today(self):
        start, end = UsageTracker.parse_period_shorthand("today")
        assert start == date.today()
        assert end == date.today()

    def test_week(self):
        start, end = UsageTracker.parse_period_shorthand("week")
        assert end == date.today()
        assert (end - start).days == 6

    def test_month(self):
        start, end = UsageTracker.parse_period_shorthand("month")
        assert start.day == 1
        assert end == date.today()

    def test_single_date(self):
        start, end = UsageTracker.parse_period_shorthand("2026-01-15")
        assert start == date(2026, 1, 15)
        assert end == date(2026, 1, 15)

    def test_date_range(self):
        start, end = UsageTracker.parse_period_shorthand("2026-01-01", "2026-01-31")
        assert start == date(2026, 1, 1)
        assert end == date(2026, 1, 31)

    def test_invalid_returns_none_start(self):
        start, end = UsageTracker.parse_period_shorthand("garbage_input")
        assert start is None
        assert end == date.today()


# ─── get_summary by_source ────────────────────────────────────────────────────

class TestSummaryBySource:

    def test_by_source_in_summary(self, tmp_path):
        t = _make_tracker(tmp_path)
        t.record(command="chat", model="m", prompt_tokens=10)
        t.record(command="agent", model="m", prompt_tokens=20)
        t.record(command="api_simple", model="m", prompt_tokens=5)
        s = t.get_summary()
        assert "by_source" in s
        assert "session" in s["by_source"]
        assert "agent" in s["by_source"]
        assert "api" in s["by_source"]

    def test_by_source_token_counts(self, tmp_path):
        t = _make_tracker(tmp_path)
        t.record(command="chat", model="m", prompt_tokens=100, completion_tokens=50)
        t.record(command="chat", model="m", prompt_tokens=200, completion_tokens=80)
        s = t.get_summary()
        assert s["by_source"]["session"]["prompt_tokens"] == 300
        assert s["by_source"]["session"]["completion_tokens"] == 130

    def test_get_summary_for_period(self, tmp_path):
        t = _make_tracker(tmp_path)
        today = str(date.today())
        yesterday = str(date.today() - timedelta(days=1))
        _rec(t, "chat", cost=1.0, ts_override=f"{yesterday}T10:00:00")
        _rec(t, "agent", cost=2.0, ts_override=f"{today}T10:00:00")
        s = t.get_summary_for_period(start=today)
        assert s["total_requests"] == 1
        assert s["total_cost_usd"] == pytest.approx(2.0)


# ─── format_period_report ─────────────────────────────────────────────────────

class TestFormatPeriodReport:

    def test_empty_period_returns_no_data_message(self, tmp_path):
        t = _make_tracker(tmp_path)
        result = t.format_period_report(start="2000-01-01", end="2000-01-02")
        assert "Нет данных" in result

    def test_returns_string_with_data(self, tmp_path):
        t = _make_tracker(tmp_path)
        t.record(command="chat", model="m", prompt_tokens=50, completion_tokens=25)
        result = t.format_period_report()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_source_section(self, tmp_path):
        t = _make_tracker(tmp_path)
        t.record(command="chat", model="m")
        result = t.format_period_report()
        assert "ИСТОЧНИКАМ" in result or "session" in result

    def test_today_label(self, tmp_path):
        t = _make_tracker(tmp_path)
        today = str(date.today())
        t.record(command="api_simple", model="m")
        result = t.format_period_report(start=today, end=today)
        assert today in result

    def test_custom_label(self, tmp_path):
        t = _make_tracker(tmp_path)
        t.record(command="chat", model="m")
        result = t.format_period_report(label="МОЙ ОТЧЁТ")
        assert "МОЙ ОТЧЁТ" in result

    def test_format_summary_all_time(self, tmp_path):
        t = _make_tracker(tmp_path)
        t.record(command="agent", model="m", iterations=3)
        result = t.format_summary()
        assert "всё время" in result.lower() or "СТАТИСТИКА" in result


# ─── AgentLoop.tracker integration ───────────────────────────────────────────

class TestAgentTrackerIntegration:

    def test_tracker_param_accepted(self, tmp_path):
        from news_agent.agent import AgentLoop
        from openai import OpenAI
        client = MagicMock(spec=OpenAI)
        t = _make_tracker(tmp_path)
        agent = AgentLoop(client=client, model="test", tracker=t)
        assert agent.tracker is t

    def test_tracker_none_by_default(self):
        from news_agent.agent import AgentLoop
        from openai import OpenAI
        client = MagicMock(spec=OpenAI)
        agent = AgentLoop(client=client, model="test")
        assert agent.tracker is None

    def test_record_to_tracker_called_on_success(self, tmp_path):
        from news_agent.agent import AgentLoop
        t = _make_tracker(tmp_path)
        agent = AgentLoop.__new__(AgentLoop)
        agent.tracker = t
        agent.model = "test-model"
        agent._counter = MagicMock()
        agent._counter.method_label = "tiktoken"
        result = {
            "final_post": "Some post content",
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "token_counter_method": "tiktoken",
            "iterations": 3,
            "elapsed_ms": 1200.0,
        }
        agent._record_to_tracker(result)
        assert len(t.get_all_records()) == 1
        rec = t.get_all_records()[0]
        assert rec.command == "agent"
        assert rec.prompt_tokens == 100
        assert rec.completion_tokens == 50
        assert rec.iterations == 3
        assert rec.success is True

    def test_record_to_tracker_none_does_nothing(self, tmp_path):
        from news_agent.agent import AgentLoop
        agent = AgentLoop.__new__(AgentLoop)
        agent.tracker = None
        result = {
            "final_post": "Post",
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "token_counter_method": "tiktoken",
            "iterations": 1,
            "elapsed_ms": 500.0,
        }
        agent._record_to_tracker(result)

    def test_failed_result_recorded_as_failure(self, tmp_path):
        from news_agent.agent import AgentLoop
        t = _make_tracker(tmp_path)
        agent = AgentLoop.__new__(AgentLoop)
        agent.tracker = t
        agent.model = "test-model"
        result = {
            "final_post": "Агент не создал финальный пост",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "token_counter_method": "~chars÷4",
            "iterations": 12,
            "elapsed_ms": 9000.0,
        }
        agent._record_to_tracker(result)
        rec = t.get_all_records()[0]
        assert rec.success is False
        assert rec.tokens_estimated is True
