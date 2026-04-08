"""
Тесты для news_agent.usage_tracker.

Покрывают:
  - RequestRecord: создание, сериализация
  - UsageTracker: запись, загрузка, агрегации, форматирование
  - RequestTimer: замер времени
  - _agg: внутренняя функция агрегации
"""

import json
import time
import pytest
from pathlib import Path
from datetime import date

from news_agent.usage_tracker import RequestRecord, UsageTracker, RequestTimer, _agg


class TestRequestRecord:
    def test_default_creation(self):
        rec = RequestRecord(command="chat", model="zai-org/GLM-4.7-Flash")
        assert rec.command == "chat"
        assert rec.model == "zai-org/GLM-4.7-Flash"
        assert rec.prompt_tokens == 0
        assert rec.completion_tokens == 0
        assert rec.total_tokens == 0
        assert rec.cost_usd == 0.0
        assert rec.success is True
        assert rec.error is None
        assert rec.tokens_estimated is False
        assert rec.session_id is None

    def test_total_tokens_calculated(self):
        rec = RequestRecord(command="api_simple", model="gpt-5.4",
                            prompt_tokens=100, completion_tokens=50)
        assert rec.total_tokens == 150

    def test_full_creation(self):
        rec = RequestRecord(
            command="api_advanced",
            model="gpt-5.4-mini",
            prompt_tokens=200,
            completion_tokens=80,
            cost_usd=0.00042,
            response_time_ms=1234.5,
            session_id="abc12345",
            success=True,
            tokens_estimated=True,
        )
        assert rec.cost_usd == 0.00042
        assert rec.response_time_ms == 1234.5
        assert rec.session_id == "abc12345"
        assert rec.tokens_estimated is True

    def test_failed_record(self):
        rec = RequestRecord(command="chat", model="gpt-5.4",
                            success=False, error="Timeout")
        assert rec.success is False
        assert rec.error == "Timeout"

    def test_timestamp_is_iso(self):
        rec = RequestRecord(command="generate", model="zai-org/GLM-4.7-Flash")
        from datetime import datetime
        parsed = datetime.fromisoformat(rec.timestamp)
        assert parsed is not None

    def test_to_dict_has_all_keys(self):
        rec = RequestRecord(command="chat", model="zai-org/GLM-4.7-Flash",
                            prompt_tokens=10, completion_tokens=20)
        d = rec.to_dict()
        for key in ("timestamp", "command", "model", "prompt_tokens",
                    "completion_tokens", "total_tokens", "cost_usd",
                    "response_time_ms", "session_id", "success", "error",
                    "tokens_estimated"):
            assert key in d

    def test_to_dict_total_tokens(self):
        rec = RequestRecord(command="chat", model="m", prompt_tokens=30, completion_tokens=70)
        assert rec.to_dict()["total_tokens"] == 100

    def test_from_dict_roundtrip(self):
        rec = RequestRecord(
            command="api_simple", model="gpt-5.4",
            prompt_tokens=100, completion_tokens=50,
            cost_usd=0.001, response_time_ms=500.0,
            session_id="xyz", success=True, tokens_estimated=True,
        )
        d = rec.to_dict()
        rec2 = RequestRecord.from_dict(d)
        assert rec2.command == rec.command
        assert rec2.model == rec.model
        assert rec2.prompt_tokens == rec.prompt_tokens
        assert rec2.completion_tokens == rec.completion_tokens
        assert rec2.total_tokens == rec.total_tokens
        assert rec2.cost_usd == rec.cost_usd
        assert rec2.session_id == rec.session_id
        assert rec2.tokens_estimated is True

    def test_from_dict_missing_fields_use_defaults(self):
        minimal = {"command": "chat", "model": "m"}
        rec = RequestRecord.from_dict(minimal)
        assert rec.prompt_tokens == 0
        assert rec.success is True
        assert rec.error is None


class TestUsageTrackerBasic:
    @pytest.fixture
    def tracker(self, tmp_path):
        return UsageTracker(logs_dir=str(tmp_path / "logs"))

    def test_creates_logs_directory(self, tmp_path):
        d = tmp_path / "new_logs"
        assert not d.exists()
        UsageTracker(logs_dir=str(d))
        assert d.exists()

    def test_empty_on_init(self, tracker):
        assert tracker.get_all_records() == []

    def test_record_adds_entry(self, tracker):
        tracker.record(command="chat", model="zai-org/GLM-4.7-Flash")
        assert len(tracker.get_all_records()) == 1

    def test_record_returns_request_record(self, tracker):
        rec = tracker.record(command="api_simple", model="gpt-5.4",
                             prompt_tokens=50, completion_tokens=30)
        assert isinstance(rec, RequestRecord)
        assert rec.total_tokens == 80

    def test_persists_to_file(self, tracker):
        tracker.record(command="chat", model="zai-org/GLM-4.7-Flash",
                       prompt_tokens=10, completion_tokens=20)
        assert tracker.stats_file.exists()
        with open(tracker.stats_file, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1

    def test_loads_existing_records(self, tmp_path):
        d = tmp_path / "logs"
        t1 = UsageTracker(logs_dir=str(d))
        t1.record(command="chat", model="zai-org/GLM-4.7-Flash",
                  prompt_tokens=100, completion_tokens=50)
        t1.record(command="api_simple", model="gpt-5-nano",
                  prompt_tokens=20, completion_tokens=10)

        t2 = UsageTracker(logs_dir=str(d))
        records = t2.get_all_records()
        assert len(records) == 2

    def test_corrupted_file_returns_empty(self, tmp_path):
        d = tmp_path / "logs"
        d.mkdir()
        (d / "usage_stats.json").write_text("not valid json", encoding="utf-8")
        t = UsageTracker(logs_dir=str(d))
        assert t.get_all_records() == []

    def test_multiple_records_accumulate(self, tracker):
        for i in range(5):
            tracker.record(command="chat", model="zai-org/GLM-4.7-Flash",
                           prompt_tokens=10, completion_tokens=5)
        assert len(tracker.get_all_records()) == 5


class TestUsageTrackerSummary:
    @pytest.fixture
    def filled_tracker(self, tmp_path):
        t = UsageTracker(logs_dir=str(tmp_path / "logs"))
        t.record(command="chat", model="zai-org/GLM-4.7-Flash",
                 prompt_tokens=100, completion_tokens=50, cost_usd=0.0,
                 response_time_ms=800)
        t.record(command="api_simple", model="gpt-5.4",
                 prompt_tokens=200, completion_tokens=100, cost_usd=0.002,
                 response_time_ms=1200)
        t.record(command="chat", model="zai-org/GLM-4.7-Flash",
                 prompt_tokens=50, completion_tokens=25, cost_usd=0.0,
                 success=False, error="Timeout")
        return t

    def test_empty_summary(self, tmp_path):
        t = UsageTracker(logs_dir=str(tmp_path / "logs"))
        s = t.get_summary()
        assert s["total_requests"] == 0
        assert s["total_tokens"] == 0
        assert s["total_cost_usd"] == 0.0
        assert s["first_request_at"] is None

    def test_total_requests(self, filled_tracker):
        s = filled_tracker.get_summary()
        assert s["total_requests"] == 3

    def test_successful_requests(self, filled_tracker):
        s = filled_tracker.get_summary()
        assert s["successful_requests"] == 2
        assert s["failed_requests"] == 1

    def test_total_tokens(self, filled_tracker):
        s = filled_tracker.get_summary()
        assert s["total_tokens"] == 525

    def test_total_cost(self, filled_tracker):
        s = filled_tracker.get_summary()
        assert abs(s["total_cost_usd"] - 0.002) < 1e-9

    def test_by_model_keys(self, filled_tracker):
        s = filled_tracker.get_summary()
        assert "zai-org/GLM-4.7-Flash" in s["by_model"]
        assert "gpt-5.4" in s["by_model"]

    def test_by_model_request_count(self, filled_tracker):
        s = filled_tracker.get_summary()
        assert s["by_model"]["zai-org/GLM-4.7-Flash"]["requests"] == 2
        assert s["by_model"]["gpt-5.4"]["requests"] == 1

    def test_by_command_keys(self, filled_tracker):
        s = filled_tracker.get_summary()
        assert "chat" in s["by_command"]
        assert "api_simple" in s["by_command"]

    def test_by_command_request_count(self, filled_tracker):
        s = filled_tracker.get_summary()
        assert s["by_command"]["chat"]["requests"] == 2
        assert s["by_command"]["api_simple"]["requests"] == 1

    def test_avg_response_time(self, filled_tracker):
        s = filled_tracker.get_summary()
        assert s["avg_response_time_ms"] == pytest.approx(1000.0, abs=1)

    def test_first_and_last_request(self, filled_tracker):
        s = filled_tracker.get_summary()
        assert s["first_request_at"] is not None
        assert s["last_request_at"] is not None

    def test_tokens_today(self, filled_tracker):
        s = filled_tracker.get_summary()
        assert s["tokens_today"] == s["total_tokens"]

    def test_requests_today(self, filled_tracker):
        s = filled_tracker.get_summary()
        assert s["requests_today"] == 3

    def test_by_model_contains_avg_response_time(self, filled_tracker):
        s = filled_tracker.get_summary()
        for model_data in s["by_model"].values():
            assert "avg_response_time_ms" in model_data

    def test_by_command_contains_avg_response_time(self, filled_tracker):
        s = filled_tracker.get_summary()
        for cmd_data in s["by_command"].values():
            assert "avg_response_time_ms" in cmd_data


class TestFormatSummary:
    @pytest.fixture
    def tracker_with_data(self, tmp_path):
        t = UsageTracker(logs_dir=str(tmp_path / "logs"))
        t.record(command="chat", model="zai-org/GLM-4.7-Flash",
                 prompt_tokens=100, completion_tokens=50, cost_usd=0.0)
        t.record(command="api_simple", model="gpt-5.4",
                 prompt_tokens=200, completion_tokens=100, cost_usd=0.002)
        return t

    def test_empty_returns_no_data_message(self, tmp_path):
        t = UsageTracker(logs_dir=str(tmp_path / "logs"))
        result = t.format_summary()
        assert "нет данных" in result.lower() or "нет" in result.lower()

    def test_format_summary_contains_stats_header(self, tracker_with_data):
        result = tracker_with_data.format_summary()
        assert "СТАТИСТИКА" in result

    def test_format_summary_contains_total_requests(self, tracker_with_data):
        result = tracker_with_data.format_summary()
        assert "2" in result

    def test_format_summary_contains_model_names(self, tracker_with_data):
        result = tracker_with_data.format_summary()
        assert "GLM-4.7-Flash" in result
        assert "gpt-5.4" in result

    def test_format_summary_contains_commands(self, tracker_with_data):
        result = tracker_with_data.format_summary()
        assert "chat" in result
        assert "api_simple" in result

    def test_format_summary_contains_cost(self, tracker_with_data):
        result = tracker_with_data.format_summary()
        assert "$" in result

    def test_format_summary_contains_tokens_section(self, tracker_with_data):
        result = tracker_with_data.format_summary()
        assert "prompt" in result.lower()
        assert "completion" in result.lower()


class TestRequestTimer:
    def test_measures_elapsed_time(self):
        with RequestTimer() as t:
            time.sleep(0.05)
        assert t.elapsed_ms >= 40.0
        assert t.elapsed_ms < 500.0

    def test_elapsed_is_float(self):
        with RequestTimer() as t:
            pass
        assert isinstance(t.elapsed_ms, float)

    def test_initial_elapsed_zero(self):
        t = RequestTimer()
        assert t.elapsed_ms == 0.0

    def test_returns_self_on_enter(self):
        t = RequestTimer()
        result = t.__enter__()
        t.__exit__(None, None, None)
        assert result is t


class TestAgg:
    def test_creates_new_key(self):
        group = {}
        rec = RequestRecord(command="chat", model="m",
                            prompt_tokens=10, completion_tokens=20, cost_usd=0.001,
                            response_time_ms=500)
        _agg(group, "test_key", rec)
        assert "test_key" in group
        assert group["test_key"]["requests"] == 1
        assert group["test_key"]["total_tokens"] == 30

    def test_accumulates_multiple(self):
        group = {}
        for _ in range(3):
            rec = RequestRecord(command="chat", model="m",
                                prompt_tokens=10, completion_tokens=10)
            _agg(group, "key", rec)
        assert group["key"]["requests"] == 3
        assert group["key"]["total_tokens"] == 60

    def test_cost_accumulated(self):
        group = {}
        for _ in range(2):
            rec = RequestRecord(command="chat", model="m", cost_usd=0.001)
            _agg(group, "key", rec)
        assert abs(group["key"]["cost_usd"] - 0.002) < 1e-9


class TestUsageTrackerIntegration:
    def test_full_workflow(self, tmp_path):
        t = UsageTracker(logs_dir=str(tmp_path / "logs"))

        cmds = [
            ("chat", "zai-org/GLM-4.7-Flash", 100, 50, 0.0),
            ("api_simple", "gpt-5.4", 200, 80, 0.0008),
            ("api_advanced", "gpt-5.4-mini", 150, 60, 0.0003),
            ("generate", "zai-org/GLM-4.7-Flash", 500, 300, 0.0),
        ]
        for cmd, model, pt, ct, cost in cmds:
            t.record(command=cmd, model=model,
                     prompt_tokens=pt, completion_tokens=ct,
                     cost_usd=cost, response_time_ms=1000.0)

        s = t.get_summary()
        assert s["total_requests"] == 4
        assert s["total_tokens"] == (150 + 280 + 210 + 800)
        assert len(s["by_model"]) == 3
        assert len(s["by_command"]) == 4

        text = t.format_summary()
        assert "api_simple" in text
        assert "api_advanced" in text
        assert "generate" in text
