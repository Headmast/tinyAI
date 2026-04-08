"""
Тесты для news_agent/token_counter.py.

Покрывают:
  - TokenCounter: подсчёт текста, сообщений, ответов, полного breakdown
  - DialogTokenTracker: рост токенов, таблица, overflow, стоимость
  - TokenBudget: effective_max_completion, check_overflow, format_budget_line
  - Fallback-поведение при отсутствии tiktoken
"""

import pytest
from unittest.mock import patch

from news_agent.token_counter import (
    TokenCounter,
    DialogTokenTracker,
    TurnSnapshot,
    TokenBudget,
)


# ─────────────────────────────────────────────────────────────
# TokenCounter — базовые свойства
# ─────────────────────────────────────────────────────────────

class TestTokenCounterProperties:

    def test_is_exact_bool(self):
        counter = TokenCounter()
        assert isinstance(counter.is_exact, bool)

    def test_method_label_non_empty(self):
        counter = TokenCounter()
        assert counter.method_label in ("tiktoken", "~chars÷4")

    def test_model_stored(self):
        counter = TokenCounter(model="gpt-5.4")
        assert counter.model == "gpt-5.4"

    def test_default_model(self):
        counter = TokenCounter()
        assert counter.model == "zai-org/GLM-4.7-Flash"


# ─────────────────────────────────────────────────────────────
# TokenCounter.count_text
# ─────────────────────────────────────────────────────────────

class TestCountText:

    def test_empty_string_returns_zero(self):
        counter = TokenCounter()
        assert counter.count_text("") == 0

    def test_positive_for_nonempty(self):
        counter = TokenCounter()
        assert counter.count_text("Hello, world!") > 0

    def test_longer_text_more_tokens(self):
        counter = TokenCounter()
        short = counter.count_text("Hi")
        long_ = counter.count_text("Hi " * 100)
        assert long_ > short

    def test_returns_int(self):
        counter = TokenCounter()
        result = counter.count_text("test text")
        assert isinstance(result, int)

    def test_cyrillic_text(self):
        counter = TokenCounter()
        result = counter.count_text("Привет, мир!")
        assert result > 0

    def test_consistent_results(self):
        counter = TokenCounter()
        text = "Repeated call consistency test"
        assert counter.count_text(text) == counter.count_text(text)


# ─────────────────────────────────────────────────────────────
# TokenCounter.count_messages
# ─────────────────────────────────────────────────────────────

class TestCountMessages:

    def test_empty_messages(self):
        counter = TokenCounter()
        result = counter.count_messages([])
        assert result["total"] >= 0
        assert result["message_count"] == 0

    def test_keys_present(self):
        counter = TokenCounter()
        msgs = [{"role": "user", "content": "Hello"}]
        result = counter.count_messages(msgs)
        assert "total" in result
        assert "content_tokens" in result
        assert "overhead" in result
        assert "per_message" in result
        assert "message_count" in result
        assert "is_exact" in result

    def test_message_count_correct(self):
        counter = TokenCounter()
        msgs = [
            {"role": "system",    "content": "You are helpful."},
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = counter.count_messages(msgs)
        assert result["message_count"] == 3

    def test_per_message_length_matches(self):
        counter = TokenCounter()
        msgs = [
            {"role": "user",      "content": "Short"},
            {"role": "assistant", "content": "A much longer response text here."},
        ]
        result = counter.count_messages(msgs)
        assert len(result["per_message"]) == 2

    def test_total_includes_overhead(self):
        counter = TokenCounter()
        msgs = [{"role": "user", "content": "Hello"}]
        result = counter.count_messages(msgs)
        assert result["total"] >= result["content_tokens"]
        assert result["overhead"] > 0

    def test_more_messages_more_tokens(self):
        counter = TokenCounter()
        msgs1 = [{"role": "user", "content": "Hello"}]
        msgs2 = msgs1 + [
            {"role": "assistant", "content": "Hi"},
            {"role": "user",      "content": "How are you?"},
        ]
        r1 = counter.count_messages(msgs1)
        r2 = counter.count_messages(msgs2)
        assert r2["total"] > r1["total"]

    def test_is_exact_consistent(self):
        counter = TokenCounter()
        msgs = [{"role": "user", "content": "test"}]
        result = counter.count_messages(msgs)
        assert result["is_exact"] == counter.is_exact


# ─────────────────────────────────────────────────────────────
# TokenCounter.count_request_breakdown
# ─────────────────────────────────────────────────────────────

class TestCountRequestBreakdown:

    def _make_messages(self):
        return [
            {"role": "system",    "content": "System prompt."},
            {"role": "user",      "content": "User question here."},
        ]

    def test_keys_present(self):
        counter = TokenCounter()
        result = counter.count_request_breakdown(self._make_messages(), "Response text.")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                    "per_message", "history_tokens", "current_msg_tokens", "is_exact"):
            assert key in result

    def test_total_is_sum(self):
        counter = TokenCounter()
        result = counter.count_request_breakdown(self._make_messages(), "Response.")
        assert result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"]

    def test_no_response_gives_zero_completion(self):
        counter = TokenCounter()
        result = counter.count_request_breakdown(self._make_messages(), "")
        assert result["completion_tokens"] == 0

    def test_longer_response_more_completion_tokens(self):
        counter = TokenCounter()
        short = counter.count_request_breakdown(self._make_messages(), "Short.")
        long_ = counter.count_request_breakdown(self._make_messages(), "Long " * 100)
        assert long_["completion_tokens"] > short["completion_tokens"]

    def test_history_tokens_less_than_prompt(self):
        counter = TokenCounter()
        msgs = self._make_messages()
        result = counter.count_request_breakdown(msgs, "resp")
        assert result["history_tokens"] <= result["prompt_tokens"]


# ─────────────────────────────────────────────────────────────
# TokenCounter — fallback (no tiktoken)
# ─────────────────────────────────────────────────────────────

class TestTokenCounterFallback:

    def test_fallback_is_exact_false(self):
        counter = TokenCounter()
        with patch.object(counter, "_enc", None):
            assert not counter.is_exact
            assert counter.method_label == "~chars÷4"

    def test_fallback_count_text(self):
        counter = TokenCounter()
        with patch.object(counter, "_enc", None):
            text = "Hello world test"
            result = counter.count_text(text)
            assert result == max(1, len(text) // 4)

    def test_fallback_count_messages(self):
        counter = TokenCounter()
        with patch.object(counter, "_enc", None):
            msgs = [{"role": "user", "content": "Hello world"}]
            result = counter.count_messages(msgs)
            assert result["total"] > 0
            assert not result["is_exact"]

    def test_fallback_empty_string(self):
        counter = TokenCounter()
        with patch.object(counter, "_enc", None):
            assert counter.count_text("") == 0


# ─────────────────────────────────────────────────────────────
# DialogTokenTracker
# ─────────────────────────────────────────────────────────────

class TestDialogTokenTracker:

    def _make_tracker(self, max_tokens: int = 128_000) -> DialogTokenTracker:
        return DialogTokenTracker(model="gpt-5-nano", max_tokens=max_tokens)

    def _messages_after_n_turns(self, n: int):
        base = [
            ("user",      "Hello, can you help me?"),
            ("assistant", "Of course! What do you need?"),
        ]
        msgs = []
        for i in range(n):
            role, text = base[i % len(base)]
            msgs.append({"role": role, "content": text})
        return msgs

    def test_initial_state(self):
        tracker = self._make_tracker()
        assert tracker.current_tokens == 0
        assert tracker.fill_percentage == 0.0
        assert not tracker.is_overflowed()
        assert tracker.snapshots == []

    def test_add_turn_returns_snapshot(self):
        tracker = self._make_tracker()
        msgs = [{"role": "user", "content": "Hi"}]
        snap = tracker.add_turn("user", "Hi", msgs)
        assert isinstance(snap, TurnSnapshot)
        assert snap.turn == 1
        assert snap.role == "user"

    def test_tokens_grow_with_turns(self):
        tracker = self._make_tracker()
        messages = []
        prev = 0
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            text = f"Message number {i} with some extra words."
            messages.append({"role": role, "content": text})
            snap = tracker.add_turn(role, text, messages)
            assert snap.total_tokens >= prev
            prev = snap.total_tokens

    def test_fill_percentage_increases(self):
        tracker = self._make_tracker(max_tokens=10_000)
        messages = []
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            text = "This is message number " + str(i) + " " * 20
            messages.append({"role": role, "content": text})
            tracker.add_turn(role, text, messages)
        assert tracker.fill_percentage > 0

    def test_is_overflowed_false_normal(self):
        tracker = self._make_tracker(max_tokens=128_000)
        messages = [{"role": "user", "content": "Hi"}]
        tracker.add_turn("user", "Hi", messages)
        assert not tracker.is_overflowed()

    def test_is_overflowed_true_tiny_limit(self):
        tracker = self._make_tracker(max_tokens=5)
        long_text = "This is a much longer text that surely exceeds five tokens."
        messages = [{"role": "user", "content": long_text}]
        tracker.add_turn("user", long_text, messages)
        assert tracker.is_overflowed()

    def test_remaining_tokens_decreases(self):
        tracker = self._make_tracker(max_tokens=10_000)
        messages = []
        prev_remaining = tracker.remaining_tokens
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            text = "Some message here " * 5
            messages.append({"role": role, "content": text})
            tracker.add_turn(role, text, messages)
        assert tracker.remaining_tokens < prev_remaining

    def test_format_growth_table_empty(self):
        tracker = self._make_tracker()
        output = tracker.format_growth_table()
        assert "нет данных" in output.lower() or "(нет" in output

    def test_format_growth_table_non_empty(self):
        tracker = self._make_tracker()
        messages = [{"role": "user", "content": "Hello"}]
        tracker.add_turn("user", "Hello", messages)
        output = tracker.format_growth_table()
        assert "Turn" in output or "turn" in output.lower() or "1" in output

    def test_format_context_bar_contains_percent(self):
        tracker = self._make_tracker()
        messages = [{"role": "user", "content": "Test"}]
        tracker.add_turn("user", "Test", messages)
        bar = tracker.format_context_bar()
        assert "%" in bar

    def test_context_bar_icons(self):
        tracker_green  = self._make_tracker(max_tokens=1_000_000)
        tracker_red    = self._make_tracker(max_tokens=10)

        msgs = [{"role": "user", "content": "Short text."}]
        tracker_green.add_turn("user", "Short text.", msgs)
        tracker_red.add_turn("user", "Short text.", msgs)

        assert "🟢" in tracker_green.format_context_bar()
        assert "🔴" in tracker_red.format_context_bar()

    def test_turn_numbers_sequential(self):
        tracker = self._make_tracker()
        messages = []
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            text = f"Message {i}"
            messages.append({"role": role, "content": text})
            snap = tracker.add_turn(role, text, messages)
            assert snap.turn == i + 1

    def test_cost_at_turn_free_model(self):
        tracker = DialogTokenTracker(model="zai-org/GLM-4.7-Flash")
        messages = [{"role": "user", "content": "Hi"}]
        tracker.add_turn("user", "Hi", messages)
        cost = tracker.cost_at_turn(1, 0.0, 0.0)
        assert cost == 0.0

    def test_cost_at_turn_paid_model(self):
        tracker = self._make_tracker()
        messages = [{"role": "user", "content": "Hello, this is a test message."}]
        tracker.add_turn("user", "Hello, this is a test message.", messages)
        cost = tracker.cost_at_turn(1, 0.01, 0.03)
        assert cost >= 0

    def test_cumulative_cost_non_negative(self):
        tracker = self._make_tracker()
        messages = []
        for i in range(3):
            role = "user" if i % 2 == 0 else "assistant"
            text = f"Turn {i} message"
            messages.append({"role": role, "content": text})
            tracker.add_turn(role, text, messages)
        cost = tracker.cumulative_cost(0.002, 0.01)
        assert cost >= 0

    def test_cost_at_invalid_turn(self):
        tracker = self._make_tracker()
        assert tracker.cost_at_turn(0, 0.01, 0.03) == 0.0
        assert tracker.cost_at_turn(99, 0.01, 0.03) == 0.0


# ─────────────────────────────────────────────────────────────
# TokenBudget
# ─────────────────────────────────────────────────────────────

class TestTokenBudget:

    def test_effective_max_completion_normal(self):
        budget = TokenBudget(context_limit=128_000, max_completion=4_000)
        eff = budget.effective_max_completion(1_000)
        assert eff == 4_000

    def test_effective_max_completion_constrained(self):
        budget = TokenBudget(context_limit=128_000, max_completion=10_000)
        eff = budget.effective_max_completion(125_000)
        assert eff == 3_000

    def test_effective_max_completion_overflow(self):
        budget = TokenBudget(context_limit=128_000, max_completion=4_000)
        eff = budget.effective_max_completion(130_000)
        assert eff == 0

    def test_check_overflow_false_normal(self):
        budget = TokenBudget(context_limit=128_000, max_completion=4_000)
        is_ov, msg = budget.check_overflow(1_000)
        assert not is_ov
        assert msg == ""

    def test_check_overflow_true_exceeded(self):
        budget = TokenBudget(context_limit=128_000, max_completion=4_000)
        is_ov, msg = budget.check_overflow(130_000)
        assert is_ov
        assert len(msg) > 0
        assert "OVERFLOW" in msg.upper() or "overflow" in msg.lower()

    def test_check_overflow_warning_near_limit(self):
        budget = TokenBudget(context_limit=128_000, max_completion=4_000, reserved_ratio=0.25)
        is_ov, msg = budget.check_overflow(110_000)
        assert not is_ov
        assert len(msg) > 0

    def test_format_budget_line_contains_tokens(self):
        budget = TokenBudget(context_limit=128_000, max_completion=4_000)
        line = budget.format_budget_line(5_000)
        assert "5,000" in line or "5000" in line
        assert "128,000" in line or "128000" in line

    def test_format_budget_line_returns_string(self):
        budget = TokenBudget(context_limit=128_000, max_completion=4_000)
        result = budget.format_budget_line(1_000)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_effective_respects_max_completion(self):
        budget = TokenBudget(context_limit=128_000, max_completion=100)
        eff = budget.effective_max_completion(1_000)
        assert eff == 100

    def test_effective_respects_remaining(self):
        budget = TokenBudget(context_limit=1_000, max_completion=500)
        eff = budget.effective_max_completion(800)
        assert eff == 200


# ─────────────────────────────────────────────────────────────
# Интеграционные тесты
# ─────────────────────────────────────────────────────────────

class TestIntegration:

    def test_counter_and_tracker_consistent(self):
        model = "gpt-5-nano"
        counter = TokenCounter(model=model)
        tracker = DialogTokenTracker(model=model, max_tokens=128_000)

        messages = [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": "What is 2 + 2?"},
        ]
        snap = tracker.add_turn("user", "What is 2 + 2?", messages)

        counter_total = counter.count_messages(messages)["total"]
        assert abs(snap.total_tokens - counter_total) <= 5

    def test_budget_tracks_dialog_growth(self):
        budget = TokenBudget(context_limit=500, max_completion=100)
        counter = TokenCounter()

        messages = []
        for i in range(8):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message turn {i}" * 3})

        total = counter.count_messages(messages)["total"]
        eff = budget.effective_max_completion(total)

        if total >= 500:
            assert eff == 0
        else:
            assert eff <= 100

    def test_full_request_pipeline(self):
        counter = TokenCounter(model="gpt-5-nano")
        messages = [
            {"role": "system",    "content": "Be concise."},
            {"role": "user",      "content": "Explain quantum computing briefly."},
        ]
        response = (
            "Quantum computing uses quantum mechanical phenomena like "
            "superposition and entanglement to process information."
        )
        breakdown = counter.count_request_breakdown(messages, response)

        assert breakdown["prompt_tokens"] > 0
        assert breakdown["completion_tokens"] > 0
        assert breakdown["total_tokens"] == (
            breakdown["prompt_tokens"] + breakdown["completion_tokens"]
        )

    def test_tracker_overflow_detection_with_budget(self):
        tiny_limit = 50
        tracker = DialogTokenTracker(model="gpt-5-nano", max_tokens=tiny_limit)
        budget = TokenBudget(context_limit=tiny_limit, max_completion=20)

        messages = []
        long_text = "This message is deliberately long enough to overflow the tiny limit. " * 3
        messages.append({"role": "user", "content": long_text})
        snap = tracker.add_turn("user", long_text, messages)

        if snap.total_tokens > tiny_limit:
            is_ov, _ = budget.check_overflow(snap.total_tokens)
            assert is_ov
