"""
Тесты для модуля news_agent.session_manager.

Покрывают:
  - ConversationSession: создание, добавление сообщений, оценка токенов,
    индикатор контекста, сериализация/десериализация
  - SessionStorage: сохранение, загрузка, список, удаление, индекс
  - Интеграционные: полный цикл сессии (create → chat → close → reload)
"""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

from news_agent.session_manager import (
    ConversationSession,
    SessionStorage,
    MODEL_CONTEXT_SIZES,
    DEFAULT_CONTEXT_SIZE,
    CONTEXT_WARN_THRESHOLD,
)


class TestModelContextSizes:
    def test_all_known_models_have_context(self):
        for model in [
            "zai-org/GLM-4.7-Flash",
            "zai-org/GLM-4.7",
            "gpt-5-nano",
            "gpt-5.4",
            "gpt-5.4-mini",
        ]:
            assert model in MODEL_CONTEXT_SIZES
            assert MODEL_CONTEXT_SIZES[model] > 0

    def test_default_context_size_positive(self):
        assert DEFAULT_CONTEXT_SIZE > 0

    def test_warn_threshold_valid_range(self):
        assert 0.0 < CONTEXT_WARN_THRESHOLD < 1.0


class TestConversationSessionCreation:
    def test_default_creation(self):
        s = ConversationSession()
        assert s.session_id
        assert len(s.session_id) == 8
        assert s.status == "active"
        assert s.messages == []
        assert s.token_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_custom_name(self):
        s = ConversationSession(name="Тестовая сессия")
        assert s.name == "Тестовая сессия"

    def test_system_prompt_added_to_messages(self):
        s = ConversationSession(system_prompt="Ты помощник.")
        assert len(s.messages) == 1
        assert s.messages[0]["role"] == "system"
        assert s.messages[0]["content"] == "Ты помощник."

    def test_no_system_prompt_empty_messages(self):
        s = ConversationSession()
        assert len(s.messages) == 0

    def test_model_assigned(self):
        s = ConversationSession(model="gpt-5.4")
        assert s.model == "gpt-5.4"

    def test_max_context_tokens_known_model(self):
        s = ConversationSession(model="zai-org/GLM-4.7-Flash")
        assert s.max_context_tokens == MODEL_CONTEXT_SIZES["zai-org/GLM-4.7-Flash"]

    def test_max_context_tokens_unknown_model(self):
        s = ConversationSession(model="unknown-model-xyz")
        assert s.max_context_tokens == DEFAULT_CONTEXT_SIZE

    def test_repr_contains_key_info(self):
        s = ConversationSession(name="Test")
        r = repr(s)
        assert "ConversationSession" in r
        assert s.session_id in r
        assert "active" in r


class TestConversationSessionMessages:
    def test_add_user_message(self):
        s = ConversationSession()
        s.add_user_message("Привет!")
        assert len(s.messages) == 1
        assert s.messages[0] == {"role": "user", "content": "Привет!"}

    def test_add_assistant_message(self):
        s = ConversationSession()
        s.add_assistant_message("Здравствуй!")
        assert len(s.messages) == 1
        assert s.messages[0] == {"role": "assistant", "content": "Здравствуй!"}

    def test_alternating_messages_order(self):
        s = ConversationSession()
        s.add_user_message("Вопрос 1")
        s.add_assistant_message("Ответ 1")
        s.add_user_message("Вопрос 2")
        assert len(s.messages) == 3
        assert s.messages[0]["role"] == "user"
        assert s.messages[1]["role"] == "assistant"
        assert s.messages[2]["role"] == "user"

    def test_updated_at_changes_on_message(self):
        s = ConversationSession()
        old_updated = s.updated_at
        import time; time.sleep(0.01)
        s.add_user_message("test")
        assert s.updated_at >= old_updated

    def test_get_messages_for_api_returns_copy(self):
        s = ConversationSession()
        s.add_user_message("Hello")
        api_messages = s.get_messages_for_api()
        api_messages.append({"role": "user", "content": "extra"})
        assert len(s.messages) == 1

    def test_get_messages_for_api_with_system_prompt(self):
        s = ConversationSession(system_prompt="Be helpful.")
        s.add_user_message("Hi")
        msgs = s.get_messages_for_api()
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


class TestTokenUsage:
    def test_initial_usage_zero(self):
        s = ConversationSession()
        assert s.token_usage["total_tokens"] == 0

    def test_update_token_usage(self):
        s = ConversationSession()
        s.update_token_usage(100, 200)
        assert s.token_usage["prompt_tokens"] == 100
        assert s.token_usage["completion_tokens"] == 200
        assert s.token_usage["total_tokens"] == 300

    def test_update_token_usage_accumulates(self):
        s = ConversationSession()
        s.update_token_usage(50, 100)
        s.update_token_usage(50, 100)
        assert s.token_usage["total_tokens"] == 300

    def test_estimate_tokens_empty(self):
        s = ConversationSession()
        # With tiktoken: system prompt overhead produces a few tokens
        assert s.estimate_tokens() < 10

    def test_estimate_tokens_approx(self):
        s = ConversationSession()
        s.add_user_message("A" * 400)
        estimated = s.estimate_tokens()
        # tiktoken: 400 'A's ≈ 50-110 tokens depending on encoding
        assert 30 <= estimated <= 150

    def test_estimate_tokens_multiple_messages(self):
        s = ConversationSession()
        s.add_user_message("A" * 400)
        s.add_assistant_message("B" * 400)
        assert s.estimate_tokens() >= 60  # at least both messages' tokens


class TestContextInfo:
    def test_context_info_structure(self):
        s = ConversationSession()
        info = s.get_context_info()
        assert "used_tokens" in info
        assert "max_tokens" in info
        assert "percentage" in info
        assert "messages_count" in info
        assert "warning" in info

    def test_context_info_empty_session(self):
        s = ConversationSession()
        info = s.get_context_info()
        assert info["used_tokens"] < 10
        assert info["percentage"] < 1.0
        assert info["warning"] is False

    def test_context_info_percentage_calculation(self):
        s = ConversationSession(model="zai-org/GLM-4.7-Flash")
        max_t = s.max_context_tokens
        # With tiktoken, 'A' * N gives ~N/5.5 tokens, not N/4
        # Use enough chars to get 40-60% range
        s.add_user_message("A" * (max_t * 5 // 2))
        info = s.get_context_info()
        assert 20.0 <= info["percentage"] <= 70.0

    def test_context_warning_triggered(self):
        s = ConversationSession(model="zai-org/GLM-4.7-Flash")
        max_t = s.max_context_tokens
        # Directly set a large message count to simulate 85% fill
        # Patch estimate_tokens to return 85% of max
        target = int(max_t * 0.85)
        s.estimate_tokens = lambda: target
        info = s.get_context_info()
        assert info["warning"] is True

    def test_context_warning_not_triggered_below_threshold(self):
        s = ConversationSession()
        info = s.get_context_info()
        assert info["warning"] is False


class TestContextBar:
    def test_format_context_bar_returns_string(self):
        s = ConversationSession()
        bar = s.format_context_bar()
        assert isinstance(bar, str)

    def test_format_context_bar_contains_tokens(self):
        s = ConversationSession()
        bar = s.format_context_bar()
        assert "токенов" in bar

    def test_format_context_bar_green_when_empty(self):
        s = ConversationSession()
        bar = s.format_context_bar()
        assert "🟢" in bar

    def test_format_context_bar_red_when_full(self):
        s = ConversationSession(model="zai-org/GLM-4.7-Flash")
        max_t = s.max_context_tokens
        # Patch estimate_tokens to return 95% of max (triggers red bar)
        target = int(max_t * 0.95)
        s.estimate_tokens = lambda: target
        bar = s.format_context_bar()
        assert "🔴" in bar

    def test_format_context_bar_contains_percentage(self):
        s = ConversationSession()
        bar = s.format_context_bar()
        assert "%" in bar

    def test_format_context_bar_has_progress_bar(self):
        s = ConversationSession()
        bar = s.format_context_bar()
        assert "[" in bar and "]" in bar


class TestSessionClose:
    def test_close_changes_status(self):
        s = ConversationSession()
        assert s.status == "active"
        s.close()
        assert s.status == "closed"

    def test_close_updates_timestamp(self):
        s = ConversationSession()
        old_ts = s.updated_at
        import time; time.sleep(0.01)
        s.close()
        assert s.updated_at >= old_ts


class TestSerialization:
    def test_to_dict_contains_required_keys(self):
        s = ConversationSession(name="Test", model="gpt-5.4")
        d = s.to_dict()
        for key in ("session_id", "name", "model", "system_prompt",
                    "messages", "created_at", "updated_at", "status", "token_usage"):
            assert key in d

    def test_to_dict_roundtrip(self):
        s = ConversationSession(name="Round trip", model="gpt-5.4", system_prompt="sys")
        s.add_user_message("hello")
        s.add_assistant_message("world")
        s.update_token_usage(10, 20)

        d = s.to_dict()
        s2 = ConversationSession.from_dict(d)

        assert s2.session_id == s.session_id
        assert s2.name == s.name
        assert s2.model == s.model
        assert s2.system_prompt == s.system_prompt
        assert s2.messages == s.messages
        assert s2.status == s.status
        assert s2.token_usage == s.token_usage

    def test_from_dict_missing_optional_fields(self):
        minimal = {"session_id": "abc12345", "messages": []}
        s = ConversationSession.from_dict(minimal)
        assert s.session_id == "abc12345"
        assert s.messages == []
        assert s.status == "active"


class TestSessionStorage:
    @pytest.fixture
    def tmp_storage(self, tmp_path):
        return SessionStorage(base_dir=str(tmp_path / "sessions"))

    def test_creates_directory(self, tmp_path):
        d = tmp_path / "new_sessions"
        assert not d.exists()
        SessionStorage(base_dir=str(d))
        assert d.exists()

    def test_creates_index_file(self, tmp_storage):
        assert tmp_storage.index_file.exists()

    def test_save_and_load(self, tmp_storage):
        s = ConversationSession(name="Save test")
        s.add_user_message("Hello")
        tmp_storage.save(s)

        loaded = tmp_storage.load(s.session_id)
        assert loaded is not None
        assert loaded.session_id == s.session_id
        assert loaded.name == s.name
        assert len(loaded.messages) == 1

    def test_load_nonexistent_returns_none(self, tmp_storage):
        result = tmp_storage.load("nonexistent")
        assert result is None

    def test_load_corrupted_returns_none(self, tmp_storage):
        bad_file = tmp_storage.base_dir / "badfile.json"
        bad_file.write_text("not valid json", encoding="utf-8")
        result = tmp_storage.load("badfile")
        assert result is None

    def test_save_updates_index(self, tmp_storage):
        s1 = ConversationSession(name="First")
        s2 = ConversationSession(name="Second")
        tmp_storage.save(s1)
        tmp_storage.save(s2)

        sessions = tmp_storage.list_sessions()
        ids = [e["session_id"] for e in sessions]
        assert s1.session_id in ids
        assert s2.session_id in ids

    def test_save_overwrites_existing_in_index(self, tmp_storage):
        s = ConversationSession(name="Original")
        tmp_storage.save(s)

        s.name = "Updated"
        tmp_storage.save(s)

        sessions = tmp_storage.list_sessions()
        entry = next(e for e in sessions if e["session_id"] == s.session_id)
        assert entry["name"] == "Updated"

    def test_list_sessions_empty(self, tmp_storage):
        assert tmp_storage.list_sessions() == []

    def test_list_sessions_respects_limit(self, tmp_storage):
        for i in range(5):
            tmp_storage.save(ConversationSession(name=f"Session {i}"))
        result = tmp_storage.list_sessions(n=3)
        assert len(result) == 3

    def test_list_sessions_filter_by_status(self, tmp_storage):
        active = ConversationSession(name="Active")
        closed = ConversationSession(name="Closed")
        closed.close()
        tmp_storage.save(active)
        tmp_storage.save(closed)

        active_list = tmp_storage.list_sessions(status="active")
        closed_list = tmp_storage.list_sessions(status="closed")

        active_ids = [e["session_id"] for e in active_list]
        closed_ids = [e["session_id"] for e in closed_list]
        assert active.session_id in active_ids
        assert closed.session_id not in active_ids
        assert closed.session_id in closed_ids

    def test_delete_removes_file(self, tmp_storage):
        s = ConversationSession(name="To delete")
        tmp_storage.save(s)
        assert (tmp_storage.base_dir / f"{s.session_id}.json").exists()

        result = tmp_storage.delete(s.session_id)
        assert result is True
        assert not (tmp_storage.base_dir / f"{s.session_id}.json").exists()

    def test_delete_removes_from_index(self, tmp_storage):
        s = ConversationSession(name="To delete")
        tmp_storage.save(s)
        tmp_storage.delete(s.session_id)

        sessions = tmp_storage.list_sessions()
        ids = [e["session_id"] for e in sessions]
        assert s.session_id not in ids

    def test_delete_nonexistent_returns_false(self, tmp_storage):
        assert tmp_storage.delete("nonexistent") is False

    def test_find_active(self, tmp_storage):
        s1 = ConversationSession(name="Active 1")
        s2 = ConversationSession(name="Closed 1")
        s2.close()
        tmp_storage.save(s1)
        tmp_storage.save(s2)

        active = tmp_storage.find_active()
        ids = [e["session_id"] for e in active]
        assert s1.session_id in ids
        assert s2.session_id not in ids

    def test_find_by_name_case_insensitive(self, tmp_storage):
        s = ConversationSession(name="Python Talk")
        tmp_storage.save(s)

        result = tmp_storage.find_by_name("python talk")
        assert result is not None
        assert result["session_id"] == s.session_id

    def test_find_by_name_not_found(self, tmp_storage):
        assert tmp_storage.find_by_name("nonexistent name") is None

    def test_index_json_is_valid(self, tmp_storage):
        s = ConversationSession(name="Test")
        tmp_storage.save(s)
        with open(tmp_storage.index_file, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_session_file_is_valid_json(self, tmp_storage):
        s = ConversationSession(name="Test")
        s.add_user_message("Hello")
        tmp_storage.save(s)

        session_file = tmp_storage.base_dir / f"{s.session_id}.json"
        with open(session_file, encoding="utf-8") as f:
            data = json.load(f)
        assert data["session_id"] == s.session_id
        assert len(data["messages"]) == 1


class TestIntegration:
    """Полный цикл: создание → переписка → закрытие → перезагрузка."""

    def test_full_session_lifecycle(self, tmp_path):
        storage = SessionStorage(str(tmp_path / "sessions"))

        sess = ConversationSession(name="Integration test", model="zai-org/GLM-4.7-Flash")
        sess.add_user_message("Привет!")
        sess.add_assistant_message("Здравствуй!")
        sess.add_user_message("Как дела?")
        sess.add_assistant_message("Отлично!")
        sess.update_token_usage(20, 40)
        storage.save(sess)

        loaded = storage.load(sess.session_id)
        assert loaded is not None
        assert len(loaded.messages) == 4
        assert loaded.status == "active"

        loaded.close()
        storage.save(loaded)

        closed = storage.load(sess.session_id)
        assert closed.status == "closed"

        closed.status = "active"
        closed.add_user_message("Продолжаем?")
        storage.save(closed)

        resumed = storage.load(sess.session_id)
        assert len(resumed.messages) == 5
        assert resumed.status == "active"

    def test_context_bar_reflects_conversation(self, tmp_path):
        storage = SessionStorage(str(tmp_path / "sessions"))
        sess = ConversationSession(model="zai-org/GLM-4.7-Flash")

        bar_empty = sess.format_context_bar()
        assert "0.0%" in bar_empty

        sess.add_user_message("Hello " * 100)
        bar_with_content = sess.format_context_bar()
        assert bar_empty != bar_with_content
