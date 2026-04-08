"""
Тесты для news_agent/context_compressor.py и интеграции с ConversationSession.

Покрывают:
  - CompressionStats: свойства tokens_saved, compression_ratio, format()
  - ContextCompressor: создание, needs_compression, get_compressed_messages,
    compress() с моком LLM, get_comparison, format_stats, сериализация
  - ConversationSession: enable/disable compression, maybe_compress,
    get_messages_for_api со сжатием, get_compression_info, сериализация
"""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from news_agent.context_compressor import (
    ContextCompressor,
    CompressionStats,
    DEFAULT_SUMMARIZE_EVERY,
    DEFAULT_KEEP_LAST_N,
    SUMMARY_PREFIX,
)
from news_agent.session_manager import ConversationSession


# ─────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────

def _make_messages(n: int, include_system: bool = False) -> list:
    """Создаёт список из n user+assistant пар (2n сообщений)."""
    msgs = []
    if include_system:
        msgs.append({"role": "system", "content": "Ты помощник."})
    for i in range(n):
        msgs.append({"role": "user", "content": f"Вопрос {i + 1}: что-то важное"})
        msgs.append({"role": "assistant", "content": f"Ответ {i + 1}: подробный ответ"})
    return msgs


def _make_mock_client(summary_text: str = "Краткое изложение диалога.") -> MagicMock:
    """Создаёт мок OpenAI клиента для суммаризации."""
    client = MagicMock()
    response = MagicMock()
    response.choices[0].message.content = summary_text
    client.chat.completions.create.return_value = response
    return client


# ─────────────────────────────────────────────────────────────
# CompressionStats
# ─────────────────────────────────────────────────────────────

class TestCompressionStats:

    def test_tokens_saved_positive(self):
        stats = CompressionStats(
            compression_index=1,
            messages_compressed=10,
            tokens_before=500,
            tokens_after=200,
            summary_length=150,
        )
        assert stats.tokens_saved == 300

    def test_tokens_saved_no_negative(self):
        stats = CompressionStats(
            compression_index=1,
            messages_compressed=5,
            tokens_before=100,
            tokens_after=120,  # после > до (невозможно в норме, но защита)
            summary_length=80,
        )
        assert stats.tokens_saved == 0

    def test_compression_ratio_less_than_one(self):
        stats = CompressionStats(
            compression_index=1,
            messages_compressed=10,
            tokens_before=500,
            tokens_after=200,
            summary_length=150,
        )
        assert stats.compression_ratio == pytest.approx(0.4)

    def test_compression_ratio_zero_before(self):
        stats = CompressionStats(
            compression_index=1,
            messages_compressed=0,
            tokens_before=0,
            tokens_after=0,
            summary_length=0,
        )
        assert stats.compression_ratio == 1.0

    def test_format_contains_key_info(self):
        stats = CompressionStats(
            compression_index=2,
            messages_compressed=10,
            tokens_before=500,
            tokens_after=200,
            summary_length=150,
        )
        text = stats.format()
        assert "#2" in text
        assert "10" in text
        assert "300" in text  # tokens_saved


# ─────────────────────────────────────────────────────────────
# ContextCompressor — создание и константы
# ─────────────────────────────────────────────────────────────

class TestContextCompressorCreation:

    def test_default_params(self):
        c = ContextCompressor()
        assert c.summarize_every == DEFAULT_SUMMARIZE_EVERY
        assert c.keep_last_n == DEFAULT_KEEP_LAST_N
        assert c.summaries == []
        assert c.summary_covers_up_to == 0
        assert c.compression_count == 0
        assert c.total_tokens_saved == 0

    def test_custom_params(self):
        c = ContextCompressor(summarize_every=5, keep_last_n=3)
        assert c.summarize_every == 5
        assert c.keep_last_n == 3

    def test_model_stored(self):
        c = ContextCompressor(model="gpt-5.4")
        assert c.model == "gpt-5.4"


# ─────────────────────────────────────────────────────────────
# ContextCompressor.needs_compression
# ─────────────────────────────────────────────────────────────

class TestNeedsCompression:

    def test_no_compression_needed_empty(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        assert not c.needs_compression([])

    def test_no_compression_needed_small_history(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        msgs = _make_messages(7)  # 14 сообщений, но (14 - 0 - 6) = 8 < 10
        assert not c.needs_compression(msgs)

    def test_compression_needed_exactly_at_threshold(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        # chat_msgs = 16, compressible = 16 - 0 - 6 = 10 == threshold
        msgs = _make_messages(8)
        assert c.needs_compression(msgs)

    def test_compression_needed_above_threshold(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        msgs = _make_messages(12)  # 24 chat msgs, compressible = 24 - 0 - 6 = 18 > 10
        assert c.needs_compression(msgs)

    def test_no_compression_after_already_done(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        # Симулируем уже выполненное сжатие
        c.summary_covers_up_to = 10
        c.summaries = ["Краткое изложение."]
        # 16 chat msgs: compressible = 16 - 10 - 6 = 0 < 10
        msgs = _make_messages(8)
        assert not c.needs_compression(msgs)

    def test_system_messages_ignored_in_count(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        msgs = _make_messages(8, include_system=True)  # system не считается
        # Всё равно 16 chat msgs
        assert c.needs_compression(msgs)


# ─────────────────────────────────────────────────────────────
# ContextCompressor.get_compressed_messages
# ─────────────────────────────────────────────────────────────

class TestGetCompressedMessages:

    def test_no_summaries_returns_full_history(self):
        c = ContextCompressor()
        msgs = _make_messages(5)
        result = c.get_compressed_messages(msgs)
        assert result == msgs

    def test_with_summary_contains_summary_message(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        c.summaries = ["Ключевые моменты диалога."]
        c.summary_covers_up_to = 10
        msgs = _make_messages(8)  # 16 chat msgs
        result = c.get_compressed_messages(msgs)

        # Найти summary сообщение
        summary_msgs = [m for m in result if SUMMARY_PREFIX in m.get("content", "")]
        assert len(summary_msgs) == 1
        assert "Ключевые моменты диалога." in summary_msgs[0]["content"]

    def test_with_summary_recent_messages_preserved(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        c.summaries = ["Summary."]
        c.summary_covers_up_to = 10
        msgs = _make_messages(8)  # chat: 16 msgs
        result = c.get_compressed_messages(msgs)

        chat_in_result = [m for m in result if m["role"] in ("user", "assistant")]
        # Должны быть последние 6 сообщений (16 - 10 = 6 несжатых)
        assert len(chat_in_result) == 6

    def test_system_messages_preserved(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        c.summaries = ["Summary."]
        c.summary_covers_up_to = 10
        msgs = _make_messages(8, include_system=True)
        result = c.get_compressed_messages(msgs)

        system_msgs = [m for m in result if m["role"] == "system" and SUMMARY_PREFIX not in m.get("content", "")]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Ты помощник."

    def test_multiple_summaries_combined(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        c.summaries = ["Часть 1.", "Часть 2."]
        c.summary_covers_up_to = 20
        msgs = _make_messages(13)  # 26 chat msgs
        result = c.get_compressed_messages(msgs)

        summary_msgs = [m for m in result if SUMMARY_PREFIX in m.get("content", "")]
        assert len(summary_msgs) == 1
        content = summary_msgs[0]["content"]
        assert "Часть 1." in content
        assert "Часть 2." in content

    def test_compressed_fewer_messages_than_full(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        c.summaries = ["Summary."]
        c.summary_covers_up_to = 10
        msgs = _make_messages(8)
        full = list(msgs)
        compressed = c.get_compressed_messages(msgs)
        assert len(compressed) < len(full)


# ─────────────────────────────────────────────────────────────
# ContextCompressor.compress (с моком LLM)
# ─────────────────────────────────────────────────────────────

class TestCompress:

    def test_returns_none_when_not_needed(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        msgs = _make_messages(4)  # слишком мало
        client = _make_mock_client()
        result = c.compress(client, msgs)
        assert result is None
        client.chat.completions.create.assert_not_called()

    def test_returns_stats_when_needed(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        msgs = _make_messages(8)  # 16 chat msgs, compressible=10 >= threshold
        client = _make_mock_client("Краткое изложение первых 10 вопросов.")
        result = c.compress(client, msgs)
        assert result is not None
        assert isinstance(result, CompressionStats)

    def test_compress_increments_count(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        msgs = _make_messages(8)
        client = _make_mock_client()
        c.compress(client, msgs)
        assert c.compression_count == 1

    def test_compress_updates_summary_covers(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        msgs = _make_messages(8)  # 16 chat msgs, compress_end = 16-6=10
        client = _make_mock_client()
        c.compress(client, msgs)
        assert c.summary_covers_up_to == 10

    def test_compress_stores_summary_text(self):
        summary_text = "Это краткое изложение первых 10 сообщений."
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        msgs = _make_messages(8)
        client = _make_mock_client(summary_text)
        c.compress(client, msgs)
        assert summary_text in c.summaries

    def test_compress_stats_messages_compressed(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        msgs = _make_messages(8)  # compress 16-6=10 messages
        client = _make_mock_client()
        stats = c.compress(client, msgs)
        assert stats.messages_compressed == 10

    def test_compress_stats_tokens_before_gt_after(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        msgs = _make_messages(8)
        client = _make_mock_client("Краткое.")
        stats = c.compress(client, msgs)
        # После сжатия токенов должно быть меньше или равно
        assert stats.tokens_after <= stats.tokens_before

    def test_second_compression_after_more_messages(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)

        # Первое сжатие
        msgs = _make_messages(8)
        client = _make_mock_client("Summary 1.")
        c.compress(client, msgs)
        assert c.compression_count == 1

        # Добавляем ещё 10+ сообщений
        msgs = _make_messages(14)  # 28 chat msgs; compressible = 28-10-6=12 >= 10
        stats2 = c.compress(client, msgs)
        assert stats2 is not None
        assert c.compression_count == 2
        assert len(c.summaries) == 2

    def test_llm_called_with_correct_model(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6, model="gpt-5.4")
        msgs = _make_messages(8)
        client = _make_mock_client()
        c.compress(client, msgs)
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-5.4"

    def test_gpt5nano_no_temperature(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6, model="gpt-5-nano")
        msgs = _make_messages(8)
        client = _make_mock_client()
        c.compress(client, msgs)
        call_kwargs = client.chat.completions.create.call_args[1]
        assert "temperature" not in call_kwargs


# ─────────────────────────────────────────────────────────────
# ContextCompressor.get_comparison
# ─────────────────────────────────────────────────────────────

class TestGetComparison:

    def test_keys_present(self):
        c = ContextCompressor()
        msgs = _make_messages(4)
        result = c.get_comparison(msgs)
        for key in (
            "full_messages_count", "full_tokens",
            "compressed_messages_count", "compressed_tokens",
            "tokens_saved", "savings_pct", "compression_count",
        ):
            assert key in result

    def test_no_compression_equal_tokens(self):
        c = ContextCompressor()
        msgs = _make_messages(4)
        result = c.get_comparison(msgs)
        assert result["tokens_saved"] == 0
        assert result["full_tokens"] == result["compressed_tokens"]

    def test_with_compression_saves_tokens(self):
        c = ContextCompressor(summarize_every=10, keep_last_n=6)
        c.summaries = ["Brief summary."]
        c.summary_covers_up_to = 10
        msgs = _make_messages(8)
        result = c.get_comparison(msgs)
        assert result["tokens_saved"] > 0
        assert result["savings_pct"] > 0


# ─────────────────────────────────────────────────────────────
# ContextCompressor.format_stats / format_summary_preview
# ─────────────────────────────────────────────────────────────

class TestFormatStats:

    def test_no_compressions(self):
        c = ContextCompressor()
        text = c.format_stats()
        assert "не выполнялось" in text.lower()

    def test_with_compressions(self):
        c = ContextCompressor()
        c.compression_count = 2
        c.summary_covers_up_to = 20
        c.total_tokens_saved = 1500
        c.summaries = ["S1", "S2"]
        text = c.format_stats()
        assert "2" in text
        assert "20" in text
        assert "1,500" in text or "1500" in text

    def test_summary_preview_no_summaries(self):
        c = ContextCompressor()
        text = c.format_summary_preview()
        assert "отсутствует" in text.lower()

    def test_summary_preview_with_summary(self):
        c = ContextCompressor()
        c.summaries = ["Это краткое изложение диалога с пользователем."]
        text = c.format_summary_preview()
        assert "Это краткое изложение" in text


# ─────────────────────────────────────────────────────────────
# ContextCompressor — сериализация
# ─────────────────────────────────────────────────────────────

class TestContextCompressorSerialization:

    def test_to_dict_keys(self):
        c = ContextCompressor(summarize_every=5, keep_last_n=3, model="gpt-5.4")
        d = c.to_dict()
        for key in (
            "summarize_every", "keep_last_n", "model",
            "summaries", "summary_covers_up_to",
            "compression_count", "total_tokens_saved",
        ):
            assert key in d

    def test_roundtrip(self):
        c = ContextCompressor(summarize_every=5, keep_last_n=3, model="gpt-5.4")
        c.summaries = ["S1", "S2"]
        c.summary_covers_up_to = 10
        c.compression_count = 2
        c.total_tokens_saved = 750

        restored = ContextCompressor.from_dict(c.to_dict())
        assert restored.summarize_every == 5
        assert restored.keep_last_n == 3
        assert restored.model == "gpt-5.4"
        assert restored.summaries == ["S1", "S2"]
        assert restored.summary_covers_up_to == 10
        assert restored.compression_count == 2
        assert restored.total_tokens_saved == 750

    def test_from_dict_defaults_on_missing_keys(self):
        restored = ContextCompressor.from_dict({})
        assert restored.summarize_every == DEFAULT_SUMMARIZE_EVERY
        assert restored.keep_last_n == DEFAULT_KEEP_LAST_N
        assert restored.summaries == []

    def test_to_dict_is_json_serializable(self):
        c = ContextCompressor()
        c.summaries = ["Test summary."]
        d = c.to_dict()
        json_str = json.dumps(d)  # не должно бросить исключение
        assert "Test summary." in json_str


# ─────────────────────────────────────────────────────────────
# ConversationSession — интеграция с компрессией
# ─────────────────────────────────────────────────────────────

class TestConversationSessionCompression:

    def test_compression_disabled_by_default(self):
        s = ConversationSession()
        assert s.compression_enabled is False
        assert s.compressor is None

    def test_enable_compression(self):
        s = ConversationSession()
        s.enable_compression()
        assert s.compression_enabled is True
        assert s.compressor is not None
        assert isinstance(s.compressor, ContextCompressor)

    def test_enable_compression_custom_params(self):
        s = ConversationSession()
        s.enable_compression(summarize_every=5, keep_last_n=3)
        assert s.compressor.summarize_every == 5
        assert s.compressor.keep_last_n == 3

    def test_enable_compression_reuses_existing(self):
        s = ConversationSession()
        s.enable_compression()
        original_compressor = s.compressor
        s.enable_compression()  # повторный вызов
        assert s.compressor is original_compressor  # тот же объект

    def test_disable_compression(self):
        s = ConversationSession()
        s.enable_compression()
        s.disable_compression()
        assert s.compression_enabled is False
        assert s.compressor is not None  # summaries сохранены

    def test_get_messages_for_api_no_compression(self):
        s = ConversationSession()
        s.add_user_message("Hello")
        s.add_assistant_message("World")
        msgs = s.get_messages_for_api()
        assert len(msgs) == 2

    def test_get_messages_for_api_with_compression_no_summaries(self):
        s = ConversationSession()
        s.enable_compression()
        s.add_user_message("Hello")
        s.add_assistant_message("World")
        # Компрессор есть, но summaries пусты — полная история
        msgs = s.get_messages_for_api()
        assert len(msgs) == 2

    def test_get_messages_for_api_with_summaries(self):
        s = ConversationSession()
        s.enable_compression(summarize_every=10, keep_last_n=6)
        # Добавляем 16 chat msgs
        for i in range(8):
            s.add_user_message(f"Вопрос {i}")
            s.add_assistant_message(f"Ответ {i}")
        # Симулируем выполненное сжатие
        s.compressor.summaries = ["Summary of first 10."]
        s.compressor.summary_covers_up_to = 10

        msgs = s.get_messages_for_api()
        # Должно быть: 1 summary + 6 recent
        assert len(msgs) == 7
        assert SUMMARY_PREFIX in msgs[0]["content"]

    def test_maybe_compress_returns_none_when_disabled(self):
        s = ConversationSession()
        client = _make_mock_client()
        result = s.maybe_compress(client)
        assert result is None

    def test_maybe_compress_returns_none_when_not_needed(self):
        s = ConversationSession()
        s.enable_compression(summarize_every=10, keep_last_n=6)
        s.add_user_message("Hi")
        s.add_assistant_message("Hello")
        client = _make_mock_client()
        result = s.maybe_compress(client)
        assert result is None

    def test_maybe_compress_triggers_when_needed(self):
        s = ConversationSession()
        s.enable_compression(summarize_every=10, keep_last_n=6)
        for i in range(8):
            s.add_user_message(f"Вопрос {i}")
            s.add_assistant_message(f"Ответ {i}")
        client = _make_mock_client("Краткое изложение.")
        result = s.maybe_compress(client)
        assert result is not None
        assert isinstance(result, CompressionStats)
        assert s.compressor.compression_count == 1

    def test_get_compression_info_no_compressor(self):
        s = ConversationSession()
        info = s.get_compression_info()
        assert info["enabled"] is False
        assert info["compression_count"] == 0
        assert info["summarize_every"] is None

    def test_get_compression_info_with_compressor(self):
        s = ConversationSession()
        s.enable_compression(summarize_every=10, keep_last_n=6)
        s.compressor.compression_count = 3
        s.compressor.summary_covers_up_to = 30
        s.compressor.total_tokens_saved = 1200
        info = s.get_compression_info()
        assert info["enabled"] is True
        assert info["compression_count"] == 3
        assert info["messages_summarized"] == 30
        assert info["total_tokens_saved"] == 1200
        assert info["summarize_every"] == 10
        assert info["keep_last_n"] == 6

    def test_to_dict_includes_compression(self):
        s = ConversationSession()
        s.enable_compression()
        d = s.to_dict()
        assert "compression_enabled" in d
        assert "compressor" in d
        assert d["compression_enabled"] is True

    def test_to_dict_no_compressor_key_when_none(self):
        s = ConversationSession()
        d = s.to_dict()
        assert d["compression_enabled"] is False
        assert "compressor" not in d

    def test_from_dict_restores_compression_state(self):
        s = ConversationSession()
        s.enable_compression(summarize_every=10, keep_last_n=6)
        s.compressor.summaries = ["Saved summary."]
        s.compressor.summary_covers_up_to = 10
        s.compressor.compression_count = 1

        data = s.to_dict()
        restored = ConversationSession.from_dict(data)

        assert restored.compression_enabled is True
        assert restored.compressor is not None
        assert restored.compressor.summaries == ["Saved summary."]
        assert restored.compressor.summary_covers_up_to == 10
        assert restored.compressor.compression_count == 1

    def test_from_dict_backward_compat_no_compression_keys(self):
        """Старые JSON-файлы без полей компрессии должны загружаться корректно."""
        data = {
            "session_id": "abc12345",
            "name": "Старая сессия",
            "model": "zai-org/GLM-4.7-Flash",
            "system_prompt": "",
            "messages": [],
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "status": "active",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        s = ConversationSession.from_dict(data)
        assert s.compression_enabled is False
        assert s.compressor is None
