"""
Тесты для llm_cli.py.

Покрывают: модели, режимы, расчёт стоимости, подсчёт слов,
execute_mode, meta_prompting, параметры температуры,
а также новые функции управления сессиями (_print_session_info,
_cmd_chat_list, _cmd_chat_delete) и вспомогательные.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from llm_cli import (
    get_available_modes,
    get_available_models,
    calculate_cost,
    count_words,
    execute_mode,
    execute_meta_prompting,
    _cmd_chat_list,
    _cmd_chat_delete,
    _print_session_info,
    _build_token_comparison,
    _format_token_comparison,
)
from news_agent.session_manager import ConversationSession, SessionStorage

@pytest.fixture
def mock_client():
    client = MagicMock()
    
    # Mock для streaming ответа
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta = MagicMock()
    mock_chunk.choices[0].delta.content = "Test response from model"
    mock_chunk.choices[0].delta.reasoning_content = None
    
    # Mock для usage запроса
    usage_response = MagicMock()
    usage_response.usage.prompt_tokens = 15
    usage_response.usage.completion_tokens = 25
    usage_response.usage.total_tokens = 40
    
    # Настраиваем возврат разных значений для разных вызовов
    def create_side_effect(**kwargs):
        if kwargs.get('stream'):
            return iter([mock_chunk])
        else:
            return usage_response
    
    client.chat.completions.create.side_effect = create_side_effect
    return client

class TestModels:
    def test_get_available_models_count(self):
        models = get_available_models()
        assert len(models) == 5, "Should have 5 models (2 GLM + 3 OpenAI)"
    
    def test_all_models_have_pricing(self):
        models = get_available_models()
        for model_id, model_info in models.items():
            assert "prompt_price" in model_info
            assert "completion_price" in model_info
            assert model_info["prompt_price"] >= 0
            assert model_info["completion_price"] >= 0
    
    def test_all_models_present(self):
        models = get_available_models()
        assert "zai-org/GLM-4.7-Flash" in models
        assert "zai-org/GLM-4.7" in models
        assert "gpt-5-nano" in models
        assert "gpt-5.4" in models
        assert "gpt-5.4-mini" in models
    
    def test_glm_models_free(self):
        models = get_available_models()
        for model_key in ["zai-org/GLM-4.7-Flash", "zai-org/GLM-4.7"]:
            glm_model = models[model_key]
            assert glm_model["prompt_price"] == 0.0
            assert glm_model["completion_price"] == 0.0
    
    def test_models_have_ids_and_providers(self):
        models = get_available_models()
        assert models["zai-org/GLM-4.7-Flash"]["id"] == 1
        assert models["zai-org/GLM-4.7-Flash"]["provider"] == "cloud_ru"
        assert models["zai-org/GLM-4.7"]["id"] == 2
        assert models["zai-org/GLM-4.7"]["provider"] == "cloud_ru"
        assert models["gpt-5-nano"]["id"] == 3
        assert models["gpt-5-nano"]["provider"] == "openai"
        assert models["gpt-5.4"]["id"] == 4
        assert models["gpt-5.4"]["provider"] == "openai"
        assert models["gpt-5.4-mini"]["id"] == 5
        assert models["gpt-5.4-mini"]["provider"] == "openai"
    
    def test_gpt54_mini_pricing(self):
        models = get_available_models()
        mini = models["gpt-5.4-mini"]
        assert mini["prompt_price"] == 0.00075
        assert mini["completion_price"] == 0.0045
        assert mini["prompt_price"] < models["gpt-5.4"]["prompt_price"]
        assert mini["completion_price"] < models["gpt-5.4"]["completion_price"]

class TestModes:
    def test_get_available_modes_count(self):
        modes = get_available_modes()
        assert len(modes) == 6, "Should have 6 modes"
    
    def test_all_modes_have_required_fields(self):
        modes = get_available_modes()
        for mode in modes:
            assert "id" in mode
            assert "name" in mode
            assert "params" in mode
            assert "metadata" in mode
    
    def test_mode_ids_sequential(self):
        modes = get_available_modes()
        ids = [mode["id"] for mode in modes]
        assert ids == [1, 2, 3, 4, 5, 6]
    
    def test_mode3_uses_max_completion_tokens(self):
        modes = get_available_modes()
        mode3 = next(m for m in modes if m["id"] == 3)
        assert "max_completion_tokens" in mode3["params"]
        assert "max_tokens" not in mode3["params"]
    
    def test_mode4_no_stop_parameter(self):
        modes = get_available_modes()
        mode4 = next(m for m in modes if m["id"] == 4)
        assert "stop" not in mode4["params"]
    
    def test_mode6_has_two_stage_metadata(self):
        modes = get_available_modes()
        mode6 = next(m for m in modes if m["id"] == 6)
        assert mode6["metadata"]["two_stage"] == True

class TestCostCalculation:
    def test_calculate_cost_glm_free(self):
        cost = calculate_cost(100, 200, "zai-org/GLM-4.7-Flash")
        assert cost == 0.0
    
    def test_calculate_cost_unknown_model_defaults_to_glm(self):
        cost = calculate_cost(100, 200, "unknown-model")
        expected = calculate_cost(100, 200, "zai-org/GLM-4.7-Flash")
        assert cost == expected
        assert cost == 0.0

class TestWordCount:
    def test_count_words_simple(self):
        assert count_words("Hello world") == 2
    
    def test_count_words_empty(self):
        assert count_words("") == 0
    
    def test_count_words_multiline(self):
        text = "Line one\nLine two\nLine three"
        assert count_words(text) == 6

class TestExecuteMode:
    def test_execute_mode_basic(self, mock_client):
        modes = get_available_modes()
        mode1 = modes[0]
        response = execute_mode(mock_client, mode1, "Test question")
        
        assert mock_client.chat.completions.create.called
        assert response.choices[0].message.content == "Test response from model"
    
    def test_execute_mode_with_system_prompt(self, mock_client):
        modes = get_available_modes()
        mode2 = modes[1]
        execute_mode(mock_client, mode2, "Test question")
        
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
    
    def test_execute_mode6_calls_meta_prompting(self, mock_client):
        modes = get_available_modes()
        mode6 = next(m for m in modes if m["id"] == 6)
        
        with patch('llm_cli.execute_meta_prompting') as mock_meta:
            mock_meta.return_value = MagicMock()
            execute_mode(mock_client, mode6, "Test question")
            assert mock_meta.called

class TestMetaPrompting:
    def test_meta_prompting_two_calls(self, mock_client):
        modes = get_available_modes()
        mode6 = next(m for m in modes if m["id"] == 6)
        
        execute_meta_prompting(mock_client, mode6, "Explain quantum physics")
        
        assert mock_client.chat.completions.create.call_count == 2
    
    def test_meta_prompting_combines_usage(self, mock_client):
        modes = get_available_modes()
        mode6 = next(m for m in modes if m["id"] == 6)
        
        response = execute_meta_prompting(mock_client, mode6, "Test question")
        
        assert response.usage.prompt_tokens == 30
        assert response.usage.completion_tokens == 50
        assert response.usage.total_tokens == 80
    
    def test_meta_prompting_output_format(self, mock_client):
        modes = get_available_modes()
        mode6 = next(m for m in modes if m["id"] == 6)
        
        response = execute_meta_prompting(mock_client, mode6, "Test question")
        content = response.choices[0].message.content
        
        assert "[МЕТААНАЛИЗ]" in content
        assert "[ОТВЕТ]" in content

class TestTemperatureParameters:
    def test_modes_have_temperature(self):
        modes = get_available_modes()
        for mode in modes:
            if mode["id"] != 6:
                assert "temperature" in mode["params"]
                assert mode["params"]["temperature"] == 0.7
    
    def test_temperature_range_valid(self):
        modes = get_available_modes()
        for mode in modes:
            if "temperature" in mode["params"]:
                temp = mode["params"]["temperature"]
                assert 0 <= temp <= 2.0, f"Temperature {temp} out of valid range"
    
    def test_gpt5_nano_no_temperature_support(self):
        from llm_cli import get_model_params
        params = {"model": "gpt-5-nano", "temperature": 0.7, "max_completion_tokens": 100}
        adapted = get_model_params("gpt-5-nano", params)
        assert "temperature" not in adapted, "gpt-5-nano should not have temperature parameter"
    
    def test_other_models_keep_temperature(self):
        from llm_cli import get_model_params
        for model_name in ["gpt-5.4", "gpt-5.4-mini", "zai-org/GLM-4.7-Flash"]:
            params = {"model": model_name, "temperature": 0.7, "max_completion_tokens": 100}
            adapted = get_model_params(model_name, params)
            assert "temperature" in adapted
            assert adapted["temperature"] == 0.7

class TestIntegration:
    def test_mode_uses_glm_model(self):
        modes = get_available_modes()
        assert all(m["params"]["model"] == "zai-org/GLM-4.7-Flash" for m in modes)
    
    def test_all_modes_executable(self, mock_client):
        modes = get_available_modes()
        for mode in modes:
            try:
                response = execute_mode(mock_client, mode, "Test question")
                assert response is not None
            except Exception as e:
                pytest.fail(f"Mode {mode['id']} failed: {str(e)}")
    
    def test_modes_with_different_models(self):
        model_names = ["zai-org/GLM-4.7-Flash", "gpt-5.4", "gpt-5.4-mini"]
        for model_name in model_names:
            modes = get_available_modes(model_name)
            assert len(modes) == 6
            assert all(m["params"]["model"] == model_name for m in modes)


class TestChatSessionFunctions:
    """Тесты для CLI-функций управления сессиями."""

    @pytest.fixture
    def tmp_storage(self, tmp_path):
        return SessionStorage(str(tmp_path / "sessions"))

    def test_cmd_chat_list_empty(self, tmp_storage, capsys):
        _cmd_chat_list(tmp_storage)
        out = capsys.readouterr().out
        assert "сессий" in out.lower() or "нет" in out.lower()

    def test_cmd_chat_list_shows_sessions(self, tmp_storage, capsys):
        s1 = ConversationSession(name="Первая сессия")
        s2 = ConversationSession(name="Вторая сессия")
        tmp_storage.save(s1)
        tmp_storage.save(s2)

        _cmd_chat_list(tmp_storage)
        out = capsys.readouterr().out
        assert s1.session_id in out
        assert s2.session_id in out

    def test_cmd_chat_list_shows_status_icons(self, tmp_storage, capsys):
        active = ConversationSession(name="Active")
        closed = ConversationSession(name="Closed")
        closed.close()
        tmp_storage.save(active)
        tmp_storage.save(closed)

        _cmd_chat_list(tmp_storage)
        out = capsys.readouterr().out
        assert "●" in out
        assert "○" in out

    def test_cmd_chat_delete_existing(self, tmp_storage, capsys):
        s = ConversationSession(name="To delete")
        tmp_storage.save(s)

        _cmd_chat_delete(s.session_id, tmp_storage)
        out = capsys.readouterr().out
        assert "✅" in out
        assert tmp_storage.load(s.session_id) is None

    def test_cmd_chat_delete_nonexistent(self, tmp_storage, capsys):
        _cmd_chat_delete("nonexistent_id", tmp_storage)
        out = capsys.readouterr().out
        assert "❌" in out

    def test_print_session_info_shows_all_fields(self, capsys):
        s = ConversationSession(name="Test Info", model="gpt-5.4")
        s.add_user_message("Hello")
        s.add_assistant_message("World")
        s.update_token_usage(10, 20)

        _print_session_info(s)
        out = capsys.readouterr().out

        assert s.session_id in out
        assert "Test Info" in out
        assert "gpt-5.4" in out
        assert "active" in out
        assert "токенов" in out.lower() or "контекст" in out.lower()

    def test_print_session_info_shows_context_bar(self, capsys):
        s = ConversationSession(name="Context test")
        _print_session_info(s)
        out = capsys.readouterr().out
        assert "%" in out


class TestSessionManagerImport:
    """Проверяем корректность импорта из session_manager в llm_cli."""

    def test_model_context_sizes_imported(self):
        from llm_cli import MODEL_CONTEXT_SIZES
        assert "zai-org/GLM-4.7-Flash" in MODEL_CONTEXT_SIZES

    def test_conversation_session_imported(self):
        from llm_cli import ConversationSession
        s = ConversationSession()
        assert s.status == "active"

    def test_session_storage_imported(self):
        from llm_cli import SessionStorage
        assert SessionStorage is not None


class TestTokenComparison:
    """Тесты для _build_token_comparison и _format_token_comparison."""

    def _api_usage(self, prompt=100, completion=50):
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}

    # ── _build_token_comparison ───────────────────────────────────────

    def test_no_api_usage_returns_has_api_false(self):
        cmp = _build_token_comparison(100, 50, None, "tiktoken")
        assert cmp is not None
        assert cmp["has_api"] is False
        assert cmp["method"] == "tiktoken"

    def test_with_api_usage_has_api_true(self):
        cmp = _build_token_comparison(100, 50, self._api_usage(100, 50), "tiktoken")
        assert cmp["has_api"] is True

    def test_exact_match_zero_diff(self):
        cmp = _build_token_comparison(100, 50, self._api_usage(100, 50), "tiktoken")
        assert cmp["diff_prompt"] == 0
        assert cmp["diff_compl"] == 0
        assert cmp["pct_prompt"] == pytest.approx(0.0)
        assert cmp["pct_compl"] == pytest.approx(0.0)

    def test_local_over_api(self):
        cmp = _build_token_comparison(110, 55, self._api_usage(100, 50), "tiktoken")
        assert cmp["diff_prompt"] == 10
        assert cmp["diff_compl"] == 5
        assert cmp["pct_prompt"] == pytest.approx(10.0)
        assert cmp["pct_compl"] == pytest.approx(10.0)

    def test_local_under_api(self):
        cmp = _build_token_comparison(90, 45, self._api_usage(100, 50), "~chars÷4")
        assert cmp["diff_prompt"] == -10
        assert cmp["diff_compl"] == -5
        assert cmp["pct_prompt"] == pytest.approx(-10.0)
        assert cmp["pct_compl"] == pytest.approx(-10.0)

    def test_api_total_stored(self):
        cmp = _build_token_comparison(100, 50, self._api_usage(100, 50), "tiktoken")
        assert cmp["api_total"] == 150

    def test_method_label_stored(self):
        cmp = _build_token_comparison(100, 50, self._api_usage(), "~chars÷4")
        assert cmp["method"] == "~chars÷4"

    def test_zero_api_completion_no_division_error(self):
        cmp = _build_token_comparison(100, 0, self._api_usage(100, 0), "tiktoken")
        assert cmp["pct_compl"] == pytest.approx(0.0)

    # ── _format_token_comparison ──────────────────────────────────────

    def test_format_none_returns_empty(self):
        assert _format_token_comparison(None) == ""

    def test_format_no_api_returns_empty(self):
        cmp = {"has_api": False, "method": "tiktoken"}
        assert _format_token_comparison(cmp) == ""

    def test_format_returns_string(self):
        cmp = _build_token_comparison(100, 50, self._api_usage(100, 50), "tiktoken")
        result = _format_token_comparison(cmp)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_contains_api_values(self):
        cmp = _build_token_comparison(100, 50, self._api_usage(100, 50), "tiktoken")
        result = _format_token_comparison(cmp)
        assert "100" in result
        assert "50" in result

    def test_format_contains_method_label(self):
        cmp = _build_token_comparison(100, 50, self._api_usage(100, 50), "tiktoken")
        result = _format_token_comparison(cmp)
        assert "tiktoken" in result

    def test_format_accurate_shows_checkmark(self):
        cmp = _build_token_comparison(100, 50, self._api_usage(100, 50), "tiktoken")
        result = _format_token_comparison(cmp)
        assert "✅" in result

    def test_format_inaccurate_shows_ruler(self):
        cmp = _build_token_comparison(200, 100, self._api_usage(100, 50), "~chars÷4")
        result = _format_token_comparison(cmp)
        assert "📐" in result

    def test_format_shows_percentage(self):
        cmp = _build_token_comparison(110, 55, self._api_usage(100, 50), "tiktoken")
        result = _format_token_comparison(cmp)
        assert "%" in result
