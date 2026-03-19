import os
import pytest
from unittest.mock import patch, MagicMock
from llm_cli import (
    get_available_modes, 
    get_available_models, 
    calculate_cost, 
    count_words,
    execute_mode,
    execute_meta_prompting
)

@pytest.fixture
def mock_client():
    client = MagicMock()
    response = MagicMock()
    response.choices[0].message.content = "Test response from model"
    response.usage.prompt_tokens = 15
    response.usage.completion_tokens = 25
    response.usage.total_tokens = 40
    client.chat.completions.create.return_value = response
    return client

class TestModels:
    def test_get_available_models_count(self):
        models = get_available_models()
        assert len(models) == 3, "Should have 3 models (2 GLM + 1 OpenAI)"
    
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
