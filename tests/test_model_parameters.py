#!/usr/bin/env python3
"""
Интеграционные тесты для проверки параметров всех моделей на коротких запросах
"""
import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from llm_cli import (
    get_available_models,
    get_client_for_model,
    get_model_params,
    calculate_cost,
    execute_mode,
    get_available_modes
)

SHORT_TEST_PROMPTS = [
    "Что такое AI?",
    "2+2=?",
    "Назови цвет неба",
    "Столица России?",
    "Привет!"
]

@pytest.fixture(scope="module")
def clients():
    """Создаёт клиенты для всех доступных провайдеров"""
    cloud_api_key = os.getenv("CLOUD_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    clients_dict = {}
    
    if cloud_api_key:
        cloud_url = "https://foundation-models.api.cloud.ru/v1"
        clients_dict["cloud_ru"] = OpenAI(api_key=cloud_api_key, base_url=cloud_url, timeout=60.0)
    
    if openai_api_key:
        clients_dict["openai"] = OpenAI(api_key=openai_api_key, timeout=60.0)
    
    return clients_dict

@pytest.fixture(scope="module")
def available_models_for_testing(clients):
    """Возвращает только те модели, для которых есть API ключи"""
    all_models = get_available_models()
    return {k: v for k, v in all_models.items() if v["provider"] in clients}

class TestModelParameters:
    """Тесты параметров моделей"""
    
    def test_all_models_have_required_params(self):
        """Проверка наличия обязательных параметров у всех моделей"""
        models = get_available_models()
        required_fields = ["name", "prompt_price", "completion_price", "description", "provider", "id"]
        
        for model_id, model_info in models.items():
            for field in required_fields:
                assert field in model_info, f"Model {model_id} missing field: {field}"
    
    def test_temperature_parameter_handling(self):
        """Проверка корректной обработки параметра temperature"""
        test_params = {
            "model": "test",
            "temperature": 0.7,
            "max_completion_tokens": 100,
            "messages": []
        }
        
        # gpt-5-nano не должен иметь temperature
        adapted_nano = get_model_params("gpt-5-nano", test_params.copy())
        assert "temperature" not in adapted_nano
        
        # Остальные модели должны сохранять temperature
        for model in ["gpt-5.4", "gpt-5.4-mini", "zai-org/GLM-4.7-Flash", "zai-org/GLM-4.7"]:
            adapted = get_model_params(model, test_params.copy())
            assert "temperature" in adapted
            assert adapted["temperature"] == 0.7
    
    def test_max_completion_tokens_present(self):
        """Проверка наличия max_completion_tokens во всех режимах"""
        modes = get_available_modes()
        for mode in modes:
            assert "max_completion_tokens" in mode["params"]
            assert mode["params"]["max_completion_tokens"] > 0

class TestModelPricing:
    """Тесты ценообразования моделей"""
    
    def test_gpt54_mini_cheaper_than_gpt54(self):
        """GPT-5.4 Mini должен быть дешевле GPT-5.4"""
        models = get_available_models()
        mini = models["gpt-5.4-mini"]
        full = models["gpt-5.4"]
        
        assert mini["prompt_price"] < full["prompt_price"]
        assert mini["completion_price"] < full["completion_price"]
    
    def test_cost_calculation_accuracy(self):
        """Проверка точности расчёта стоимости"""
        # Тест для GLM (бесплатно)
        cost_glm = calculate_cost(1000, 2000, "zai-org/GLM-4.7-Flash")
        assert cost_glm == 0.0
        
        # Тест для GPT-5.4 Mini
        cost_mini = calculate_cost(1000, 1000, "gpt-5.4-mini")
        expected_mini = (1000 * 0.00075 / 1000) + (1000 * 0.0045 / 1000)
        assert abs(cost_mini - expected_mini) < 0.000001
        
        # Тест для GPT-5.4
        cost_full = calculate_cost(1000, 1000, "gpt-5.4")
        expected_full = (1000 * 0.0025 / 1000) + (1000 * 0.015 / 1000)
        assert abs(cost_full - expected_full) < 0.000001
        
        # Тест для GPT-5 Nano
        cost_nano = calculate_cost(1000, 1000, "gpt-5-nano")
        expected_nano = (1000 * 0.0002 / 1000) + (1000 * 0.00125 / 1000)
        assert abs(cost_nano - expected_nano) < 0.000001

@pytest.mark.integration
class TestShortPrompts:
    """Интеграционные тесты с короткими запросами к реальным API"""
    
    @pytest.mark.parametrize("prompt", SHORT_TEST_PROMPTS[:2])
    def test_glm_flash_short_prompts(self, clients, prompt):
        """Тест GLM-4.7-Flash с короткими запросами"""
        if "cloud_ru" not in clients:
            pytest.skip("CLOUD_API_KEY not available")
        
        model_name = "zai-org/GLM-4.7-Flash"
        client = clients["cloud_ru"]
        modes = get_available_modes(model_name)
        mode = modes[0]  # Режим без ограничений
        
        try:
            response = execute_mode(client, mode, prompt, max_retries=2)
            assert response is not None
            assert response.choices[0].message.content
            assert len(response.choices[0].message.content) > 0
            
            # Проверка usage
            assert response.usage.total_tokens > 0
            assert response.usage.prompt_tokens > 0
            assert response.usage.completion_tokens > 0
            
            # Проверка стоимости (должна быть 0 для GLM)
            cost = calculate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                model_name
            )
            assert cost == 0.0
            
        except Exception as e:
            pytest.fail(f"Failed with prompt '{prompt}': {str(e)}")
    
    @pytest.mark.parametrize("prompt", SHORT_TEST_PROMPTS[:2])
    def test_gpt54_mini_short_prompts(self, clients, prompt):
        """Тест GPT-5.4 Mini с короткими запросами"""
        if "openai" not in clients:
            pytest.skip("OPENAI_API_KEY not available")
        
        model_name = "gpt-5.4-mini"
        client = clients["openai"]
        modes = get_available_modes(model_name)
        mode = modes[2]  # Режим с ограничением длины (100 токенов)
        
        try:
            response = execute_mode(client, mode, prompt, max_retries=2)
            assert response is not None
            assert response.choices[0].message.content
            assert len(response.choices[0].message.content) > 0
            
            # Проверка usage
            assert response.usage.total_tokens > 0
            assert response.usage.prompt_tokens > 0
            assert response.usage.completion_tokens > 0
            
            # Проверка стоимости
            cost = calculate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                model_name
            )
            assert cost > 0  # Должна быть ненулевая стоимость
            assert cost < 0.01  # Но небольшая для короткого запроса
            
        except Exception as e:
            pytest.fail(f"Failed with prompt '{prompt}': {str(e)}")

@pytest.mark.integration
class TestTemperatureVariations:
    """Тесты различных значений температуры"""
    
    @pytest.mark.parametrize("temperature", [0, 0.5, 0.75, 1.0])
    def test_glm_with_temperatures(self, clients, temperature):
        """Тест GLM с разными температурами"""
        if "cloud_ru" not in clients:
            pytest.skip("CLOUD_API_KEY not available")
        
        model_name = "zai-org/GLM-4.7-Flash"
        client = clients["cloud_ru"]
        
        params = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Привет!"}],
            "max_completion_tokens": 100,
            "temperature": temperature
        }
        
        adapted_params = get_model_params(model_name, params)
        
        try:
            # Проверяем, что температура присутствует в параметрах
            assert "temperature" in adapted_params
            assert adapted_params["temperature"] == temperature
            
            # Делаем запрос без streaming для простоты теста
            response = client.chat.completions.create(**adapted_params)
            assert response is not None
            assert response.choices[0].message.content
            assert len(response.choices[0].message.content) > 0
            
        except Exception as e:
            pytest.fail(f"Failed with temperature {temperature}: {str(e)}")
    
    def test_gpt5_nano_ignores_temperature(self, clients):
        """Проверка, что gpt-5-nano игнорирует параметр temperature"""
        if "openai" not in clients:
            pytest.skip("OPENAI_API_KEY not available")
        
        model_name = "gpt-5-nano"
        
        params = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_completion_tokens": 50,
            "temperature": 0.7
        }
        
        adapted_params = get_model_params(model_name, params)
        assert "temperature" not in adapted_params

@pytest.mark.integration
class TestAllModelsBasic:
    """Базовый тест работоспособности всех доступных моделей"""
    
    def test_all_available_models_respond(self, clients, available_models_for_testing):
        """Проверка, что все доступные модели отвечают на простой запрос"""
        test_prompt = "Привет"
        
        for model_id, model_info in available_models_for_testing.items():
            client = clients[model_info["provider"]]
            modes = get_available_modes(model_id)
            # Используем режим с ограничением длины для коротких ответов
            mode = modes[2]  # Режим 3: с ограничением длины (100 токенов)
            
            try:
                response = execute_mode(client, mode, test_prompt, max_retries=2)
                assert response is not None, f"Model {model_id} returned None"
                assert response.choices[0].message.content, f"Model {model_id} returned empty content"
                
                print(f"\n✓ {model_info['name']}: {len(response.choices[0].message.content)} chars")
                
            except Exception as e:
                pytest.fail(f"Model {model_id} ({model_info['name']}) failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
