"""
Общие фикстуры и конфигурация для тестов TinyAI.
"""

import os
import pytest
from unittest.mock import MagicMock, patch


# --- Фикстуры для LLM ---

@pytest.fixture
def mock_openai_client():
    """Мок OpenAI-совместимого клиента с настроенным chat.completions.create."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(
            content="Тестовый ответ от LLM",
            tool_calls=None,
        ))],
        usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )
    return client


@pytest.fixture
def mock_env_keys():
    """Устанавливает тестовые API-ключи в переменные окружения."""
    with patch.dict(os.environ, {
        "CLOUD_API_KEY": "test-cloud-key",
        "OPENAI_API_KEY": "test-openai-key",
        "BASE_URL": "https://test-api.example.com/v1",
    }):
        yield


# --- Фикстуры для данных ---

@pytest.fixture
def temp_data_dir(tmp_path):
    """Временная директория для тестовых данных."""
    return tmp_path


@pytest.fixture
def sample_messages():
    """Типовая история диалога для тестов."""
    return [
        {"role": "system", "content": "Ты — полезный ассистент."},
        {"role": "user", "content": "Привет!"},
        {"role": "assistant", "content": "Здравствуйте! Чем могу помочь?"},
        {"role": "user", "content": "Расскажи о Python."},
    ]


@pytest.fixture
def sample_article_text():
    """Пример текста статьи для RAG-тестов."""
    return (
        "Искусственный интеллект (ИИ) — это область информатики, "
        "занимающаяся созданием интеллектуальных машин, которые работают "
        "и реагируют как люди. Машинное обучение является подразделом ИИ "
        "и фокусируется на разработке алгоритмов, которые могут обучаться "
        "на данных и делать прогнозы."
    )
