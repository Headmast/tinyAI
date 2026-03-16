import os
import pytest
from unittest.mock import patch, MagicMock
from llm_cli import main

@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}):
        yield

@pytest.fixture
def mock_openai_client():
    with patch('llm_cli.OpenAI') as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        
        response = MagicMock()
        response.choices[0].message.content = "Test response"
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 20
        response.usage.total_tokens = 30
        
        client_instance.chat.completions.create.return_value = response
        
        yield client_instance

def test_api_key_missing():
    with patch.dict(os.environ, {}, clear=True):
        with patch('builtins.print') as mock_print:
            main()
            assert any("OPENAI_API_KEY not found" in str(call) for call in mock_print.call_args_list)

def test_successful_request(mock_env, mock_openai_client):
    with patch('builtins.input', side_effect=['Hello', 'quit']):
        with patch('builtins.print') as mock_print:
            main()
            
            assert mock_openai_client.chat.completions.create.called
            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]['model'] == 'gpt-3.5-turbo'
            assert call_args[1]['max_tokens'] == 500
            assert call_args[1]['temperature'] == 0.7

def test_token_counting(mock_env, mock_openai_client):
    with patch('builtins.input', side_effect=['Test message', 'quit']):
        with patch('builtins.print') as mock_print:
            main()
            
            printed_output = ' '.join(str(call) for call in mock_print.call_args_list)
            assert 'Tokens used:' in printed_output
            assert 'prompt:' in printed_output
            assert 'completion:' in printed_output
            assert 'Cost:' in printed_output

def test_empty_input(mock_env, mock_openai_client):
    with patch('builtins.input', side_effect=['', 'quit']):
        main()
        assert not mock_openai_client.chat.completions.create.called

def test_error_handling(mock_env):
    with patch('llm_cli.OpenAI') as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        client_instance.chat.completions.create.side_effect = Exception("API Error")
        
        with patch('builtins.input', side_effect=['Test', 'quit']):
            with patch('builtins.print') as mock_print:
                main()
                printed_output = ' '.join(str(call) for call in mock_print.call_args_list)
                assert 'Error:' in printed_output
