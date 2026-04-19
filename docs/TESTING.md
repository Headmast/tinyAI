# Руководство по тестированию

## Запуск тестов

```bash
# Все тесты без API (рекомендуется)
python3 -m pytest tests/ -m "not api and not integration" -v

# Через Makefile
make test

# С покрытием кода
make test-cov

# Конкретный файл
python3 -m pytest tests/test_rag_chunker.py -v

# Все тесты (включая API — нужен ключ)
python3 -m pytest tests/ -v
```

## Маркеры pytest

| Маркер | Описание | Требует API? |
|--------|----------|-------------|
| `unit` | Чистые юнит-тесты | Нет |
| `integration` | Файловые операции | Нет |
| `api` | Реальные API-вызовы | Да |
| `e2e` | End-to-end | Да |
| `slow` | Медленные (> 5 сек) | Зависит |

## Структура тестов (22 файлов)

### Unit-тесты (моки, без API)

| Файл | Тестирует |
|------|-----------|
| `test_analytics.py` | Аналитика: категоризация, UsageTracker |
| `test_article_fsm.py` | FSM статей: фазы, переходы, состояние |
| `test_context_compressor.py` | Компрессия контекста, статистика |
| `test_context_strategies.py` | Стратегии: sliding, sticky_facts, branching |
| `test_core.py` | **НОВЫЙ** — core.config, core.persistence |
| `test_intent_router.py` | **НОВЫЙ** — классификация намерений |
| `test_journalist_agent.py` | Инварианты, prompt injection |
| `test_journalist_fsm.py` | Workflow, переходы, pause/resume |
| `test_llm_cli.py` | CLI: режимы, модели, стоимость |
| `test_mcp_orchestrator.py` | Registry, Router, мульти-сервер |
| `test_new_features.py` | PipelineTemplates, IntentRouter |
| `test_news_agent.py` | Роли, инструменты, pipeline |
| `test_personalization.py` | UserProfile, ProfileManager |
| `test_pipeline_executor.py` | 4-серверный MCP pipeline |
| `test_rag_chunker.py` | **НОВЫЙ** — chunker: размер, overlap |
| `test_rag_citations.py` | Парсер цитат, edge cases |
| `test_sessions.py` | ConversationSession, SessionStorage |
| `test_token_counter.py` | TokenCounter, Budget, Tracker |
| `test_usage_tracker.py` | RequestRecord, UsageTracker |

### Тесты с API (маркер `@pytest.mark.api`)

| Файл | Тестирует |
|------|-----------|
| `test_model_parameters.py` | Параметры реальных моделей |
| `test_rag.py` | Embedder, Search (имеет и unit-тесты) |
| `test_rag_agent.py` | RagAgent с реальными запросами (имеет и unit-тесты) |

### Smoke-тесты (ручной запуск, не pytest)

| Файл | Описание |
|------|----------|
| `smoke_api_log.py` | Один API-вызов GLM-4.7, полный вывод |
| `smoke_gpt54_temperature.py` | GPT-5.4 с 4 температурами |
| `smoke_temperature.py` | GLM-4.7-Flash температуры |
| `smoke_mcp_server_day17.py` | MCP-сервер e2e через subprocess |

## Мок-стратегия

Большинство тестов используют `unittest.mock`:

```python
from unittest.mock import MagicMock, patch

# Мок OpenAI-клиента
mock_client = MagicMock()
mock_client.chat.completions.create.return_value = MagicMock(
    choices=[MagicMock(message=MagicMock(content="ответ"))]
)
```

Общие фикстуры доступны в `tests/conftest.py`:
- `mock_openai_client` — готовый мок LLM-клиента
- `mock_env_keys` — тестовые API-ключи в env
- `temp_data_dir` — временная директория
- `sample_messages` — типовой диалог

Фабрики данных в `tests/factories.py`:
- `make_message()`, `make_conversation()`
- `make_journalist_task()`, `make_rag_document()`

## Покрытие кода

```bash
make test-cov
# Отчёт: htmlcov/index.html
```

## Написание новых тестов

1. Файл называется `test_*.py` в директории `tests/`
2. Классы — `Test*`, методы — `test_*`
3. API-зависимые тесты помечаются `@pytest.mark.api`
4. Используй фикстуры из `conftest.py`
5. Моки вместо реальных API-вызовов для unit-тестов
