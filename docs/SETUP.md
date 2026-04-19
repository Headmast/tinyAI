# Установка и настройка TinyAI

## Требования

- **Python:** 3.9+
- **ОС:** macOS, Linux, Windows (WSL рекомендуется)
- **API-ключ:** Cloud.ru Foundation Models или OpenAI

## Установка

```bash
# 1. Клонировать и перейти в директорию
cd tinyAI

# 2. Создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. (опционально) Установить dev-зависимости
pip install -r requirements-dev.txt
# или
pip install -e ".[dev]"
```

## Настройка .env

Создайте файл `.env` в корне проекта:

```bash
# --- Обязательно (хотя бы один ключ) ---

# Cloud.ru Foundation Models (основной провайдер)
CLOUD_API_KEY=ваш_ключ_cloud_ru

# OpenAI (для embeddings в RAG, опционально)
OPENAI_API_KEY=ваш_ключ_openai

# --- Опционально ---

# Base URL (по умолчанию Cloud.ru)
# BASE_URL=https://foundation-models.api.cloud.ru/v1

# Модель по умолчанию
# DEFAULT_MODEL=zai-org/GLM-4.7
```

### Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `CLOUD_API_KEY` | — | API-ключ Cloud.ru (обязателен для LLM) |
| `OPENAI_API_KEY` | — | API-ключ OpenAI (для embeddings в RAG) |
| `BASE_URL` | `https://foundation-models.api.cloud.ru/v1` | Endpoint API |
| `DEFAULT_MODEL` | `zai-org/GLM-4.7` | Модель по умолчанию |
| `DEFAULT_FAST_MODEL` | `zai-org/GLM-4.7-Flash` | Быстрая модель |

## Проверка установки

```bash
# Проверить что тесты проходят
python3 -m pytest tests/ -m "not api and not integration" -q

# Или через Makefile
make test
```

## Типичные проблемы

### `ModuleNotFoundError: No module named 'openai'`
```bash
pip install -r requirements.txt
```

### `CLOUD_API_KEY не найден`
Проверьте что файл `.env` существует в корне проекта и содержит ключ.

### Тесты с маркером `api` падают
Эти тесты требуют реального API-ключа. Запускайте без них:
```bash
python3 -m pytest tests/ -m "not api"
```

### `tiktoken` медленно загружается
Первый импорт tiktoken загружает модель из интернета. Последующие вызовы кэшируются.
