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

## Локальный приватный LLM-сервис

TinyAI может работать не только с облачными API, но и с приватной локальной моделью через HTTP-сервис.

### Сценарий

- VPS или домашний сервер поднимает Ollama и `local_llm_service.py`
- сервис публикует HTTP API без авторизации
- локальная машина подключается через `local_llm_client.py` или через журналист-агент

### 1. Поднять Ollama на сервере

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### 2. Поднять HTTP-сервис TinyAI

```bash
cd tinyAI
source .venv/bin/activate

python3 local_llm_service.py \
	--host 0.0.0.0 \
	--port 8000 \
	--upstream-base-url http://127.0.0.1:11434/v1 \
	--model qwen3:8b \
	--max-requests-per-minute 30 \
	--max-context-tokens 12000 \
	--max-completion-tokens 2048
```

### 3. Проверить доступность по сети

```bash
python3 local_llm_client.py --base-url http://SERVER_IP:8000 health
python3 local_llm_client.py --base-url http://SERVER_IP:8000 chat "Напиши 2 факта о локальных LLM"

# веб-морда наблюдаемости
open http://SERVER_IP:8000/dashboard
```

### 4. Проверить стабильность и лимиты

```bash
python3 demos/check_local_llm_service.py --base-url http://SERVER_IP:8000 --parallel 4
```

Проверка делает три вещи:

- убеждается, что `/health` и `/chat` доступны по сети
- отправляет несколько параллельных запросов
- проверяет базовое ограничение контекста

### 5. Веб-морда (status/tokens/history)

После запуска сервиса откройте:

`http://SERVER_IP:8000/dashboard`

Что показывает интерфейс:

- текущий статус сервиса, модель и лимиты
- активные запросы, количество успешных/ошибочных вызовов
- суммарные токены: prompt/completion/total
- историю запросов и ответов (preview), latency, status code

История в UI берётся только из трафика, прошедшего через `local_llm_service.py`.

### 6. Запуск журналист-агента с локальной машины

```bash
python3 demos/run_local_journalist_agent.py \
	--base-url http://SERVER_IP:8000 \
	--model qwen3:8b
```

### Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LOCAL_LLM_BASE_URL` | `http://localhost:8000` | Адрес TinyAI local LLM service |
| `LOCAL_LLM_MODEL` | `qwen3:8b` | Локальная модель |
| `LOCAL_LLM_TIMEOUT` | `120` | Таймаут HTTP-запросов |
| `LOCAL_LLM_RATE_LIMIT` | `30` | Запросов в минуту на IP |
| `LOCAL_LLM_MAX_CONTEXT` | `12000` | Максимальный контекст в токенах (грубая оценка) |
| `LOCAL_LLM_MAX_COMPLETION` | `2048` | Верхний предел completion tokens |
| `LOCAL_LLM_DASHBOARD_HISTORY_LIMIT` | `200` | Максимум записей в истории dashboard |

Подробная инструкция по VPS и домашнему серверу: `docs/LOCAL_LLM_SERVICE.md`.
