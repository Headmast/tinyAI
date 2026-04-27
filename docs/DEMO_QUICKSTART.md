# Local LLM Service — Demo Quickstart

Полное руководство: запуск сервера, CLI-клиент, веб-дашборд, агент-журналист.

---

## 1. Предварительные требования

### Установить Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Скачать модель

```bash
ollama pull qwen3:8b
```

### Запустить Ollama (если не запущен автоматически)

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

Проверка: `curl http://127.0.0.1:11434/api/tags` — должен вернуть список моделей.

---

## 2. Запуск сервиса

### Минимальный запуск (всё по умолчанию)

```bash
python3 local_llm_service.py
```

Сервер стартует на `http://0.0.0.0:8000`, подключается к Ollama на `http://127.0.0.1:11434/v1`, модель `qwen3:8b`.

### Полный запуск с параметрами

```bash
python3 local_llm_service.py \
  --host 0.0.0.0 \
  --port 8000 \
  --upstream-base-url http://127.0.0.1:11434/v1 \
  --model qwen3:8b \
  --timeout 120 \
  --max-requests-per-minute 30 \
  --max-context-tokens 4096 \
  --max-completion-tokens 1024 \
  --dashboard-history-limit 100
```

### Через переменные окружения

```bash
export LOCAL_LLM_HOST=0.0.0.0
export LOCAL_LLM_PORT=8000
export LOCAL_LLM_MODEL=qwen3:8b
export LOCAL_LLM_RATE_LIMIT=30
python3 local_llm_service.py
```

### Ожидаемый вывод при старте

```
[2026-04-27 10:00:00] INFO  Local LLM Service starting
[2026-04-27 10:00:00] INFO  Upstream: http://127.0.0.1:11434/v1  Model: qwen3:8b
[2026-04-27 10:00:00] INFO  Rate limit: 30 req/min per IP
[2026-04-27 10:00:00] INFO  Context limit: 4096 tokens  Completion limit: 1024 tokens
[2026-04-27 10:00:00] INFO  Dashboard: http://0.0.0.0:8000/dashboard
[2026-04-27 10:00:00] INFO  Serving on http://0.0.0.0:8000
```

---

## 3. CLI-клиент (`local_llm_client.py`)

### Проверить статус сервера

```bash
python3 local_llm_client.py health
```

```json
{"status": "ok", "model": "qwen3:8b", "upstream": "http://127.0.0.1:11434/v1"}
```

### Список доступных моделей

```bash
python3 local_llm_client.py models
```

### Отправить запрос

```bash
python3 local_llm_client.py chat "Что такое квантовые вычисления?"
```

### Подключиться к удалённому серверу

```bash
python3 local_llm_client.py --base-url http://SERVER_IP:8000 chat "Привет!"
```

### Все параметры CLI

```
python3 local_llm_client.py [OPTIONS] COMMAND [ARGS]

OPTIONS:
  --base-url URL     Адрес сервиса (default: http://127.0.0.1:8000)
  --timeout N        Таймаут запроса в секундах (default: 120)

COMMANDS:
  health             Проверка статуса
  models             Список моделей
  chat PROMPT        Отправить текстовый запрос
```

---

## 4. Прямые HTTP-запросы (curl)

### Health check

```bash
curl http://localhost:8000/health
```

### Список моделей (OpenAI-совместимый формат)

```bash
curl http://localhost:8000/v1/models
```

### Чат (простой endpoint)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Расскажи о чёрных дырах", "system": "Ты — научный журналист"}'
```

### Чат (OpenAI-совместимый endpoint)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:8b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ]
  }'
```

---

## 5. Веб-дашборд

Откройте браузер: **http://localhost:8000/dashboard**

### Что показывает дашборд

| Карточка | Описание |
|---|---|
| Status | Статус сервиса (running / error) и имя модели |
| Requests | Всего запросов / активных прямо сейчас |
| Tokens | Суммарно prompt + completion токенов |
| Rate Limits | Количество отклонённых запросов по rate-limit |

### История запросов

Таблица с последними N запросами (по умолчанию 50):
- Время запроса
- IP клиента
- Snippet промпта (первые 80 символов)
- Статус (OK / error)
- Токены использовано

Дашборд обновляется автоматически каждые 3 секунды.

### API дашборда (для интеграции)

```bash
# Статус сервиса
curl http://localhost:8000/dashboard/api/status

# Метрики токенов и rate-limit
curl http://localhost:8000/dashboard/api/metrics

# История запросов (последние 20)
curl http://localhost:8000/dashboard/api/history?limit=20
```

---

## 6. Агент-журналист в CLI-режиме

Интерактивный REPL поверх локального сервиса:

```bash
python3 demos/run_local_journalist_agent.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3:8b
```

### Пример сессии

```
Local Journalist Agent (model: qwen3:8b @ http://127.0.0.1:8000)
Commands: /reset  /tokens  /quit
──────────────────────────────────────────────────────────────
You> Напиши вступление к статье о колонизации Марса

Agent> Красная планета всегда манила человечество. На горизонте XXI века...

You> Добавь цитату учёного

Agent> По словам астробиолога Сары Джонсон, «Марс — это наш следующий шаг...»

You> /tokens
Tokens used this session: prompt=1240  completion=380  total=1620

You> /reset
Context cleared.

You> /quit
Bye!
```

### Параметры запуска

```
--base-url URL      Адрес сервиса (default: http://127.0.0.1:8000)
--model MODEL       Имя модели (default: qwen3:8b)
--max-tokens N      Лимит токенов на ответ (default: 1024)
```

---

## 7. Smoke-тест и проверка стабильности

```bash
python3 demos/check_local_llm_service.py \
  --base-url http://127.0.0.1:8000 \
  --parallel 4 \
  --prompt "Кратко о квантовых компьютерах"
```

Запускает N параллельных запросов, проверяет:
- Health endpoint
- Список моделей
- Параллельный чат (4 потока)
- Контекстный лимит
- Rate-limit

Ожидаемый вывод:
```
[OK] health → {"status": "ok", ...}
[OK] models → 1 model(s) listed
[OK] parallel chat (4 workers) → all 4 succeeded
[OK] context limit enforced
[OK] rate limit enforced (got 429 after N requests)
All checks passed.
```

---

## 8. Запуск на VPS / удалённом сервере

```bash
# На сервере
python3 local_llm_service.py --host 0.0.0.0 --port 8000

# С локальной машины
python3 local_llm_client.py --base-url http://SERVER_IP:8000 health
python3 local_llm_client.py --base-url http://SERVER_IP:8000 chat "Hello!"

# Dashboard
open http://SERVER_IP:8000/dashboard
```

> **Важно**: сервис не имеет авторизации. При открытом доступе из интернета рекомендуется ограничить порт фаерволом или nginx reverse proxy.

---

## 9. Использование из Python-кода

```python
from local_llm_client import LocalLLMHttpClient, LocalLLMClientConfig

client = LocalLLMHttpClient(LocalLLMClientConfig(base_url="http://127.0.0.1:8000"))

# Простой чат
response = client.chat(
    prompt="Напиши заголовок статьи о ИИ",
    system="Ты — редактор новостного издания",
)
print(response["content"])

# OpenAI-совместимый вызов
result = client.chat_completions(
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    model="qwen3:8b",
)
print(result["choices"][0]["message"]["content"])
```

---

## 10. Запуск тестов

```bash
pytest tests/test_local_llm_service.py -v
```

Тесты покрывают: health, models, chat, параллельный чат, контекстный лимит, rate-limit, дашборд (index, metrics/history, active_requests).

```
tests/test_local_llm_service.py::test_health PASSED
tests/test_local_llm_service.py::test_models PASSED
tests/test_local_llm_service.py::test_chat PASSED
tests/test_local_llm_service.py::test_parallel_stability PASSED
tests/test_local_llm_service.py::test_context_limit PASSED
tests/test_local_llm_service.py::test_rate_limit PASSED
tests/test_local_llm_service.py::test_dashboard_index PASSED
tests/test_local_llm_service.py::test_dashboard_metrics_and_history PASSED
tests/test_local_llm_service.py::test_dashboard_status_active_requests PASSED
9 passed
```
