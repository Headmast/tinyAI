# Локальная LLM как приватный сервис

Этот режим добавляет в TinyAI приватный AI-сервис поверх локальной LLM, рассчитанный на два сценария:

- VPS с Linux и постоянным IP
- домашний сервер в локальной сети или через reverse proxy/VPN

Сервис не требует авторизации и публикует HTTP API для чата.

## Что входит

- `local_llm_service.py` — HTTP-сервис поверх OpenAI-compatible upstream (например, Ollama)
- `local_llm_client.py` — настраиваемый HTTP-клиент
- `dashboard_ui/` — встроенная веб-морда observability (status/tokens/history)
- `demos/run_local_journalist_agent.py` — CLI журналист-агента через локальный сервис
- `demos/check_local_llm_service.py` — smoke-check сети, параллельной нагрузки и лимитов

## Архитектура

```text
Local machine
  -> local_llm_client.py / JournalistAgent
  -> http://SERVER_IP:8000
  -> local_llm_service.py
  -> http://127.0.0.1:11434/v1
  -> Ollama (qwen3:8b)
```

## Развёртывание на VPS

### 1. Подготовить машину

```bash
sudo apt update
sudo apt install -y python3 python3-venv git curl
git clone <repo-url> tinyAI
cd tinyAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Установить и запустить Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b
OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

Если нужен внешний доступ к самой Ollama, используйте отдельный reverse proxy. Для TinyAI это не требуется: сервис ходит к Ollama локально.

### 3. Поднять TinyAI local service

```bash
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

### 4. Проверить с другой машины

```bash
python3 local_llm_client.py --base-url http://SERVER_IP:8000 health
python3 local_llm_client.py --base-url http://SERVER_IP:8000 chat "Привет, ответь одной фразой"

# dashboard
open http://SERVER_IP:8000/dashboard
```

## Развёртывание на домашнем сервере

Подход такой же, но обычно сервис публикуется:

- только в локальную сеть по адресу вроде `http://192.168.1.50:8000`
- или через WireGuard/Tailscale/VPN
- или через reverse proxy с ограничением доступа по сети

Без авторизации сервис лучше не выставлять в открытый интернет напрямую.

## HTTP API

### `GET /health`

Возвращает статус сервиса, модель и лимиты.

### `GET /v1/models`

Возвращает список доступных моделей в OpenAI-compatible формате.

### `POST /chat`

Простой endpoint для клиентских приложений.

Пример запроса:

```json
{
  "prompt": "Сделай краткую выжимку новости",
  "system": "Ты аккуратный редактор",
  "model": "qwen3:8b",
  "temperature": 0.3,
  "max_completion_tokens": 512
}
```

### `POST /v1/chat/completions`

OpenAI-compatible endpoint для уже существующих клиентов проекта.

### Dashboard API

- `GET /dashboard` — HTML-интерфейс
- `GET /dashboard/static/*` — css/js ресурсы
- `GET /dashboard/api/status` — статус сервиса, активные запросы, лимиты
- `GET /dashboard/api/metrics` — счётчики запросов и токенов
- `GET /dashboard/api/history?limit=50` — история запросов и ответов

История включает только запросы, прошедшие через `local_llm_service.py`.

## Базовые ограничения

Сервис применяет три простых ограничения:

- rate limit по IP, окно 60 секунд
- ограничение на размер контекста по грубой оценке токенов
- верхний предел `max_completion_tokens`

Эти ограничения нужны, чтобы локальная модель не деградировала от длинных и слишком частых запросов.

## Что показывает веб-морда

- статус, uptime, текущая модель
- лимиты сервиса (rpm/context/completion)
- активные запросы
- total/success/fail counters
- prompt/completion/total token counters
- rate-limit hits и ошибки
- историю запросов/ответов с latency и status code

История хранится в `logs/dashboard_telemetry.json` и ограничивается переменной `LOCAL_LLM_DASHBOARD_HISTORY_LIMIT`.

## Проверка стабильности

```bash
python3 demos/check_local_llm_service.py \
  --base-url http://SERVER_IP:8000 \
  --parallel 4 \
  --prompt "Сформулируй один факт о частной LLM"
```

Скрипт проверяет:

- доступность по сети
- успешный чат-запрос
- несколько параллельных запросов
- срабатывание базового ограничения контекста

## Журналист-агент через локальный сервис

```bash
python3 demos/run_local_journalist_agent.py \
  --base-url http://SERVER_IP:8000 \
  --model qwen3:8b
```

Команды в REPL:

- `/reset` — сбросить историю
- `/tokens` — показать счётчики токенов
- `/quit` — выйти

## Systemd для VPS

Пример юнита:

```ini
[Unit]
Description=TinyAI Local LLM Service
After=network.target

[Service]
WorkingDirectory=/opt/tinyAI
ExecStart=/opt/tinyAI/.venv/bin/python /opt/tinyAI/local_llm_service.py --host 0.0.0.0 --port 8000 --upstream-base-url http://127.0.0.1:11434/v1 --model qwen3:8b
Restart=always
User=ubuntu

[Install]
WantedBy=multi-user.target
```

## Ограничения текущей реализации

- сервис не поддерживает streaming response
- ограничение контекста считается приближённо, без tokenizer конкретной модели
- аутентификация намеренно отсутствует по условиям задачи