# Support Assistant — Ассистент поддержки пользователей

AI-ассистент, который отвечает на вопросы пользователей о продукте, используя RAG-поиск по FAQ и контекст пользователя/тикета через MCP.

## Быстрый старт

```bash
# 1. Проиндексировать FAQ
python3 support_index.py

# 2. Запустить чат и использовать /support
python3 chat_cli.py
```

В чате:
```
💬 Вы: /support Почему не работает авторизация?
💬 Вы: /support --user user_1 Почему не работает авторизация?
💬 Вы: /support --ticket ticket_1 Помогите с моей проблемой
```

## Компоненты

### 1. FAQ — база знаний (`support_data/faq.md`)

Markdown-документ с описанием типовых функций и проблем продукта, структурированный по разделам:

- **Авторизация и аутентификация** — логин, 2FA, сброс пароля, блокировки
- **Управление подпиской и оплата** — тарифы, платежи, отмена
- **API-доступ** — ключи, лимиты, ошибки (401, 429, 500)
- **Интеграции** — вебхуки, Telegram, Slack, GitHub
- **Данные и хранение** — загрузка, экспорт, бэкапы
- **Уведомления** — email, push, настройка
- **Командная работа** — роли, права, приглашения
- **Производительность** — таймауты, лимиты тарифов

### 2. Данные пользователей и тикетов

- `support_data/users.json` — профили пользователей (user_id, plan, features, status)
- `support_data/tickets.json` — тикеты поддержки (ticket_id, user_id, messages, status, priority)

### 3. MCP Support Server (`mcp_support_server.py`)

MCP-сервер (stdio, JSON-RPC 2.0), предоставляющий контекст пользователей и тикетов.

**Инструменты:**

| Инструмент | Описание |
|---|---|
| `get_user` | Профиль пользователя по user_id |
| `get_user_tickets` | Список тикетов пользователя |
| `get_ticket` | Полный тикет с историей сообщений |
| `search_tickets` | Поиск тикетов по ключевому слову |
| `get_active_tickets` | Все открытые/в работе тикеты |

**Проверка вручную:**
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_user","arguments":{"user_id":"user_1"}}}' \
  | python3 mcp_support_server.py 2>/dev/null
```

### 4. Индексация FAQ (`support_index.py`)

Загружает `support_data/faq.md` в RAG-индекс `rag_data_support/`.

```bash
# Стратегия по умолчанию — structure (по заголовкам Markdown)
python3 support_index.py

# Другие стратегии
python3 support_index.py --strategy fixed_size
python3 support_index.py --strategy both

# Свой путь к индексу
python3 support_index.py --index-dir my_support_index/
```

### 5. SupportAssistant (`support_assistant.py`)

Модуль, объединяющий RAG-поиск + MCP Support + LLM для ответов на вопросы.

**Использование из кода:**
```python
from support_assistant import SupportAssistant

assistant = SupportAssistant(verbose=True)

# Простой вопрос (только FAQ)
answer = assistant.answer("Как сбросить пароль?")
print(answer)

# С контекстом пользователя
answer = assistant.answer("Почему не работает API?", user_id="user_1")
print(answer)

# С контекстом тикета
answer = assistant.answer("Помогите", ticket_id="ticket_1")
print(answer)

assistant.close()
```

**Параметры конструктора:**

| Параметр | По умолчанию | Описание |
|---|---|---|
| `index_dir` | `rag_data_support/` | Путь к RAG-индексу FAQ |
| `model` | из `core/config.py` | LLM-модель для генерации ответов |
| `verbose` | `False` | Подробный вывод (источники, контекст) |

### 6. Команда `/support` в `chat_cli.py`

```
/support <вопрос>                 — ответ из FAQ
/support --user <id> <вопрос>     — с учётом профиля пользователя
/support --ticket <id> <вопрос>   — с учётом данных тикета
```

При первом вызове инициализирует `SupportAssistant` (подключается к MCP Support Server).

### 7. MCPStdioClient (`mcp_stdio_client.py`)

Универсальный MCP-клиент для stdio-серверов. Используется и DevAssistant, и SupportAssistant.

```python
from mcp_stdio_client import MCPStdioClient

with MCPStdioClient("mcp_support_server.py") as client:
    result = client.call_tool("get_user", {"user_id": "user_1"})
    print(result)
```

## Как это работает

```
Пользователь: /support --user user_1 Почему не работает авторизация?
         │
         ▼
    ┌──────────────────┐
    │ SupportAssistant │
    └──┬───────┬───────┘
       │       │
       ▼       ▼
  ┌────────┐ ┌─────────────────────┐
  │RAG     │ │MCP Support Server   │
  │Search  │ │(users + tickets)    │
  │(FAQ)   │ │(subprocess/stdio)   │
  └───┬────┘ └──────────┬──────────┘
      │                  │
      │ top-5 FAQ чанков │ профиль + тикеты user_1
      ▼                  ▼
    ┌────────────────────────────┐
    │   LLM                      │
    │   system prompt (агент     │
    │     поддержки)             │
    │   + FAQ-документация       │
    │   + профиль пользователя   │
    │   + вопрос                 │
    └───────────┬────────────────┘
                │
                ▼
         Персонализированный ответ
         + источники FAQ
```

## Типовые функции и проблемы

### Типовые функции системы:
1. **Авторизация** — логин, регистрация, 2FA, OAuth, сброс пароля
2. **Подписка** — тарифы Free/Pro/Enterprise, оплата, лимиты
3. **API** — ключи, rate limits, версионирование
4. **Интеграции** — вебхуки, Telegram, Slack
5. **Данные** — загрузка, экспорт, бэкапы
6. **Уведомления** — email, push, настройки
7. **Команда** — роли, права доступа

### Типовые проблемы:
1. Не работает авторизация (токен, блокировка, 2FA)
2. Ошибки API (401, 429, 500)
3. Проблемы с оплатой (карта, промокод)
4. Интеграция не работает (SSL, вебхук)
5. Медленная работа (таймауты)
6. Потеря данных (удаление, конфликт)
7. Лимиты тарифа (запросы, feature gating)

## Рефакторинг

В рамках этой задачи был извлечён общий `MCPStdioClient` из `dev_assistant.py`:
- **До:** `DevAssistant` содержал внутренний класс `_GitMCPClient` с логикой subprocess + JSON-RPC
- **После:** общий `MCPStdioClient` в `mcp_stdio_client.py` используется и `DevAssistant`, и `SupportAssistant`
- Это устраняет дублирование кода и упрощает добавление новых MCP-серверов

## Тесты

```bash
# Unit-тесты (без API)
pytest tests/test_support_assistant.py -v -m unit

# Интеграционные тесты (subprocess)
pytest tests/test_mcp_support_server.py -v -m integration

# Все тесты
pytest tests/test_support_assistant.py tests/test_mcp_support_server.py -v
```

## Требования

- Python 3.9+
- API-ключ (Cloud.ru или OpenAI) в `.env`
- Установленные зависимости: `pip install -r requirements.txt`
- FAISS-индекс FAQ: `python3 support_index.py` (нужен один раз)
