# Задание 7 — Диалоговые сессии с историей контекста

## Что реализовано

### 1. Модуль `news_agent/session_manager.py`

Новый модуль управления диалоговыми сессиями:

| Класс / Константа | Назначение |
|---|---|
| `ConversationSession` | Одна диалоговая сессия с историей сообщений |
| `SessionStorage` | Файловое хранилище сессий в `sessions/` |
| `MODEL_CONTEXT_SIZES` | Размеры контекстных окон по моделям (все = 128K) |
| `CONTEXT_WARN_THRESHOLD` | Порог предупреждения о заполнении контекста (80%) |

### 2. Новые команды в `llm_cli.py`

```
chat                     — начать новую сессию (вход в chat-режим)
chat new [имя]           — начать именованную сессию
chat list                — список всех сохранённых сессий
chat load <id>           — загрузить и продолжить сессию (сохранённую или закрытую)
chat resume <id>         — псевдоним для chat load
chat delete <id>         — удалить сессию
chat info [id]           — подробная информация о сессии + индикатор контекста
```

**Внутри чата:**
```
info                     — использование контекстного окна (с прогресс-баром)
close / q                — закрыть сессию и вернуться в главное меню
```

### 3. Отображение контекста

После **каждого ответа** модели выводится строка вида:

```
🟢 Контекст: 1,234 / 128,000 токенов [████░░░░░░░░░░░░░░░░] 0.96%
   Токены ответа: prompt ~310 | completion ~42
```

Цвет индикатора: `🟢` < 50% → `🟡` < 80% → `🔴` ≥ 80%  
При заполнении ≥ 80% — автоматическое предупреждение.

### 4. Структура хранилища сессий

```
sessions/
    index.json              — индекс всех сессий (метаданные)
    <session_id>.json       — полная сессия с историей сообщений
```

Пример `sessions/<id>.json`:
```json
{
  "session_id": "a1b2c3d4",
  "name": "Чат 24.03 22:15",
  "model": "zai-org/GLM-4.7-Flash",
  "system_prompt": "",
  "messages": [
    {"role": "user", "content": "Привет!"},
    {"role": "assistant", "content": "Здравствуй! Чем могу помочь?"}
  ],
  "created_at": "2026-03-24T22:15:00.000000",
  "updated_at": "2026-03-24T22:16:30.123456",
  "status": "active",
  "token_usage": {
    "prompt_tokens": 12,
    "completion_tokens": 18,
    "total_tokens": 30
  }
}
```

---

## Ключевая концепция: API vs Веб-чат

Это **принципиальное архитектурное отличие**, которое нужно понимать при работе с LLM через API.

### Веб-чат (ChatGPT, Claude Web, Gemini)

```
Пользователь → браузер → сервер [хранит историю] → модель
              ↑                       ↓
              ←←←←←← ответ ←←←←←←←←
```

- Браузер отправляет **только новое сообщение**
- Сервер сам ведёт историю диалога
- Пользователь видит непрерывный разговор
- Под капотом сервер автоматически формирует полный `messages[]`

### API-запрос (прямой доступ)

```
Приложение → API [stateless] → модель
  ↑ Отправляет: весь messages[] явно
```

- Каждый API-вызов **полностью независим (stateless)**
- Модель **не помнит** предыдущих запросов
- Чтобы модель «помнила» — нужно при **каждом вызове** передавать **весь массив `messages[]`** начиная с первого сообщения
- Именно поэтому реализуем собственное хранилище `sessions/`

### Практический пример

**Веб-чат** — второй запрос пользователя (браузер отправляет только новое):
```
POST /chat
{"message": "А сколько сторон у куба?"}
```

**API** — второй запрос к OpenAI (нужно отправить всю историю):
```python
client.chat.completions.create(
    model="gpt-...",
    messages=[
        {"role": "user",      "content": "Что такое квадрат?"},
        {"role": "assistant", "content": "Квадрат — плоская фигура с 4 равными сторонами."},
        {"role": "user",      "content": "А сколько сторон у куба?"},  # новое сообщение
    ]
)
```

### Где это реализовано в коде

`_chat_send()` в `llm_cli.py` (строки ~254–274):
```python
def _chat_send(client, session, user_input):
    session.add_user_message(user_input)
    messages = session.get_messages_for_api()  # полная история
    
    params = {
        "model": session.model,
        "messages": messages,   # <-- ВСЯ история при каждом запросе
        ...
    }
    response_text, reasoning = stream_response(client, params)
    session.add_assistant_message(response_text)  # сохраняем ответ
    ...
```

---

## Оценка токенов

Точный подсчёт токенов возможен только через специальный tokenizer (библиотека `tiktoken` для OpenAI). В данной реализации используется **приближённая оценка**:

```
tokens ≈ total_characters / 4
```

Это корректная аппроксимация для английского текста (≈ 4 символа = 1 токен).  
Для русского текста реальный расход токенов может быть **в 1.5–2 раза выше** из-за кириллицы в UTF-8 токенизации.

Для точного подсчёта можно подключить `tiktoken`:
```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
tokens = len(enc.encode(text))
```

---

## Логирование сессий

Каждая сессия автоматически сохраняется в `sessions/<id>.json`:
- **После каждого сообщения** — автосохранение
- **При закрытии** (`close`) — обновляется статус на `"closed"`
- **При перезагрузке** (`chat load`) — статус возвращается в `"active"`

Файл `sessions/index.json` содержит лёгкие метаданные для быстрого вывода списка.

---

## Тесты

```bash
# Тесты session manager
pytest test_sessions.py -v

# Тесты CLI включая chat-функции
pytest test_llm_cli.py -v

# Все тесты
pytest -v
```

**Покрытие `test_sessions.py`:**
- `TestModelContextSizes` — константы контекстных окон
- `TestConversationSessionCreation` — создание сессий
- `TestConversationSessionMessages` — добавление сообщений
- `TestTokenUsage` — оценка токенов
- `TestContextInfo` — информация о контексте
- `TestContextBar` — визуальный индикатор
- `TestSessionClose` — закрытие сессий
- `TestSerialization` — сериализация / десериализация
- `TestSessionStorage` — хранилище (14 тестов)
- `TestIntegration` — полный цикл жизни сессии

**Новые тесты в `test_llm_cli.py`:**
- `TestChatSessionFunctions` — CLI-функции (`_cmd_chat_list`, `_cmd_chat_delete`, `_print_session_info`)
- `TestSessionManagerImport` — корректность импортов

---

## Быстрый старт

```bash
# Запустить CLI
python3 llm_cli.py

# Начать новый чат
news-agent> chat

# Начать именованный чат
news-agent> chat new Мой Python диалог

# Список сессий
news-agent> chat list

# Продолжить сессию
news-agent> chat load a1b2c3d4

# Удалить сессию
news-agent> chat delete a1b2c3d4
```
