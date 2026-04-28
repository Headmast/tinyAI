# Developer Assistant — Ассистент разработчика

Встроенный помощник, который отвечает на вопросы о проекте TinyAI, используя RAG-поиск по документации и Git-контекст через MCP.

## Быстрый старт

```bash
# 1. Проиндексировать документацию (README + docs/)
python3 dev_assistant_index.py

# 2. Запустить чат и использовать /help
python3 chat_cli.py
```

В чате:
```
💬 Вы: /help Как устроена архитектура проекта?
💬 Вы: /help Какие MCP-серверы есть?
💬 Вы: /help На какой я ветке?
```

## Компоненты

### 1. Индексация документации (`dev_assistant_index.py`)

Загружает `README.md` + все `.md` файлы из `docs/` в отдельный RAG-индекс `rag_data_dev/`.

```bash
# Стратегия по умолчанию — structure (по заголовкам Markdown)
python3 dev_assistant_index.py

# Другие стратегии
python3 dev_assistant_index.py --strategy fixed_size
python3 dev_assistant_index.py --strategy both

# Свой путь к индексу
python3 dev_assistant_index.py --index-dir my_index/
```

При добавлении/изменении документации — переиндексируйте:
```bash
python3 dev_assistant_index.py
```

### 2. MCP Git Server (`mcp_git_server.py`)

MCP-сервер (stdio, JSON-RPC 2.0), предоставляющий Git-контекст проекта.

**Инструменты:**

| Инструмент | Описание |
|---|---|
| `git_current_branch` | Текущая ветка |
| `git_status` | Изменённые файлы (`git status --short`) |
| `git_diff_summary` | Статистика изменений (`git diff --stat`) |
| `git_log_short` | Последние N коммитов |
| `git_list_files` | Список отслеживаемых файлов |

**Проверка вручную:**
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"git_current_branch","arguments":{}}}' \
  | python3 mcp_git_server.py 2>/dev/null
```

### 3. DevAssistant (`dev_assistant.py`)

Модуль, объединяющий RAG-поиск + MCP Git + LLM для ответов о проекте.

**Использование из кода:**
```python
from dev_assistant import DevAssistant

assistant = DevAssistant(verbose=True)
answer = assistant.answer_help("Как работает RAG в проекте?")
print(answer)
assistant.close()
```

**Параметры конструктора:**

| Параметр | По умолчанию | Описание |
|---|---|---|
| `index_dir` | `rag_data_dev/` | Путь к RAG-индексу документации |
| `model` | из `core/config.py` | LLM-модель для генерации ответов |
| `verbose` | `False` | Подробный вывод (источники, git-контекст) |

### 4. Команда `/help` в `chat_cli.py`

```
/help <вопрос>    — спросить ассистента о проекте
```

При первом вызове инициализирует `DevAssistant` (подключается к MCP Git Server).

## Как это работает

```
Пользователь: /help Как устроен RAG?
         │
         ▼
    ┌─────────────┐
    │  DevAssistant│
    └──┬───────┬──┘
       │       │
       ▼       ▼
  ┌────────┐ ┌──────────────┐
  │RAG     │ │MCP Git Server│
  │Search  │ │(subprocess)  │
  └───┬────┘ └──────┬───────┘
      │             │
      │ top-5 чанков│ ветка + статус
      ▼             ▼
    ┌─────────────────┐
    │   LLM (Cloud.ru)│
    │   system prompt  │
    │   + документация │
    │   + git-контекст │
    │   + вопрос       │
    └────────┬────────┘
             │
             ▼
      Ответ + источники
```

## Требования

- Python 3.9+
- `git` в PATH
- API-ключ (Cloud.ru или OpenAI) в `.env`
- Установленные зависимости: `pip install -r requirements.txt`
- FAISS-индекс (`python3 dev_assistant_index.py` — нужен один раз)
