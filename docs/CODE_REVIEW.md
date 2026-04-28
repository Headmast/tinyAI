# AI Code Review — Автоматическое ревью кода

Автоматический AI-ревьюер анализирует diff (PR или локальные изменения) и выдаёт структурированный отчёт: баги, архитектурные проблемы, рекомендации.

## Быстрый старт

```bash
# 1. Проиндексировать документацию (если ещё не сделано)
python3 dev_assistant_index.py

# 2. Ревью незакоммиченных изменений
python3 review_cli.py

# 3. Ревью PR (текущая ветка vs main)
python3 review_cli.py --base-branch main

# 4. Или в чате
python3 chat_cli.py
# > /review
# > /review main
# > /review staged
```

## Компоненты

### 1. CodeReviewer (`code_reviewer.py`)

Основной модуль: получает diff через MCP Git Server, ищет контекст через RAG, отправляет на анализ GPT-5-nano.

**Использование из кода:**
```python
from code_reviewer import CodeReviewer

reviewer = CodeReviewer(verbose=True)

# PR-ревью (текущая ветка vs main)
report = reviewer.review_pr(base_branch="main")
print(report)

# Staged изменения
report = reviewer.review_staged()

# Рабочая директория
report = reviewer.review_working()

# Произвольный diff
report = reviewer.review_diff(diff_text, changed_files="...")

reviewer.close()
```

**Параметры конструктора:**

| Параметр | По умолчанию | Описание |
|---|---|---|
| `index_dir` | `rag_data_dev/` | Путь к RAG-индексу документации |
| `model` | `gpt-5-nano` | LLM-модель для анализа |
| `verbose` | `False` | Подробный вывод |

### 2. CLI (`review_cli.py`)

Standalone скрипт для терминала и CI/CD.

```bash
# Ревью рабочей директории
python3 review_cli.py

# PR-ревью
python3 review_cli.py --base-branch main

# Только staged
python3 review_cli.py --staged

# Из файла с diff (для CI)
python3 review_cli.py --diff-file pr.diff

# Сохранить результат
python3 review_cli.py --base-branch main --output review.md

# Подробный вывод
python3 review_cli.py -v --base-branch main
```

### 3. Команда `/review` в `chat_cli.py`

```
/review           — ревью незакоммиченных изменений
/review staged    — ревью staged изменений
/review main      — ревью PR (vs main)
/review develop   — ревью PR (vs develop)
```

### 4. GitHub Action (`.github/workflows/code_review.yml`)

Автоматически запускается при открытии или обновлении PR.

**Настройка:**
1. Добавьте секрет `OPENAI_API_KEY` в Settings → Secrets → Actions
2. PR будет автоматически получать комментарий с ревью

**Что делает:**
- Checkout кода с полной историей
- Установка Python 3.11 и зависимостей
- Запуск `review_cli.py --base-branch origin/<base_branch>`
- Публикация результата как комментарий к PR

## Как это работает

```
PR / локальные изменения
         │
         ▼
    ┌──────────────┐
    │ CodeReviewer  │
    └──┬────────┬──┘
       │        │
       ▼        ▼
  ┌────────┐ ┌──────────────┐
  │RAG     │ │MCP Git Server│
  │Search  │ │  git_diff    │
  │(docs)  │ │  git_status  │
  └───┬────┘ └──────┬───────┘
      │             │
      │ документация│ полный diff
      ▼             ▼
    ┌─────────────────┐
    │  GPT-5-nano     │
    │  system prompt   │
    │  + документация  │
    │  + diff          │
    └────────┬────────┘
             │
             ▼
    Структурированное ревью:
    🐛 Потенциальные баги
    🏗️ Архитектурные проблемы
    💡 Рекомендации
    📊 Общая оценка
```

## MCP Git Server — новые инструменты

Добавлены в `mcp_git_server.py`:

| Инструмент | Описание |
|---|---|
| `git_diff` | Полный diff (unified format). Параметры: `base_branch`, `staged` |
| `git_show_file` | Содержимое файла из HEAD |

## Тесты

```bash
# Запустить тесты ревьюера
pytest tests/test_code_reviewer.py -v

# Все тесты проекта
pytest
```

## Требования

- Python 3.9+
- `git` в PATH
- `OPENAI_API_KEY` в `.env` (для GPT-5-nano)
- Установленные зависимости: `pip install -r requirements.txt`
- RAG-индекс: `python3 dev_assistant_index.py` (один раз)
