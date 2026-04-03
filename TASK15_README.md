# День 15 — Архитектурный анализ, ревью кода и оценка качества

## Содержание

1. [Архитектурная схема проекта](#1-архитектурная-схема-проекта)
2. [Структура модулей](#2-структура-модулей)
3. [Ревью кода по модулям](#3-ревью-кода-по-модулям)
4. [Сильные стороны](#4-сильные-стороны)
5. [Слабые стороны и технический долг](#5-слабые-стороны-и-технический-долг)
6. [Риски](#6-риски)
7. [Рекомендации](#7-рекомендации)

---

## 1. Архитектурная схема проекта

```
┌──────────────────────────────────────────────────────────────────────┐
│                       TinyAI  v6.0  (15 дней курса)                  │
│               Cloud.ru Foundation Models API (OpenAI-compat.)         │
│               Models: zai-org/GLM-4.7 · zai-org/GLM-4.7-Flash        │
└──────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════╗
║  memory_agent/                                                       ║
║                                                                      ║
║   MemoryAgent ──────────► MemoryManager                             ║
║      chat(input) → str        ├─ ShortTermMemory  (deque, N=20)     ║
║      _extract_memory_block()  ├─ WorkingMemory    (dict, in-RAM)    ║
║      _apply_memory_updates()  └─ LongTermMemory   (JSON file)       ║
║                                                                      ║
║   PersonalizedAgent ────────► MemoryManager + UserProfile           ║
║      profile инжектируется в system prompt                           ║
║      ProfileManager — пресеты + пользовательские профили            ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║  news_agent/  v6.0                                                   ║
║                                                                      ║
║  ┌─────────────────────┐   ┌──────────────────────────────────────┐  ║
║  │  NewsPipeline       │   │  AgentLoop (ReAct)                   │  ║
║  │  5 LLM-вызовов:     │   │  Думай → Действуй → Наблюдай        │  ║
║  │  Planner            │   │  max_iterations, token budget        │  ║
║  │  Researcher         │   │  ToolDispatcher ─► function calling  │  ║
║  │  Writer             │   └──────────────────────────────────────┘  ║
║  │  Editor             │                                              ║
║  │  SEO                │   ┌──────────────────────────────────────┐  ║
║  └─────────────────────┘   │  FSMAgent                            │  ║
║                             │  planning → execution                │  ║
║  ┌───────────────────────────────────────────────────────────────┐ ║
║  │  ConversationSession              SessionStorage               │ ║
║  │                                                                │ ║
║  │  get_messages_for_api():                                      │ ║
║  │   1. context_strategy  (приоритет)                            │ ║
║  │   2. ContextCompressor (если нет стратегии)                   │ ║
║  │   3. полная история    (fallback)                             │ ║
║  │                                                                │ ║
║  │  Стратегии:                     Компрессия:                   │ ║
║  │  SlidingWindowStrategy          LLM суммаризирует             │ ║
║  │  StickyFactsStrategy            старые сообщения → summary    │ ║
║  │  BranchingStrategy              токены экономятся на ~40-60%  │ ║
║  └───────────────────────────────────────────────────────────────┘ ║
║                                                                      ║
║  PostStorage ── OutputFormatter (md/html/tg/json/plain)              ║
║  TokenCounter (tiktoken + fallback chars÷4)                          ║
║  UsageTracker (logs/usage_stats.json) · StrategyLogger (.jsonl)      ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║  journalist_agent/                                                   ║
║                                                                      ║
║  JournalistAgent                                                     ║
║    │                                                                 ║
║    ├─ Layer 1: InvariantStore.pre_check()                           ║
║    │    keyword matching + stem (первые 6 символов каждого слова)   ║
║    │    → отказ БЕЗ вызова LLM                                      ║
║    │                                                                 ║
║    └─ Layer 2: LLM + инварианты в system prompt                     ║
║         → парсинг ОТКАЗ:[INV-XXX] из ответа                        ║
║                                                                      ║
║  InvariantStore ── Invariant ── ViolationResult                      ║
║  invariants.json: INV-001 (гимнастика), INV-002 (криминал)          ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║  llm_cli.py  (104 KB, ~2600 строк)                                   ║
║  Единая CLI-точка входа:                                             ║
║  chat · session · generate · agent · api_simple · api_advanced       ║
║  batch · compare · temperature · stats · compression · strategies    ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 2. Структура модулей

| Модуль | Файлов | Строк кода | Тестов | Версия |
|--------|--------|-----------|--------|--------|
| `memory_agent/` | 5 | ~900 | ~0 выделенных | — |
| `news_agent/` | 15 | ~4200 | 10 файлов | 6.0.0 |
| `journalist_agent/` | 4 | ~560 | 38 (100%) | — |
| `llm_cli.py` | 1 | ~2600 | ~100 (test_llm_cli.py) | — |
| Корень (demo/run) | ~15 | ~2800 | — | — |

---

## 3. Ревью кода по модулям

### 3.1 `memory_agent/`

**`memory.py`** — три слоя памяти, хорошо разделены по ответственности.

```
ShortTermMemory  ─ deque(maxlen=20)  ─ сброс на каждый запуск
WorkingMemory    ─ dict in RAM        ─ volatile, не персистирован
LongTermMemory   ─ long_term.json     ─ персистирован, растёт неограниченно
```

Проблемы:
- `WorkingMemory` теряется при перезапуске — но в промпте передаётся как «постоянная память». Несоответствие ожиданий.
- `LongTermMemory.save()` пишет **весь файл целиком** при каждом обновлении. При большом объёме — медленно и рискованно (нет атомарной записи через tmp-файл).
- `_extract_memory_block()` в `agent.py` парсит JSON из ответа LLM. Если LLM забывает обернуть блок — обновление молча не происходит.
- Нет политики вытеснения для `LongTermMemory` — список записей растёт вечно.

**`profile.py`** — хорошо структурированный `dataclass`. `build_prompt_section()` генерирует корректный текст для system prompt.

---

### 3.2 `news_agent/pipeline.py`

**Сильные места:**
- Чистый 5-шаговый конвейер, каждый шаг изолирован
- Retry-логика с `time.sleep(2)` на каждую попытку
- Стриминг с отображением `reasoning_content` (GLM thinking)

**Проблемы:**

```python
# pipeline.py:303-308 — оценка токенов через chars÷4, хотя TokenCounter уже есть
def _update_token_estimate(self, content: str, reasoning: str, prompt: str) -> None:
    estimated_prompt = len(prompt) // 4      # <-- неточно
    estimated_completion = (len(content) + len(reasoning)) // 4
```
Надо: использовать `TokenCounter` из того же проекта.

```python
# pipeline.py:281-288 — парсинг JSON только по первому ``` блоку
if "```json" in text:
    start = text.index("```json") + 7
    end = text.index("```", start)   # <-- str.index() бросает ValueError если нет закрывающего ```
```
`str.index()` вместо `str.find()` — возможен необработанный `ValueError`.

---

### 3.3 `news_agent/agent.py` (AgentLoop)

**Сильные места:**
- Корректная реализация ReAct: Thought / Action / Observation
- Token budget enforcement: агент останавливается при превышении лимита
- Чёткое извлечение `FINAL_POST:` из ответа

**Проблемы:**
- `_extract_final_post()` использует простой `str.split("FINAL_POST:")` — нет fallback если LLM не использует маркер
- Нет механизма возобновления прерванного агентного цикла
- Связь с `FSMState` декларативная: FSM создаётся, но агент не проверяет допустимость переходов

---

### 3.4 `news_agent/storage.py`

```python
# storage.py:107-133 — search_similar() как Jaccard similarity по словам
similarity = overlap / max(len(topic_words), len(title_words))
```
Именование вводит в заблуждение: это не семантический поиск, а пересечение мешков слов. Параметр `threshold=0.7` выглядит как «70% семантической похожести» — на самом деле требует совпадения 70% слов заголовка, что почти никогда не срабатывает.

Нет файловой блокировки при конкурентной записи:
```python
index = self._read_index()   # чтение
index.insert(0, {...})        # модификация
self._write_index(index)      # запись — race condition при concurrent use
```

---

### 3.5 `news_agent/context_strategies.py`

**`StickyFactsStrategy.on_user_message()`** — скрытый риск:
```python
except Exception:
    pass  # Молча продолжаем
```
Любая ошибка (сетевая, JSON, etc.) проглатывается. Факты просто не обновляются, но пользователь не знает об этом.

**`BranchingStrategy`** — хорошая идея, но `on_assistant_message()` нужно вызывать явно. `ConversationSession` не делает этого — стратегия частично нерабочая без ручной интеграции.

---

### 3.6 `news_agent/context_compressor.py`

```python
# context_compressor.py:376
if self.model == "gpt-5-nano":
    del params["temperature"]
```
Хрупкая проверка по имени модели. Нужна общая конфигурация: `MODELS_WITHOUT_TEMPERATURE = {...}`.

---

### 3.7 `journalist_agent/`

**Самый чистый модуль.** Чёткая ответственность каждого класса, 38/38 тестов.

Проблемы:
- `_history` в `JournalistAgent` растёт неограниченно — нет интеграции с `SlidingWindowStrategy` или `ContextCompressor`
- `STEM_LEN = 6` — глобальная константа без возможности настройки per-invariant
- Нет API для добавления/отключения инвариантов в runtime (только JSON-файл)
- `get_total_tokens()` молча возвращает нули если API не вернул `usage`

---

### 3.8 `llm_cli.py`

- **104 KB / ~2600 строк в одном файле** — God Object
- Содержит 15+ команд, каждая со своим обработчиком, конфигурацией, форматированием
- Нет иерархической структуры команд (argparse subparsers есть, но логика не разделена по файлам)
- Дублирование кода: паттерн `create_client() → call API → print` повторяется 10+ раз

---

## 4. Сильные стороны

### Архитектурные

- **Прогрессивная сложность** — каждый день добавляет один паттерн поверх предыдущего. Проект — живая документация курса.
- **OpenAI-совместимый клиент** — смена провайдера (Cloud.ru → OpenAI → Anthropic) требует правки только `base_url` и `api_key`.
- **Чистое разделение ответственностей** в `news_agent/`: хранилище, форматтер, роли, инструменты — независимые модули.
- **Три независимые стратегии управления контекстом** с единым интерфейсом `ContextStrategy` + реестром `STRATEGY_REGISTRY`.
- **Graceful fallback** в `TokenCounter`: tiktoken → chars÷4. Код не падает при отсутствии зависимости.
- **Двухуровневая защита инвариантов** в `journalist_agent/` — pre-check без LLM экономит токены, LLM-уровень ловит косвенные нарушения.

### Качество кода

- Docstrings на всех публичных классах и методах
- `dataclass` везде где уместно
- `from __future__ import annotations` для forward references
- Сериализация/десериализация через `to_dict()` / `from_dict()` во всех stateful классах
- `__all__` в `news_agent/__init__.py`
- Тесты для `journalist_agent/` полностью изолированы (mock LLM-клиент)

### Наблюдаемость

- `UsageTracker` пишет каждый запрос в JSON-журнал с временем ответа, токенами, стоимостью
- `StrategyLogger` пишет посменные метрики в JSONL
- `format_context_bar()` — визуальный индикатор заполнения контекста
- `pipeline_stats` в результате `NewsPipeline.run()` — время каждого шага

---

## 5. Слабые стороны и технический долг

### Критические несоответствия

| # | Файл | Проблема |
|---|------|----------|
| 1 | `pipeline.py:303` | Оценка токенов chars÷4 вместо уже готового `TokenCounter` |
| 2 | `pipeline.py:284` | `str.index()` без fallback → возможен `ValueError` |
| 3 | `storage.py:107` | `search_similar()` — не семантический поиск, несмотря на параметр `threshold` |
| 4 | `context_strategies.py:259` | `except Exception: pass` в `StickyFacts` скрывает ошибки |
| 5 | `context_compressor.py:376` | Хардкод имени модели `"gpt-5-nano"` как проверка |

### Технический долг

- **`llm_cli.py`** (104 KB) — God Object, нужна декомпозиция на подмодули
- **`WorkingMemory`** не персистируется, хотя используется как «долгосрочная» память в промпте
- **`LongTermMemory`** растёт без ограничений — нет политики вытеснения старых записей
- **`journalist_agent/_history`** тоже растёт без ограничений — нет интеграции с `ContextCompressor`
- **`BranchingStrategy.on_assistant_message()`** требует явного вызова, который нигде не сделан автоматически

### Беспорядок в корне проекта

- 9 файлов `TASK*_README.md` без индекса
- Логи и JSON-артефакты сравнений (`chuck_norris_*`, `model_comparison_*`, `comparison_output.log`) — не в `.gitignore`
- 6 demo-скриптов (`demo_*.py`) без тестов

---

## 6. Риски

| Уровень | Риск | Место |
|---------|------|-------|
| 🔴 Высокий | Race condition при конкурентной записи индекса | `PostStorage.save()`, `SessionStorage.save()` |
| 🔴 Высокий | `llm_cli.py` слишком велик, любое изменение ломает соседние команды | `llm_cli.py` |
| 🟡 Средний | Неограниченный рост памяти агентов при длинных сессиях | `MemoryAgent`, `JournalistAgent` |
| 🟡 Средний | Тихое проглатывание ошибок в `StickyFacts` делает отладку невозможной | `context_strategies.py:259` |
| 🟡 Средний | tiktoken cl100k_base используется для GLM — токены считаются приближённо для нелатинских текстов | `token_counter.py` |
| 🟢 Низкий | Хардкод имён моделей в 8+ местах — при смене модели придётся менять везде | Повсеместно |
| 🟢 Низкий | `requirements.txt` с `openai>=2.29.0` — мажорное обновление может сломать код | `requirements.txt` |

---

## 7. Рекомендации

### Приоритет 1 — быстрые исправления

```python
# 1. pipeline.py: заменить chars÷4 на TokenCounter
from news_agent.token_counter import TokenCounter
self._counter = TokenCounter(model=self.model)

def _update_token_estimate(self, content, reasoning, prompt):
    msgs = [{"role": "user", "content": prompt},
            {"role": "assistant", "content": content + reasoning}]
    data = self._counter.count_messages(msgs)
    self._token_usage["prompt_tokens"] += data["by_role"].get("user", 0)
    ...
```

```python
# 2. pipeline.py: str.index() → str.find() с fallback
if "```json" in text:
    start = text.find("```json") + 7
    end = text.find("```", start)
    if end == -1:
        end = len(text)
    text = text[start:end].strip()
```

```python
# 3. context_compressor.py: убрать хардкод имени модели
MODELS_NO_TEMPERATURE = {"gpt-5-nano", "o1", "o3"}
if self.model in MODELS_NO_TEMPERATURE:
    del params["temperature"]
```

```python
# 4. context_strategies.py: StickyFacts — логировать ошибки
import logging
_log = logging.getLogger(__name__)
except Exception as exc:
    _log.warning("StickyFacts fact extraction failed: %s", exc)
```

### Приоритет 2 — архитектурные улучшения

1. **`JournalistAgent`** — добавить `SlidingWindowStrategy` или `max_history` параметр:
   ```python
   def __init__(self, ..., max_history: int = 20):
       self._max_history = max_history
   
   def _trim_history(self):
       if len(self._history) > self._max_history * 2:
           self._history = self._history[-self._max_history * 2:]
   ```

2. **`LongTermMemory`** — добавить `max_entries` и политику FIFO:
   ```python
   def add(self, key, value, max_entries=500):
       self._store[key] = value
       if len(self._store) > max_entries:
           oldest = next(iter(self._store))
           del self._store[oldest]
   ```

3. **`PostStorage` / `SessionStorage`** — атомарная запись через tmp:
   ```python
   import os
   tmp = self.index_file.with_suffix(".tmp")
   with open(tmp, "w", encoding="utf-8") as f:
       json.dump(index, f, ...)
   os.replace(tmp, self.index_file)  # атомарная замена
   ```

4. **`llm_cli.py`** — декомпозиция:
   ```
   llm_cli/
     __init__.py
     commands/
       chat.py
       session.py
       generate.py
       agent.py
       stats.py
   ```

### Приоритет 3 — качество проекта

- Добавить `pyproject.toml` с pinned-версиями зависимостей
- Добавить `.gitignore` для `*.log`, `*_comparison_*.json`, `posts/`, `sessions/`, `logs/`
- Создать единый `README.md` с индексом всех задач
- Добавить pre-commit hook: `pytest` + `flake8` / `ruff`
- Покрыть `memory_agent/` тестами (сейчас 0 выделенных тестов)

---

## Итоговая оценка

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Архитектурная чистота | 8/10 | Хорошее разделение, есть God Object (`llm_cli.py`) |
| Качество кода | 7/10 | Docstrings хороши, есть несогласованности |
| Тестовое покрытие | 6/10 | `journalist_agent` — 100%, `memory_agent` — ~0% |
| Наблюдаемость | 8/10 | UsageTracker, StrategyLogger, context bar |
| Надёжность | 5/10 | Race conditions, тихие ошибки, нет атомарных записей |
| Масштабируемость | 6/10 | Файловое хранилище, нет БД, всё O(n) на чтение индекса |
| Документация | 7/10 | Много README, нет центрального индекса |

**Общая оценка: 7/10** — хорошая учебная база с продуманными паттернами, 
требует hardening для production-использования.
