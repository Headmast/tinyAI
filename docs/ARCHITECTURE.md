# TinyAI — Архитектурная документация

> Версия документа: v8.0 — Task 25+: Рефакторинг и RAG-улучшения  
> Проект: учебный фреймворк LLM-агентов (Cloud.ru / OpenAI-совместимый API)

---

## Содержание

1. [Обзор проекта](#1-обзор-проекта)
2. [Структура репозитория](#2-структура-репозитория)
3. [Архитектурная схема](#3-архитектурная-схема)
4. [Модуль `memory_agent`](#4-модуль-memory_agent)
5. [Модуль `news_agent`](#5-модуль-news_agent)
6. [Модуль `journalist_agent`](#6-модуль-journalist_agent)
7. [Точка входа `llm_cli.py`](#7-точка-входа-llm_clipy)
8. [Сквозные паттерны](#8-сквозные-паттерны)
9. [Конфигурация и зависимости](#9-конфигурация-и-зависимости)
10. [Примеры использования](#10-примеры-использования)
11. [Известные ограничения](#11-известные-ограничения)
12. [RAG v2: Rewrite и Reranking](#12-rag-v2-rewrite-и-reranking)

---

## 1. Обзор проекта

TinyAI — учебный фреймворк для изучения архитектурных паттернов LLM-агентов. Три независимых модуля демонстрируют разные подходы к управлению памятью, контекстом и редакционными ограничениями:

| Модуль | Паттерн | Ключевая идея |
|--------|---------|---------------|
| `memory_agent` | Layered Memory | Три слоя памяти с явным управлением маршрутизацией |
| `news_agent` | Pipeline + ReAct + FSM + MCP | Несколько стратегий генерации и MCP-бекэнд для логов/памяти |
| `journalist_agent` | Invariant Enforcement | Двухуровневая защита от нарушения редакционных правил |
| `mcp_server` | MCP stdio | JSON-RPC 2.0 бэкенд для логов, памяти, статистики |

**Используемый API:** Cloud.ru Foundation Models (OpenAI-совместимый).  
**Модели по умолчанию:** `zai-org/GLM-4.7`, `zai-org/GLM-4.7-Flash`.

---

## 2. Структура репозитория

```
tinyAI/
│
├── memory_agent/               # Агент с трёхслойной памятью
│   ├── __init__.py
│   ├── memory.py               # ShortTermMemory, WorkingMemory, LongTermMemory, MemoryManager
│   │                           # + утилиты extract_memory_block(), apply_memory_updates()
│   ├── agent.py                # MemoryAgent — основной диалоговый агент
│   ├── personalized_agent.py   # PersonalizedAgent — агент с профилем пользователя
│   └── profile.py              # UserProfile, ProfileManager, BUILTIN_PROFILES
│
├── news_agent/                 # Агент генерации новостей (v7.0)
│   ├── __init__.py
│   ├── agent.py                # AgentLoop — ReAct-агент (автономный режим)
│   ├── mcp_bridge.py           # MCPBridge — мост к mcp_server.py через stdio
│   ├── pipeline.py             # NewsPipeline — 5-шаговый конвейер
│   ├── fsm_agent.py            # ArticleFSMAgent — агент с конечным автоматом
│   ├── roles.py                # Роли агентов и профили типов постов
│   ├── tools.py                # Инструменты (function calling) + ToolDispatcher
│   ├── storage.py              # PostStorage — файловое хранилище постов
│   ├── session_manager.py      # ConversationSession, SessionStorage
│   ├── context_compressor.py   # ContextCompressor — LLM-суммаризация контекста
│   ├── context_strategies.py   # SlidingWindow, StickyFacts, Branching стратегии
│   ├── token_counter.py        # TokenCounter, TokenBudget, MODELS_NO_TEMPERATURE
│   ├── formatter.py            # OutputFormatter — Markdown/HTML/Telegram/JSON
│   ├── usage_tracker.py        # UsageTracker — JSONL-лог запросов к API
│   ├── strategy_logger.py      # StrategyLogger — метрики стратегий контекста
│   └── fsm_state.py            # TaskPhase, TaskState, TaskStateStorage
│
├── journalist_agent/           # Агент-журналист с инвариантами
│   ├── __init__.py
│   ├── agent.py                # JournalistAgent — двухуровневая проверка инвариантов
│   ├── invariants.py           # Invariant, InvariantStore, ViolationResult
│   └── invariants.json         # Конфигурация редакционных инвариантов
│
├── mcp_server.py               # MCP-сервер (протокол: JSON-RPC 2.0 / stdio)
│   └── инструменты: list_logs, read_log, search_logs,
│             list_memory, read_memory, get_usage_stats
├── mcp_client.py               # MCP-клиент (handshake + вывод инструментов + демо)
│
├── llm_cli.py                  # CLI-точка входа (все команды)
├── run_memory_agent.py         # Демо-скрипт: memory_agent
├── run_journalist_agent.py     # Демо-скрипт: journalist_agent
├── test_news_agent.py          # Тесты news_agent (pytest)
├── test_journalist_agent.py    # Тесты journalist_agent (pytest)
├── test_context_strategies.py  # Тесты стратегий контекста (pytest)
├── test_context_compressor.py  # Тесты компрессора (pytest)
├── requirements.txt            # Зависимости Python
├── TASK16_README.md            # Задание 16: MCP-клиент и интеграция
├── TASK14_README.md            # Задание 14: journalist_agent
├── TASK15_README.md            # Задание 15: анализ и ревью кода
└── ARCHITECTURE.md             # Этот файл
```

---

## 3. Архитектурная схема

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              llm_cli.py                                 │
│          CLI-точка входа: chat / session / generate / agent / api       │
└────────┬──────────────┬─────────────────────────────┬───────────────────┘
         │              │                             │
         ▼              ▼                             ▼
┌─────────────┐  ┌──────────────────────┐   ┌──────────────────────┐
│ memory_agent│  │     news_agent       │   │  journalist_agent    │
│             │  │                      │   │                      │
│ MemoryAgent │  │  ┌──────────────┐    │   │  JournalistAgent     │
│     │       │  │  │  AgentLoop   │    │   │     │                │
│     ▼       │  │  │  (ReAct)     │    │   │     ▼                │
│ MemoryManager│  │  └──────┬───────┘    │   │  InvariantStore     │
│  ├ ShortTerm│  │         │            │   │  (pre_check + LLM)  │
│  ├ Working  │  │  ┌──────┴───────┐    │   └──────────────────────┘
│  └ LongTerm │  │  │NewsPipeline  │    │
│  (JSON-файл)│  │  │ (5 шагов)    │    │
│             │  │  └──────┬───────┘    │
│ PersonalizedAgent         │           │
│  + UserProfile│  │  ┌──────┴───────┐    │
└─────────────┘  │  │ArticleFSMAgent│    │
                 │  │ (FSM: plan→   │    │
                 │  │  exec→valid)  │    │
                 │  └───────────────┘    │
                 │                      │
                 │  ─────────────────── │
                 │  Общая инфраструктура │
                 │  ┌───────────────────┤
                 │  │ PostStorage       │  posts/*.json, *.md
                 │  │ SessionStorage    │  sessions/*.json
                 │  │ TokenCounter      │  tiktoken / ~chars÷4
                 │  │ ContextCompressor │  LLM-суммаризация
                 │  │ ContextStrategies │  SlidingWindow / StickyFacts / Branching
                 │  │ UsageTracker      │  usage.jsonl
                 │  │ OutputFormatter   │  MD / HTML / Telegram / JSON
                 │  └───────────────────┘
                 └──────────────────────┘

                        │
                        ▼
              ┌─────────────────────┐
              │  OpenAI-совместимый │
              │  API (Cloud.ru)     │
              │  zai-org/GLM-4.7    │
              │  zai-org/GLM-4.7-Flash│
              └─────────────────────┘
```

---

## 4. Модуль `memory_agent`

### 4.1 Концепция

Агент с **явной трёхслойной моделью памяти**. Каждый ответ LLM анализируется — модель сама решает, что нужно сохранить в рабочую или долговременную память, добавляя JSON-блок вида ` ```memory ... ``` ` в конец ответа.

```
Пользователь → MemoryAgent.chat()
    │
    ├── [1] add_message("user") → ShortTermMemory
    ├── [2] build_system_prompt() ← LongTermMemory + WorkingMemory
    ├── [3] LLM-вызов с полным контекстом
    ├── [4] extract_memory_block() → парсим ```memory блок
    ├── [5] apply_memory_updates() → обновляем Working + LongTerm
    └── [6] add_message("assistant") → ShortTermMemory
```

### 4.2 Классы `memory.py`

#### `ShortTermMemory`

Текущий диалог в формате `messages[]` для API.

```python
class ShortTermMemory:
    max_messages: int = 40       # лимит сообщений (старые удаляются)
    messages: List[Dict]         # список {role, content, timestamp}

    def add_message(role: str, content: str) -> None
    def get_messages_for_api() -> List[Dict]   # добавляет system-prompt первым
    def get_last_n(n: int = 5) -> List[Dict]
    def clear() -> None
    def _trim() -> None          # O(1) — slice вместо while+pop(0)
```

**Поведение при переполнении:** `messages = messages[-max_messages:]` — без перебора.

#### `WorkingMemory`

Данные текущей задачи. Очищается при `set_task()`. **Не персистентна** — живёт только в памяти процесса.

```python
class WorkingMemory:
    current_task: str
    facts: List[str]            # дедупликация: add_fact() проверяет наличие
    goals: List[str]
    intermediate: List[Dict]    # {label, data, timestamp}
    context: Dict[str, Any]

    def set_task(task: str, goals: List[str] = None) -> None   # сбрасывает факты/цели/context
    def add_fact(fact: str) -> None                            # дедупликация
    def add_facts(facts: List[str]) -> None
    def add_intermediate(label: str, data: Any) -> None
    def set_context(key: str, value: Any) -> None
    def get_summary() -> str                                   # для инъекции в промпт
```

#### `LongTermMemory`

Персистентное хранилище в `memory_data/long_term.json`. Разделено на три категории.

```python
class LongTermMemory:
    max_decisions: int = 100    # лимит записей (старые вытесняются)
    max_knowledge: int = 200

    # Профиль
    def set_profile(key: str, value: Any) -> None
    def get_profile(key: str, default: Any = None) -> Any
    def get_full_profile() -> Dict

    # Решения
    def add_decision(decision: str, reasoning: str = "") -> None  # с авто-вытеснением
    def get_decisions(last_n: int = None) -> List[Dict]

    # Знания
    def add_knowledge(topic: str, content: str, source: str = "") -> None  # с авто-вытеснением
    def search_knowledge(query: str) -> List[Dict]  # keyword-поиск по topic и content

    # Утилиты
    def get_summary() -> str    # последние 3 решения + 5 знаний для промпта
    def clear() -> None
    def dump() -> Dict
```

**Атомарная запись:** `_save()` пишет во временный `.tmp`-файл, затем `os.replace()`.

#### `MemoryManager`

Координатор всех трёх слоёв. Единственный публичный интерфейс для агента.

```python
class MemoryManager:
    short_term: ShortTermMemory
    working: WorkingMemory
    long_term: LongTermMemory

    def build_system_prompt(base_prompt: str) -> str
    # Собирает: base + профиль + LongTerm summary + Working summary

    def get_full_state() -> Dict      # для отладки
    def save_state_to_file(path: str) -> None
```

#### Модульные утилиты (не методы класса)

Выделены в отдельные функции для устранения дублирования между `MemoryAgent` и `PersonalizedAgent`:

```python
def extract_memory_block(raw_text: str) -> Tuple[str, Optional[Dict]]
# Парсит ```memory {...} ``` из ответа LLM
# Returns: (visible_text, memory_dict_or_None)

def apply_memory_updates(manager: MemoryManager, data: Dict) -> None
# Применяет распарсенный блок к менеджеру памяти
```

### 4.3 `MemoryAgent`

```python
class MemoryAgent:
    def __init__(
        client: OpenAI,
        model: str = "zai-org/GLM-4.7",
        memory_manager: MemoryManager = None,
        verbose: bool = True,
        max_completion_tokens: int = 4000,
        temperature: float = 0.7,
    )

    def chat(user_input: str) -> str         # основной метод
    def reset_working_memory() -> None       # смена задачи
    def reset_all() -> None                  # полный сброс
    def get_memory_state() -> Dict
    def get_token_usage() -> Dict
```

### 4.4 `PersonalizedAgent`

Расширяет `MemoryAgent` поддержкой профиля пользователя. Профиль инжектируется в `system_prompt` при каждом запросе и дублируется в `LongTermMemory.profile`.

```python
class PersonalizedAgent:
    def set_profile(profile: UserProfile) -> None   # горячая смена профиля
    def clear_profile() -> None
    def get_profile_info() -> Dict
```

#### `UserProfile` (profile.py)

```python
@dataclass
class UserProfile:
    name: str
    role: str
    tone: str                      # formal | casual | technical | creative
    response_style: str
    preferred_format: str          # prose | bullet_points | code | mixed
    expertise_areas: List[str]
    constraints: List[str]
    language: str = "ru"

    def build_prompt_section() -> str   # форматирует профиль для system prompt
```

Встроенные профили (`BUILTIN_PROFILES`): `developer`, `journalist`, `student`, `manager`.

---

## 5. Модуль `news_agent`

Три независимых режима генерации новостного контента:

| Класс | Файл | Режим |
|-------|------|-------|
| `AgentLoop` | `agent.py` | Автономный ReAct-агент с инструментами |
| `NewsPipeline` | `pipeline.py` | Детерминированный 5-шаговый конвейер |
| `ArticleFSMAgent` | `fsm_agent.py` | Агент с конечным автоматом и паузой/продолжением |

### 5.1 `AgentLoop` — ReAct-агент

**Паттерн:** Reason → Act (tool call) → Observe → Repeat.

```
run(task) →
  while iteration < max_iterations:
    [1] Проверить бюджет токенов (TokenBudget.check_overflow)
    [2] Вызов LLM с TOOL_DEFINITIONS (12 инструментов: 6 локальных + 6 MCP)
    [3] Если FINAL_POST: в ответе → завершение
    [4] Если tool_calls → ToolDispatcher.dispatch()
          ├─ локальный инструмент → обработчик
          └─ MCP-инструмент → MCPBridge.call_tool() → mcp_server.py
    [5] Добавить observation в messages
    [6] Следующая итерация
```

```python
class AgentLoop:
    def __init__(
        client: OpenAI,
        model: str = "zai-org/GLM-4.7-Flash",
        storage: PostStorage = None,
        verbose: bool = True,
        max_iterations: int = 12,
        max_completion_tokens: int = 4000,
        context_limit: int = 128_000,
        tracker: UsageTracker = None,
    )

    def run(task: str) -> Dict:
    # Returns:
    #   final_post, saved_post_id, iterations,
    #   token_usage, token_history, elapsed_ms
```

**Доступные инструменты** (`tools.py`), всего 12 шт.:

**Локальные (6):**

| Инструмент | Описание |
|------------|----------|
| `save_post` | Сохранить пост в `PostStorage` |
| `load_post_history` | Список последних N постов |
| `analyze_topic` | Анализ темы (ключевые слова, тональность) |
| `format_post` | Конвертация в Markdown/HTML/Telegram |
| `get_post_by_id` | Загрузить конкретный пост |
| `check_duplicate` | Поиск похожих постов (Jaccard similarity) |

**MCP-инструменты (6) — проксируются через `MCPBridge` → `mcp_server.py`:**

| Инструмент | Описание |
|------------|----------|
| `list_logs` | Список файлов логов разговоров с метаданными |
| `read_log` | Содержимое конкретного лога разговора |
| `search_logs` | Поиск текста по всем логам |
| `list_memory` | Список файлов долгосрочной памяти агента |
| `read_memory` | Содержимое файла памяти |
| `get_usage_stats` | Токены, стоимость, число разговоров |

### 5.1.1 `MCPBridge` — мост к MCP-серверу

`MCPBridge` реализует паттерн **ленивого подключения**: сервер запускается как подпроцесс только при первом вызове инструмента.

```
ToolDispatcher.dispatch(tool_name, args)
      │
      ├─ tool_name in MCP_TOOLS?
      │     │
      │     └─ MCPBridge.call_tool(name, args)
      │           │
      │           ├─ [1-й раз] Popen(mcp_server.py)
      │           │            → initialize → notifications/initialized
      │           │
      │           └─ request("tools/call", {name, arguments})
      │                    stdin ─── JSON-RPC 2.0 ──→ stdout
      │
      └─ иначе → локальный _handler(**args)
```

```python
class MCPBridge:
    def call_tool(name: str, arguments: dict) -> str
    # Запускает сервер при необходимости, выполняет MCP-рукопожатие,
    # вызывает инструмент и возвращает текстовый результат.
    # Ошибки возвращаются как строки "[MCP] ...", не бросают исключений.

    def close() -> None
    # Закрывает stdin подпроцесса и ждёт завершения сервера.
    # Вызывается автоматически в AgentLoop.run() после завершения задачи.
```

**Протокол:** MCP stdio v2024-11-05, транспорт JSON-RPC 2.0 (одна строка = одно сообщение).  
**Зависимости:** только stdlib (`subprocess`, `json`, `sys`). Не требует `mcp` пакета.

### 5.2 `NewsPipeline` — детерминированный конвейер

5 последовательных LLM-вызовов, каждый с отдельной ролью:

```
Planner  →  Researcher  →  Writer  →  Editor  →  SEO
  │              │            │           │          │
plan.json   research.json  post.txt  edited.json  seo.json
```

```python
class NewsPipeline:
    def __init__(
        client: OpenAI,
        model: str = "zai-org/GLM-4.7-Flash",
        verbose: bool = True,
        max_retries: int = 3,
    )

    def run(
        topic: str,
        post_type: str = None,       # breaking | analysis | tutorial | opinion | interview
        extra_context: str = None,
    ) -> Dict:
    # Returns финальный пост + метаданные + SEO + статистику токенов
```

**Роли** (`roles.py`): `planner`, `researcher`, `writer`, `editor`, `seo`, `autonomous`.  
Каждая роль — `system_prompt` + `temperature` + `description`.

**Подсчёт токенов** в `pipeline.py` использует `TokenCounter` (не `chars÷4`).

### 5.3 `ArticleFSMAgent` — FSM-агент

Конечный автомат с тремя фазами и 8 шагами:

```
planning                  execution                    validation
─────────────────────     ────────────────────────     ──────────────
choose_topic              write_intro                  check_structure
analyze_topic             write_body                   score_quality
create_outline            write_conclusion             final_edit
```

**Особенности:**
- Состояние персистируется в `TaskStateStorage` после каждого шага
- `Ctrl+C` перехватывается как graceful pause
- Продолжение через `agent.run(task_id="<id>")`

```python
class ArticleFSMAgent:
    def run(
        state: TaskState = None,    # готовый объект
        task_id: str = None,        # загрузить из хранилища
        topic: str = None,          # новая задача
    ) -> TaskState

    def pause(state: TaskState, reason: str = "") -> None
```

**`TaskState`** хранит: `task_id`, `phase`, `current_step`, `article_data`, `topic`, статус.

### 5.4 Управление контекстом

#### `TokenCounter` и `TokenBudget`

```python
class TokenCounter:
    def __init__(model: str)
    # Пытается загрузить tiktoken; при неудаче — fallback chars÷4

    @property
    def is_exact: bool              # True если tiktoken доступен
    @property
    def method_label: str           # "tiktoken" | "~chars÷4"

    def count_text(text: str) -> int
    def count_messages(messages: List[Dict]) -> Dict
    # Returns: total, content_tokens, overhead, per_message, is_exact

    def count_request_breakdown(messages, response_text) -> Dict

# Публичная константа:
MODELS_NO_TEMPERATURE: frozenset   # модели без параметра temperature (o1, o3, gpt-5-nano...)
```

```python
class TokenBudget:
    def __init__(context_limit: int, max_completion: int)
    def check_overflow(prompt_tokens: int) -> Tuple[bool, str]
    def effective_max_completion(prompt_tokens: int) -> int
```

#### `ContextCompressor`

LLM-суммаризация старых сообщений для экономии токенов при длинных диалогах.

```python
class ContextCompressor:
    def __init__(
        model: str,
        summarize_every: int = 10,   # каждые N сообщений
        keep_last_n: int = 6,        # сохранить N последних "живыми"
        summary_max_tokens: int = 600,
    )

    def maybe_compress(
        messages: List[Dict],
        client: OpenAI,
    ) -> Tuple[List[Dict], Optional[CompressionStats]]
    # Возвращает сжатый список и статистику (None если сжатие не применялось)
```

#### Стратегии контекста (`context_strategies.py`)

Три стратегии управления историей диалога:

| Стратегия | Описание | Параметры |
|-----------|----------|-----------|
| `SlidingWindowStrategy` | Скользящее окно по числу токенов | `max_tokens`, `keep_system` |
| `StickyFactsStrategy` | Извлечение ключевых фактов через LLM, инъекция в каждый запрос | `max_facts` |
| `BranchingStrategy` | Checkpoint'ы и ветки диалога | `auto_checkpoint_every` |

```python
# Создание по имени:
strategy = create_strategy("sliding_window", max_tokens=8000)
strategy = create_strategy("sticky_facts", max_facts=15)
strategy = create_strategy("branching", auto_checkpoint_every=5)

# Интерфейс:
class ContextStrategy(ABC):
    def apply(messages, system_prompt, client, model) -> List[Dict]
    def on_user_message(user_content, all_messages, client, model) -> None
    def get_stats() -> Dict
```

### 5.5 Хранилище `PostStorage`

```python
class PostStorage:
    def __init__(base_dir: str = "posts")

    def save(post_data: Dict) -> str          # → post_id (8 символов UUID)
    def load(post_id: str) -> Optional[Dict]
    def list_posts(n: int = 10, post_type: str = None) -> List[Dict]
    def find_by_keywords(topic: str, min_overlap: float = 0.3) -> List[Dict]
    # Keyword-поиск (Jaccard similarity), НЕ семантический

    def search_similar(topic, threshold) -> List[Dict]
    # Устаревший алиас для find_by_keywords()

    def delete(post_id: str) -> bool
    def export_post(post_id: str, fmt: str = "markdown") -> Optional[str]
```

**Структура файлов:**
```
posts/
├── index.json          ← атомарная запись через .tmp + os.replace()
└── <post_id>/
    ├── post.json
    └── post.md
```

### 5.6 `SessionStorage`

```python
class ConversationSession:
    session_id: str
    name: str
    model: str
    strategy: ContextStrategy
    messages: List[Dict]

    def add_user_message(content, client, model) -> None
    def add_assistant_message(content) -> None
    def get_context(system_prompt, client, model) -> List[Dict]
    def compress(client) -> Optional[CompressionStats]

class SessionStorage:
    def save(session: ConversationSession) -> None   # атомарная запись
    def load(session_id: str) -> Optional[ConversationSession]
    def list_sessions(n: int = 20) -> List[Dict]
    def delete(session_id: str) -> bool
```

### 5.7 `OutputFormatter`

```python
class OutputFormatter:
    def convert(
        content: str,
        fmt: str,              # markdown | html | telegram | json | plain
        title: str = None,
        tags: List[str] = None,
        meta: Dict = None,
    ) -> str
```

### 5.8 `UsageTracker`

Логирует каждый запрос к API в JSONL-файл:

```python
class UsageTracker:
    def __init__(log_file: str = "usage.jsonl")
    def record(
        command: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float = 0.0,
        response_time_ms: float = 0.0,
        success: bool = True,
        **extra_fields,
    ) -> None
    def get_summary() -> Dict    # суммарная статистика
```

---

## 6. Модуль `journalist_agent`

### 6.1 Концепция инвариантов

**Инвариант** — жёсткое редакционное ограничение, которое агент не может нарушить ни при каких условиях.

**Двухуровневая защита:**

```
Запрос пользователя
       │
       ▼
┌──────────────────────┐
│ Layer 1: pre_check   │  ← быстрая проверка ключевых слов (без LLM)
│ InvariantStore       │    Нарушение = немедленный отказ
└──────────┬───────────┘
           │ (всё ок)
           ▼
┌──────────────────────┐
│ Layer 2: LLM-вызов   │  ← инварианты инжектированы в system prompt
│ (GLM-4.7)            │    LLM сам проверяет соответствие
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Парсинг ответа       │  ← если начинается с "ОТКАЗ:[INV-XXX]" — фиксируем нарушение
│ REFUSAL_PATTERN      │
└──────────────────────┘
```

### 6.2 `invariants.json`

```json
{
  "invariants": [
    {
      "id": "INV-001",
      "rule": "Запрет освещения художественной гимнастики",
      "description": "Редакция не публикует материалы...",
      "keywords": ["художественная гимнастика", "гимнастика", ...],
      "active": true
    },
    ...
  ]
}
```

### 6.3 `InvariantStore`

```python
class InvariantStore:
    def __init__(path: str = "journalist_agent/invariants.json")

    @property
    def active: List[Invariant]        # только активные инварианты

    def pre_check(text: str) -> ViolationResult
    # Быстрый keyword-скан. Поддерживает stem-matching для падежных форм.

    def build_prompt_section() -> str
    # Форматирует инварианты для инъекции в system prompt

    def summary() -> str
```

### 6.4 `JournalistAgent`

```python
class JournalistAgent:
    def __init__(
        client: OpenAI,
        model: str = "zai-org/GLM-4.7",
        invariant_store: InvariantStore = None,
        verbose: bool = True,
        max_completion_tokens: int = 3000,
        temperature: float = 0.6,
        max_history: int = 20,          # лимит истории диалога (пар)
    )

    def chat(user_input: str) -> AgentResponse
    def reset_dialog() -> None          # сбрасывает историю (инварианты сохраняются)
    def get_invariants_summary() -> str
    def get_total_tokens() -> Dict
```

#### `AgentResponse`

```python
@dataclass
class AgentResponse:
    text: str              # видимый текст (или объяснение отказа)
    allowed: bool          # True если запрос не нарушает инварианты
    violation: ViolationResult
    tokens: Dict[str, int]
```

#### `ViolationResult`

```python
@dataclass
class ViolationResult:
    violated: bool
    invariant: Optional[Invariant]
    layer: str             # "pre_check" | "llm" | "none"
    explanation: str

    @classmethod
    def ok() -> ViolationResult
    @classmethod
    def from_pre_check(inv: Invariant) -> ViolationResult
    @classmethod
    def from_llm(invariant_id, llm_explanation, invariants) -> ViolationResult
```

#### Управление историей

`_history` хранит пары `(user, assistant)`. После каждого хода вызывается `_trim_history()`:

```python
def _trim_history(self) -> None:
    max_msgs = self.max_history * 2      # max_history пар = max_history*2 сообщений
    if len(self._history) > max_msgs:
        self._history = self._history[-max_msgs:]
```

---

## 7. Точка входа `llm_cli.py`

CLI-обёртка над всеми тремя модулями. **104 KB, ~15 команд** — известный God Object.

```bash
python llm_cli.py chat        # диалог с MemoryAgent
python llm_cli.py session     # диалог с SessionManager + стратегией контекста
python llm_cli.py generate    # NewsPipeline (5-шаговый конвейер)
python llm_cli.py agent       # AgentLoop (ReAct)
python llm_cli.py fsm         # ArticleFSMAgent
python llm_cli.py journalist  # JournalistAgent
python llm_cli.py api         # прямой API-вызов
```

---

## 8. Сквозные паттерны

### 8.1 Атомарная запись файлов

Все файловые хранилища используют единый паттерн:

```python
tmp = target_path.with_suffix(".tmp")
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
os.replace(tmp, target_path)
# os.replace() атомарен на POSIX; на Windows — переименование
```

Применяется в: `PostStorage`, `SessionStorage`, `LongTermMemory`.

### 8.2 Retry при API-ошибках

Все агенты используют идентичный паттерн с тремя попытками:

```python
for attempt in range(3):
    try:
        return client.chat.completions.create(**params)
    except Exception as e:
        if attempt < 2:
            time.sleep(2)
        else:
            raise
```

### 8.3 `MODELS_NO_TEMPERATURE`

Единая константа в `token_counter.py`:

```python
MODELS_NO_TEMPERATURE: frozenset = frozenset({
    "gpt-5-nano", "o1", "o1-mini", "o3", "o3-mini", "o4-mini",
})
```

Все агенты импортируют её и используют:
```python
if model in MODELS_NO_TEMPERATURE:
    del params["temperature"]
```

Затронутые файлы: `token_counter.py` (источник), `context_compressor.py`, `context_strategies.py`, `news_agent/agent.py`.

### 8.4 `str.find()` вместо `str.index()`

Все парсеры JSON-ответов LLM используют `str.find()` с явной обработкой `-1`:

```python
start = text.find("```json") + 7
end = text.find("```", start)
if end == -1:
    end = len(text)          # fallback: берём до конца строки
text = text[start:end].strip()
```

Применяется в: `pipeline.py`, `fsm_agent.py`.

### 8.5 Logging вместо silent `pass`

```python
import logging
_log = logging.getLogger(__name__)

try:
    ...
except Exception as exc:
    _log.warning("StickyFacts: fact extraction failed: %s", exc)
```

Применяется в: `context_strategies.py` (StickyFacts).

---

## 9. Конфигурация и зависимости

### 9.1 Переменные окружения (`.env`)

```env
CLOUD_API_KEY=your_api_key_here
CLOUD_BASE_URL=https://api.cloud.ru/v1
```

Загружаются через `python-dotenv`:
```python
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("CLOUD_API_KEY"), base_url=os.getenv("CLOUD_BASE_URL"))
```

### 9.2 `requirements.txt`

```
openai>=1.0.0
python-dotenv>=1.0.0
pytest>=7.0.0
tiktoken>=0.5.0
```

### 9.3 Важные константы

| Константа | Файл | Значение |
|-----------|------|----------|
| `MAX_ITERATIONS` | `news_agent/agent.py` | 12 |
| `FINAL_ANSWER_MARKER` | `news_agent/agent.py` | `"FINAL_POST:"` |
| `DEFAULT_CONTEXT_LIMIT` | `news_agent/agent.py` | 128 000 токенов |
| `MODELS_NO_TEMPERATURE` | `news_agent/token_counter.py` | `frozenset({...})` |
| `DEFAULT_SUMMARIZE_EVERY` | `news_agent/context_compressor.py` | 10 сообщений |
| `DEFAULT_KEEP_LAST_N` | `news_agent/context_compressor.py` | 6 сообщений |

---

## 10. Примеры использования

### 10.1 `MemoryAgent`

```python
from openai import OpenAI
from memory_agent import MemoryAgent

client = OpenAI(api_key="...", base_url="...")
agent = MemoryAgent(client, model="zai-org/GLM-4.7", verbose=True)

response = agent.chat("Меня зовут Кирилл, я Python-разработчик")
print(response)

response = agent.chat("Напомни, как меня зовут?")
print(response)  # Агент ответит, используя LongTermMemory

state = agent.get_memory_state()  # инспекция памяти
```

### 10.2 `NewsPipeline`

```python
from openai import OpenAI
from news_agent import NewsPipeline, PostStorage

client = OpenAI(...)
storage = PostStorage(base_dir="posts")
pipeline = NewsPipeline(client, model="zai-org/GLM-4.7-Flash", verbose=True)

result = pipeline.run(
    topic="ЦБ повысил ставку до 21%",
    post_type="breaking",
)

post_id = storage.save({
    "title": result["edited"]["title"],
    "content": result["edited"]["edited_post"],
    "post_type": "breaking",
    "tags": result["seo"].get("tags", []),
})
print(f"Пост сохранён: {post_id}")
```

### 10.3 `AgentLoop` (ReAct)

```python
from news_agent import AgentLoop, PostStorage

storage = PostStorage()
agent = AgentLoop(
    client,
    model="zai-org/GLM-4.7-Flash",
    storage=storage,
    max_iterations=8,
)

result = agent.run("Напиши breaking-новость о запуске GPT-6")
print(result["final_post"])
print(f"Итераций: {result['iterations']}")
print(f"Токенов: {result['token_usage']['total_tokens']:,}")
```

### 10.4 `ArticleFSMAgent` с паузой/продолжением

```python
from news_agent.fsm_agent import ArticleFSMAgent
from news_agent.fsm_state import TaskStateStorage

storage = TaskStateStorage()
agent = ArticleFSMAgent(client, storage=storage)

# Новая задача
state = agent.run(topic="Будущее квантовых вычислений")
task_id = state.task_id

# Позже — продолжение (например, после Ctrl+C)
state = agent.run(task_id=task_id)
```

### 10.5 `JournalistAgent` с инвариантами

```python
from journalist_agent import JournalistAgent, InvariantStore

store = InvariantStore("journalist_agent/invariants.json")
agent = JournalistAgent(
    client,
    invariant_store=store,
    max_history=10,       # хранить не более 10 пар диалога
)

resp = agent.chat("Напиши статью про балет")
print(resp.text)
print(f"Разрешён: {resp.allowed}")

resp = agent.chat("А теперь про художественную гимнастику")
print(resp.allowed)       # False — pre_check отработал
print(resp.violation.layer)  # "pre_check"
```

### 10.6 Стратегии контекста

```python
from news_agent.session_manager import SessionStorage, ConversationSession
from news_agent.context_strategies import create_strategy

strategy = create_strategy("sticky_facts", max_facts=15)
session = ConversationSession(
    name="tech-digest",
    model="zai-org/GLM-4.7-Flash",
    strategy=strategy,
)

session.add_user_message("Расскажи про квантовые компьютеры", client, model)
session.add_assistant_message("Квантовые компьютеры используют кубиты...")

# Стратегия автоматически управляет контекстом при следующем запросе
context = session.get_context(system_prompt, client, model)
```

---

## 11. Известные ограничения

### 11.1 Технический долг

| Приоритет | Проблема | Файл |
|-----------|----------|------|
| 🔴 Высокий | `llm_cli.py` — God Object (~104 KB, 15+ команд) | `llm_cli.py` |
| 🔴 Высокий | Нет race condition protection при параллельном доступе к `PostStorage` | `storage.py` |
| 🟡 Средний | `WorkingMemory` не персистентна, но описывается как «постоянная» в промпте | `memory.py` |
| 🟡 Средний | `_history` в `JournalistAgent` не интегрирован с `ContextCompressor` | `journalist_agent/agent.py` |
| 🟢 Низкий | `find_by_keywords()` — keyword-match, а не семантический поиск | `storage.py` |
| 🟢 Низкий | `LongTermMemory.search_knowledge()` — подстроковый поиск, не семантический | `memory.py` |

### 11.2 Что хорошо работает

- Атомарные записи во всех файловых хранилищах
- Единая константа `MODELS_NO_TEMPERATURE` — один источник правды
- `str.find()` с fallback везде где парсится LLM-ответ
- Logging вместо silent `pass` в фоновых задачах
- `ShortTermMemory._trim()` — O(1) через slice
- Лимиты на рост `LongTermMemory` (max_decisions, max_knowledge)
- Устранено дублирование `_extract_memory_block` / `_apply_memory_updates`
- `JournalistAgent.max_history` — история больше не растёт бесконечно
- 171 автоматический тест с покрытием ключевых сценариев

### 11.3 Рекомендации для production

1. **Заменить `find_by_keywords()`** на векторный поиск (Chroma, pgvector, FAISS)
2. **Декомпозировать `llm_cli.py`** на отдельные `cli/` субмодули
3. **Добавить файловые блокировки** (`fcntl.flock`) в `PostStorage` и `SessionStorage`
4. **Персистировать `WorkingMemory`** — добавить JSON-сериализацию рядом с `LongTermMemory`
5. **Добавить `ContextCompressor` в `JournalistAgent`** при `max_history > 20`

---

## 12. RAG v2: Rewrite и Reranking

В RAG-модуле добавлен второй этап релевантности после первичного retrieval:

1. `query rewrite` (опционально)
2. первичный retrieval (`top_k_before`)
3. фильтрация по `similarity_threshold`
4. rerank отдельной моделью (опционально)
5. финальный `top_k_after`

### 12.1 Пайплайн

```text
user question
    -> QueryRewriter.rewrite()
    -> search(top_k_before)
    -> similarity filter (threshold)
    -> LLMReranker.rerank()
    -> top_k_after
    -> context injection in RagAgent
```

### 12.2 Новые модули

| Файл | Назначение |
|------|------------|
| `rag/query_rewrite.py` | Переформулировка запроса перед retrieval с fallback на исходный query |
| `rag/reranker.py` | Второй этап ранжирования кандидатов отдельной моделью + fallback |

### 12.3 Расширенные параметры RagAgent

| Параметр | По умолчанию | Назначение |
|----------|---------------|------------|
| `top_k_before` | `10` | Кандидаты до фильтрации/rerank |
| `top_k_after` | `5` | Финальные чанки в контексте |
| `similarity_threshold` | `0.30` | Порог отсечения нерелевантных результатов |
| `enable_query_rewrite` | `True` | Включает rewrite запроса |
| `enable_rerank` | `True` | Включает второй этап reranking |

### 12.4 Режимы сравнения качества

`RagAgent.compare_modes()` и `rag.benchmark --modes` сравнивают 4 режима:

- `baseline` — без rewrite и без rerank
- `rewrite_only` — только rewrite
- `rerank_only` — только filter/rerank
- `combined` — rewrite + filter/rerank

Агрегируемые метрики:

- `avg_keyword_hits`
- `avg_source_precision`
- `avg_tokens`
- `avg_latency_ms`
- `wins_vs_baseline`

### 12.5 Fallback-логика

- если rewrite вернул пустой/ошибочный ответ — используется исходный query
- если reranker недоступен/ошибся — сохраняется порядок по similarity
- пайплайн не падает из-за ошибок второго этапа
6. **Заменить хранилища** на SQLite или PostgreSQL для конкурентного доступа

---

## 13. MCP-оркестрация (Task 17–20)

### 13.1 Компоненты

```
mcp_server.py               — MCP-сервер логов/памяти (9 инструментов)
mcp_client.py               — Базовый MCP-клиент (handshake, call)
mcp_agent.py                — MCPAgent: диалоговый агент + function calling → MCP
mcp_registry.py             — MCPRegistry: реестр серверов, автообнаружение
mcp_router.py               — MCPRouter: маршрутизация tool_calls к серверам
mcp_orchestrator_agent.py   — MCPOrchestratorAgent: мульти-серверный агент
mcp_scheduler_agent.py      — MCPSchedulerAgent: агент + планировщик
mcp_scheduler_server.py     — MCP-сервер планировщика (8 инструментов)
mcp_pipeline.py             — PipelineExecutor: 4-серверный пайплайн
mcp_pipeline_bridge.py      — MCPPipelineBridge: мост к pipeline серверам
mcp_pipeline_server.py      — Монолитный pipeline-сервер (все 4 инструмента)
```

### 13.2 Архитектурная схема MCP

```
┌─────────────────────────────────────────────────────┐
│                MCPOrchestratorAgent                  │
│  (chat → classify → tool_call → route → respond)    │
├─────────────┬───────────────────────────────────────┤
│ MCPRegistry │         MCPRouter                     │
│  (discover) │   (name → server → call)              │
├─────┬───────┴─────────┬──────────────┬──────────────┤
│ mcp_server  │ mcp_scheduler │ pipeline/servers/  │
│ (логи+память) │ (задачи)      │ (4 сервера)        │
└─────────────┴─────────────────┴──────────────────────┘
          JSON-RPC 2.0 по stdin/stdout (stdio)
```

### 13.3 Pipeline-сервера

```
pipeline/servers/
  base.py              — BasePipelineServer (общий протокол)
  search_server.py     — search: поиск по логам
  summarize_server.py  — summarize: суммаризация через LLM
  format_server.py     — format_content: форматирование текста
  store_server.py      — save_to_file, save_to_db: сохранение
```

---

## 14. Планировщик задач (Task 18)

### 14.1 Компоненты

```
scheduler/
  db.py          — SQLite хранилище задач (scheduler_data/scheduler.db)
  scheduler.py   — SchedulerDaemon: фоновый поток, проверка расписания
  tasks.py       — 4 типа задач: report, backup, cleanup, health_check

scheduler_daemon.py  — Точка входа для запуска демона
```

---

## 15. Intent Router (Task 21)

IntentRouter — единый entry-point, классифицирующий запросы пользователя:

- **Keyword matching** (мгновенно, без API)
- **LLM fallback** (если keywords не дали результата)

7 типов намерений: `generate_content`, `schedule_task`, `search_logs`,
`memory`, `journalism`, `pipeline`, `general`.

---

## 16. Пакет core/ (Task 26: рефакторинг)

```
core/
  __init__.py
  config.py       — AppConfig: единый загрузчик конфигурации из .env
  persistence.py  — load_json/save_json: утилиты работы с JSON-файлами
```

### 16.1 Конфигурация

`core.config.get_config()` — синглтон `AppConfig`, загруженный из `.env`.
`core.config.get_llm_client(provider)` — фабрика OpenAI клиентов ("cloud"/"openai").

---

## 17. Пакет mcp_package/ (Task 26: рефакторинг)

Пакет-фасад, предоставляющий единую точку импорта MCP-компонентов:

```python
from mcp_package import MCPAgent, MCPRegistry, MCPRouter
from mcp_package.pipeline import PipelineExecutor, Step
from mcp_package.scheduler import MCPSchedulerAgent
```

Корневые модули (`mcp_agent.py`, `mcp_registry.py` и т.д.) сохранены
для обратной совместимости.

---

## 18. Support Assistant (Task 33)

AI-ассистент поддержки пользователей: RAG по FAQ + MCP-сервер пользователей/тикетов + LLM.

### 18.1 Компоненты

```
support_data/
  faq.md               — FAQ: 7 разделов (авторизация, оплата, API, интеграции...)
  users.json           — Профили пользователей (5 записей)
  tickets.json         — Тикеты поддержки (7 записей)

mcp_support_server.py  — MCP-сервер (5 инструментов: get_user, get_ticket, ...)
support_index.py       — Индексация FAQ → rag_data_support/ (FAISS)
support_assistant.py   — SupportAssistant: RAG + MCP + LLM
mcp_stdio_client.py    — Универсальный MCP stdio клиент (рефакторинг)
```

### 18.2 Архитектурная схема

```
/support --user user_1 Вопрос?
         │
         ▼
┌──────────────────┐
│ SupportAssistant │
├──────┬───────────┤
│ RAG  │    MCP    │
│search│  Support  │
│(FAQ) │  Server   │
└──┬───┴─────┬─────┘
   │         │
   ▼         ▼
┌─────────────────┐
│      LLM        │
│ FAQ + user ctx  │
│ + question      │
└────────┬────────┘
         │
         ▼
  Персонализированный ответ
```

### 18.3 Рефакторинг: MCPStdioClient

Извлечён общий `MCPStdioClient` из внутреннего `_GitMCPClient` в `dev_assistant.py`.
Теперь и DevAssistant, и SupportAssistant используют единый клиент из `mcp_stdio_client.py`.

Подробности: `docs/SUPPORT_ASSISTANT.md`.
