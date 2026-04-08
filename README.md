# TinyAI — AI Курс (17 дней)

Учебный проект: 17 дней от базового промптинга до MCP-интеграции с инструментами для управления логами и памятью.

> **Стек:** Python 3.9+ · Cloud.ru Foundation Models API · `zai-org/GLM-4.7` (reasoning) · OpenAI-compatible SDK · MCP (Model Context Protocol)

---

## Быстрый старт

```bash
# 1. Окружение
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Ключи
cp .env.example .env   # или создай вручную
# Добавь: CLOUD_API_KEY=<ключ от cloud.ru>

# 3. Запуск главного CLI
python3 llm_cli.py

# 4. Запуск FSM-агента журналиста
python3 demos/run_3topics.py --model zai-org/GLM-4.7 --max-tokens 4000

# 5. Запуск MCP-агента (День 17)
python3 demos/demo_mcp_agent.py
```

---

## Модели

| Модель | Провайдер | Особенности |
|---|---|---|
| `zai-org/GLM-4.7` | Cloud.ru | Reasoning (thinking chains), бесплатно |
| `zai-org/GLM-4.7-Flash` | Cloud.ru | Быстрый, бесплатно |

**API endpoint:** `https://foundation-models.api.cloud.ru/v1`

---

## Структура проекта

```
tinyAI/
│
├── llm_cli.py                  # Главный CLI (tasks 2–9): чат, генерация, agent, batch
├── mcp_server.py               # MCP-сервер: 9 инструментов для логов и памяти
├── mcp_client.py               # MCP-клиент: handshake + demo вызовов
├── mcp_agent.py                # MCPAgent: диалоговый агент с function calling → MCP
│
├── news_agent/                 # Пакет: новостной агент (Tasks 3–9, 16)
│   ├── agent.py                #   ReAct AgentLoop с function calling (12 итераций)
│   ├── pipeline.py             #   5-шаговый NewsPipeline
│   ├── roles.py                #   6 ролей + системные промпты
│   ├── tools.py                #   Инструменты: 6 локальных + 9 MCP
│   ├── mcp_bridge.py           #   MCP-мост → mcp_server.py (stdio, JSON-RPC)
│   ├── storage.py              #   PostStorage (JSON на диск)
│   ├── formatter.py            #   OutputFormatter (md/html/telegram/json)
│   ├── session_manager.py      #   ConversationSession + SessionStorage
│   ├── token_counter.py        #   Подсчёт токенов (tiktoken / fallback)
│   ├── usage_tracker.py        #   Трекер API: токены, стоимость, время
│   ├── context_strategies.py   #   Стратегии контекста: sliding, sticky_facts, branching
│   ├── context_compressor.py   #   Автосуммаризация длинных диалогов
│   ├── fsm_agent.py            #   ArticleFSMAgent (planning → done)
│   ├── fsm_state.py            #   FSM: фазы, переходы, сохранение
│   └── strategy_logger.py      #   Логирование стратегий
│
├── journalist_agent/           # Пакет: FSM-агент журналиста (Tasks 13–15)
│   ├── fsm_agent.py            #   JournalistFSMAgent: streaming, retry, guards
│   ├── workflow.py             #   4 типа контента × FSM-переходы
│   ├── step_prompts.py         #   Промпты для каждого шага
│   ├── invariants.py           #   Инварианты и pre-check
│   ├── invariants.json         #   Правила инвариантов (данные)
│   └── agent.py                #   Базовый агент (без FSM)
│
├── memory_agent/               # Пакет: агент с памятью (Tasks 11–12)
│   ├── memory.py               #   Трёхслойная память: short/working/long-term
│   ├── agent.py                #   MemoryAgent с автоизвлечением блоков
│   ├── personalized_agent.py   #   PersonalizedAgent с профилями
│   └── profile.py              #   UserProfile + ProfileManager
│
├── demos/                      # Демо-скрипты и раннеры
│   ├── demo_mcp_agent.py       #   День 17: MCPAgent с function calling
│   ├── demo_article_fsm.py     #   День 13: FSM-агент статей
│   ├── demo_compression.py     #   День 9: компрессия контекста
│   ├── demo_personalization.py #   День 12: два профиля — аналитик vs журналист
│   ├── demo_telegram_article.py#   День 12: статья про Telegram
│   ├── demo_pipeline.py        #   День 3: 5-шаговый pipeline (mock)
│   ├── demo_tokens.py          #   День 8: подсчёт токенов (mock)
│   ├── run_3topics.py          #   День 15: 3 темы × FSM + streaming
│   ├── run_journalist_fsm.py   #   День 14: одна задача с LLM
│   ├── run_journalist_interactive.py  # День 15: интерактивный REPL
│   ├── run_journalist_agent.py #   День 14: базовый агент (без FSM)
│   ├── run_memory_agent.py     #   День 11: тест памяти
│   └── run_strategy_benchmark.py  # День 10: бенчмарк стратегий
│
├── tests/                      # Тесты (pytest)
│   ├── test_journalist_fsm.py  #   FSM-агент: start, advance, pause, resume
│   ├── test_article_fsm.py     #   ARTICLE workflow, переходы
│   ├── test_journalist_agent.py#   Базовый агент и инварианты
│   ├── test_news_agent.py      #   Роли, инструменты, pipeline
│   ├── test_sessions.py        #   Диалоговые сессии
│   ├── test_llm_cli.py         #   CLI-функции
│   ├── test_context_strategies.py  # Стратегии контекста
│   ├── test_context_compressor.py  # Компрессия
│   ├── test_token_counter.py   #   Подсчёт токенов
│   ├── test_usage_tracker.py   #   Трекер использования
│   ├── test_analytics.py       #   Аналитика и метрики
│   ├── test_personalization.py #   Профили и персонализация
│   ├── test_temperature.py     #   Влияние температуры
│   ├── test_gpt54_temperature.py  # GPT-5.4 температура
│   ├── test_model_parameters.py#   Параметры моделей
│   └── test_mcp_server_day17.py#   MCP-сервер: 9 инструментов e2e
│
├── benchmarks/                 # Сравнения моделей
│   ├── compare_models.py       #   3 модели × стиль блога
│   ├── compare_models_chuck.py #   3 модели × новостная заметка
│   └── results/                #   Результаты: JSON, MD, логи
│
├── docs/                       # Документация
│   ├── tasks/                  #   README по каждому заданию (TASK2–TASK17)
│   ├── ARCHITECTURE.md         #   Архитектурная схема
│   ├── MODELS_INFO.md          #   Справочник моделей
│   ├── TEMPERATURE_GUIDE.md    #   Влияние температуры
│   ├── TESTING.md              #   Руководство по тестированию
│   └── ...                     #   Остальная документация
│
├── pytest.ini                  # Конфигурация pytest (pythonpath, testpaths)
├── requirements.txt            # Зависимости
├── .env                        # API-ключи (не в VCS)
└── .gitignore
```

### Runtime-директории (не в VCS, в `.gitignore`)

| Директория | Содержимое |
|---|---|
| `logs/` | Логи разговоров, стратегий, бенчмарков |
| `sessions/` | Диалоговые сессии (JSON) |
| `posts/` | Сгенерированные новостные посты |
| `memory_data/` | Долгосрочная память агента |
| `tasks/` | Задачи memory agent |
| `journalist_tasks_demo/` | JSON-состояния задач FSM + runtime логи |

---

## Task 17 — MCPAgent с function calling

День 17: агент автоматически вызывает MCP-инструменты через function calling для работы с историей и памятью.

```bash
python3 demos/demo_mcp_agent.py
```

### 9 MCP-инструментов

| Инструмент | Описание |
|-----------|----------|
| `list_logs` | Список файлов логов разговоров |
| `read_log` | Содержимое конкретного лога |
| `search_logs` | Поиск текста по всем логам |
| `list_memory` | Список файлов долгосрочной памяти |
| `read_memory` | Содержимое файла памяти |
| `get_usage_stats` | Токены, стоимость, число разговоров |
| `save_memory` | Записать данные в память (День 17) |
| `delete_memory_key` | Удалить ключ из памяти (День 17) |
| `get_conversation_summary` | Сводка разговора (День 17) |

---

## Task 16 — MCP-сервер и клиент

День 16: локальный MCP-сервер управляет логами и памятью, агент подключается к нему как к инструментальному бэкенду.

### Запуск

```bash
# Запустить MCP-клиент — он сам поднимет сервер и выведет список инструментов
python3 mcp_client.py

# Тест MCP-сервера
python3 tests/test_mcp_server_day17.py
```

### Что происходит

```
mcp_client.py
  → запускает mcp_server.py как подпроцесс
  → MCP handshake: initialize → notifications/initialized
  → tools/list → 6 инструментов
  → tools/call (демо: list_logs, list_memory, get_usage_stats, search_logs)
```

### Инструменты MCP-сервера

| Инструмент | Описание |
|-----------|----------|
| `list_logs` | Список файлов логов разговоров |
| `read_log` | Содержимое конкретного лога |
| `search_logs` | Поиск текста по всем логам |
| `list_memory` | Список файлов долгосрочной памяти |
| `read_memory` | Содержимое файла памяти |
| `get_usage_stats` | Токены, стоимость, число разговоров |

### Интеграция в AgentLoop

MCP-инструменты автоматически доступны LLM при вызове `agent` в CLI:

```bash
python3 llm_cli.py
news-agent> agent Сделай обзор предыдущих разговоров на тему ИИ
# Агент сам вызовет search_logs и get_usage_stats через MCP
```

**Архитектура интеграции:**
```
AgentLoop (agent.py)
  └── ToolDispatcher (tools.py)  ← dispatch(tool_name, args)
        ├── локальные инструменты: save_post, analyze_topic, ...
        └── MCP-инструменты: → MCPBridge (mcp_bridge.py)
                                    └── mcp_server.py (подпроцесс, stdio)
```

---

## Task 15 — JournalistFSMAgent

Главная фича проекта: FSM-агент, который проводит журналистский материал через строго контролируемые этапы.

### Типы контента и их шаги

| Тип | Шаги FSM |
|---|---|
| `ARTICLE` | planning → research → drafting → editing → validation → published |
| `NEWS_RESEARCH` | brief → search → analysis → fact_check → published |
| `REVIEW` | criteria → immersion → draft_review → scoring → published |
| `NOTE` | idea_capture → drafting → published |

### Защиты FSM

- **Guards** — нельзя пропустить шаг (planning → drafting запрещён)
- **Пауза** — при ошибке API задача переходит в `paused`, сохраняет состояние
- **Resume** — задача продолжается с того шага, где остановилась
- **Инварианты** — pre-check перед выполнением (тематические ограничения)

### Запуск демо

```bash
# Все три темы последовательно (streaming + автолог)
python3 -u demos/run_3topics.py \
  --model zai-org/GLM-4.7 \
  --max-tokens 4000 \
  --storage-dir journalist_tasks_demo

# Только задача 1 (техника — ARTICLE)
python3 -u demos/run_3topics.py --task 1 --max-tokens 4000

# Задача 2 (финансы — NEWS_RESEARCH)
python3 -u demos/run_3topics.py --task 2 --max-tokens 4000

# Задача 3 (искусство — REVIEW)
python3 -u demos/run_3topics.py --task 3 --max-tokens 4000

# Без записи лога в файл
python3 -u demos/run_3topics.py --task 1 --no-log
```

### Аргументы run_3topics.py

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--model` | `zai-org/GLM-4.7` | LLM-модель |
| `--max-tokens` | `2500` | Макс. токенов на шаг |
| `--task` | все | Запустить только задачу 1/2/3 |
| `--storage-dir` | `journalist_tasks_demo` | Директория хранилища |
| `--no-log` | — | Не писать лог в файл |

### Streaming-вывод в реальном времени

Каждый вызов LLM печатает токены немедленно по мере генерации:

```
▼▼▼▼▼ 📤 ПРОМПТ → LLM  [Планирование] ▼▼▼▼▼
  [SYSTEM] Ты — профессиональный журналист...
  [USER PROMPT]  Составь детальный план статьи...
▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼

▲▲▲▲▲ 📥 ОТВЕТ LLM  [Планирование]  (streaming) ▲▲▲▲▲
  💭 REASONING:                         ← цепочка рассуждений (live)
  ─────────────────────────────────────
  1. Analyze the Request: topic is...   ← токены идут сразу
  2. Drafting sections...
  ─────────────────────────────────────
  📝 CONTENT:                           ← финальный ответ (live)
  {"title": "...", "sections": [...]}
  🔢 Токены: prompt=271  completion=1939  total=2210
▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
```

### Логи

Каждый запуск автоматически создаёт лог-файл:
```
journalist_tasks_demo/run_task1_20260403_210342.log
journalist_tasks_demo/run_all_20260403_215900.log
```

Лог содержит **весь** вывод: промпты, reasoning-цепочки, content, токены.

```bash
# Просмотр последнего лога
ls -t journalist_tasks_demo/*.log | head -1 | xargs cat | less

# Только токены по шагам
ls -t journalist_tasks_demo/*.log | head -1 | xargs grep "Токены:"

# Длины reasoning-цепочек
ls -t journalist_tasks_demo/*.log | head -1 | xargs grep "reasoning:"
```

---

## Tasks 2–9 — News Agent CLI

```bash
python3 llm_cli.py
```

```
news-agent> generate Взлёт SpaceX Starship
news-agent> generate -t analysis Влияние ИИ на рынок труда
news-agent> agent Создай пост о GPT-5
news-agent> batch topics.txt
news-agent> chat                      # новая диалоговая сессия
news-agent> chat list                 # список сессий
news-agent> models                    # доступные модели
```

**Типы постов:** `breaking` · `analysis` · `digest` · `social` · `press`

---

## Тестирование

```bash
# Все тесты
pytest tests/ -v

# Только FSM-агент
pytest tests/test_journalist_fsm.py tests/test_article_fsm.py -v

# Только news agent
pytest tests/test_sessions.py tests/test_llm_cli.py -v

# С выводом print
pytest -v -s tests/test_journalist_fsm.py
```

| Файл | Что проверяет |
|---|---|
| `test_journalist_fsm.py` | JournalistFSMAgent: start, advance, pause, resume, guards |
| `test_article_fsm.py` | ARTICLE workflow, переходы, ошибки |
| `test_journalist_agent.py` | Базовый агент (без FSM) |
| `test_sessions.py` | ConversationSession, SessionStorage |
| `test_llm_cli.py` | CLI-функции, модели |
| `test_analytics.py` | Аналитика и метрики |
| `test_context_strategies.py` | Стратегии управления контекстом |
| `test_token_counter.py` | Подсчёт токенов |

---

## Конфигурация `.env`

```env
CLOUD_API_KEY=ваш-ключ-cloud-ru
CLOUD_BASE_URL=https://foundation-models.api.cloud.ru/v1

# Опционально (для GPT-моделей)
OPENAI_API_KEY=sk-proj-...
```

Получить ключ Cloud.ru: [cloud.ru → Foundation Models](https://cloud.ru/ru/services/foundation-models)

---

## Решение проблем

**`CLOUD_API_KEY not found`** — создайте файл `.env` с ключом

**`AuthenticationError`** — проверьте ключ и баланс

**Пустой `content` при ответе GLM-4.7** — увеличьте `--max-tokens` до 4000+.
GLM-4.7 использует большую часть токенов на reasoning перед генерацией ответа.

**Долгое ожидание (60–90 сек)** — это норма для GLM-4.7 со streaming.
Reasoning-цепочка генерируется первой и печатается токен за токеном.

**`TypeError: proxies`** — `pip install -r requirements.txt --force-reinstall`

---

> ⚠️ **Никогда не коммитьте `.env` с API-ключами в git!**
