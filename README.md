# TinyAI — AI Курс (16 дней)

Учебный проект: 16 дней от базового промптинга до MCP-интеграции с инструментами для управления логами и памятью.

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

# 4. Запуск FSM-агента журналиста (задача 15)
python3 -u run_3topics.py --model zai-org/GLM-4.7 --max-tokens 4000
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
├── journalist_agent/           # Task 15 — FSM-агент журналиста
│   ├── __init__.py
│   ├── fsm_agent.py            # JournalistFSMAgent: streaming, logging, retry
│   ├── workflow.py             # ContentType, FSM-состояния, переходы, guards
│   ├── step_prompts.py         # Промпты для каждого шага каждого типа контента
│   ├── invariants.py           # Инварианты и pre-check перед выполнением
│   ├── invariants.json         # Правила инвариантов (JSON)
│   └── agent.py                # Старый агент (без FSM, совместимость)
│
├── news_agent/                 # Tasks 3–9, 16 — новостной агент
│   ├── pipeline.py             # 5-шаговый NewsPipeline
│   ├── agent.py                # ReAct AgentLoop с function calling
│   ├── session_manager.py      # ConversationSession + SessionStorage
│   ├── storage.py              # PostStorage
│   ├── formatter.py            # OutputFormatter (md/html/telegram/json)
│   ├── roles.py                # Системные промпты
│   ├── tools.py                # Инструменты агента (6 локальных + 6 MCP)
│   └── mcp_bridge.py           # MCP-мост: запускает mcp_server.py как подпроцесс
│
├── memory_agent/               # Task 11 — агент с памятью
│
├── mcp_server.py               # MCP-сервер: инструменты для логов и памяти
├── mcp_client.py               # MCP-клиент: handshake + список инструментов
├── run_3topics.py              # Демо: 3 темы × FSM-агент, streaming, Tee-логи
├── run_journalist_fsm.py       # Демо: одна задача с LLM
├── run_journalist_interactive.py  # Интерактивный REPL для FSM
├── run_journalist_agent.py     # Демо без FSM
├── run_memory_agent.py         # Демо memory agent
├── llm_cli.py                  # Главный CLI (tasks 2–9)
│
├── test_journalist_fsm.py      # Тесты FSM-агента
├── test_journalist_agent.py    # Тесты базового агента
├── test_article_fsm.py         # Тесты workflow ARTICLE
├── test_sessions.py
├── test_llm_cli.py
├── test_*.py                   # Остальные тесты
│
├── requirements.txt
├── pytest.ini
├── .env                        # API-ключи (не в VCS)
├── .gitignore
│
├── README.md                   # Этот файл
├── ARCHITECTURE.md             # Архитектурная схема v7.0
├── TASK16_README.md            # День 16: MCP-клиент и интеграция
├── TASK15_README.md            # День 15: архитектурный анализ
├── TASK14_README.md            # День 14: FSM-агент журналиста
├── TASK13_README.md            # День 13: инварианты и guards
├── TASK12_README.md            # День 12: memory agent
├── TASK10_PLAN.md              # День 10: план разработки
├── TASK9_README.md             # День 9: персонализация
├── TASK8_README.md             # День 8: аналитика и метрики
├── TASK7_README.md             # День 7: диалоговые сессии
├── TASK6_README.md             # День 6: ReAct-агент
├── TASK3_README.md             # День 3: news pipeline
├── TASK2_README.md             # День 2: режимы форматирования
├── MODELS_INFO.md              # Справочник моделей
├── TEMPERATURE_GUIDE.md        # Влияние температуры
└── TESTING.md                  # Руководство по тестированию
```

**Runtime-директории (не в VCS, в `.gitignore`):**
```
journalist_tasks_demo/   # JSON-состояния задач + run_*.log
sessions/                # диалоговые сессии
posts/                   # сгенерированные посты
memory_data/             # долгосрочная память агента
tasks/                   # задачи memory agent
logs/                    # общие логи
```

---

## Task 16 — MCP-интеграция

День 16: локальный MCP-сервер управляет логами и памятью, агент подключается к нему как к инструментальному бэкенду.

### Запуск

```bash
# Запустить MCP-клиент — он сам поднимет сервер и выведет список инструментов
python3 mcp_client.py
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
python3 -u run_3topics.py \
  --model zai-org/GLM-4.7 \
  --max-tokens 4000 \
  --storage-dir journalist_tasks_demo

# Только задача 1 (техника — ARTICLE)
python3 -u run_3topics.py --task 1 --max-tokens 4000

# Задача 2 (финансы — NEWS_RESEARCH)
python3 -u run_3topics.py --task 2 --max-tokens 4000

# Задача 3 (искусство — REVIEW)
python3 -u run_3topics.py --task 3 --max-tokens 4000

# Без записи лога в файл
python3 -u run_3topics.py --task 1 --no-log
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
pytest -v

# Только FSM-агент
pytest test_journalist_fsm.py test_article_fsm.py -v

# Только news agent
pytest test_sessions.py test_llm_cli.py -v

# С выводом print
pytest -v -s test_journalist_fsm.py
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
