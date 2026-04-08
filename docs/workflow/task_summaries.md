# День 2 — Сравнение форматов ответов и логирование

## Требования
Реализовать 6 режимов форматирования ответов LLM, переключение моделей, сравнение выводов и логирование.

## Что реализовано
- 6 режимов ответа: без ограничений, структурный, лимит длины, маркеры, JSON, мета-промптинг
- Переключение между 7 моделями OpenAI с автоподсчётом стоимости
- Команда `compare` — все режимы на одном запросе
- Команда `temperature` — сравнение при температурах 0, 0.7, 1.2
- Команда `reasoning` — 4 подхода к решению задач
- Автологирование: количество слов, токены, символы, стоимость
- Результаты → `logs/conversation_YYYYMMDD_HHMMSS.json`

## Ключевые файлы
| Файл | Роль |
|------|------|
| `llm_cli.py` | Главный CLI со всеми 6 режимами |

## Тесты
- `tests/test_llm_cli.py` — CLI-функции, модели, режимы
- `tests/test_model_parameters.py` — параметры моделей (integration)

---

# День 3 — Миграция на Cloud.ru

## Требования
Перенести проект с OpenAI API на Cloud.ru Foundation Models API.

## Что реализовано
- API endpoint: `https://foundation-models.api.cloud.ru/v1`
- Модель: `zai-org/GLM-4.7-Flash` (бесплатная)
- Все 6 режимов и reasoning работают идентично
- Ключ: `CLOUD_API_KEY` вместо `OPENAI_API_KEY`
- 23 теста проходят

## Ключевое изменение
OpenAI-совместимый SDK → та же кодовая база, другой endpoint и модель.

---

# День 6 — News Agent CLI

## Требования
AI-агент для автоматической генерации новостных постов: pipeline и autonomous ReAct agent.

## Что реализовано
**Pipeline (5 шагов):** planner → researcher → writer → editor → seo

**Post types:** breaking, analysis, digest, social, press

**6 инструментов ReAct:** save_post, load_post_history, analyze_topic, format_post, get_post_by_id, check_duplicate

**Команды:** generate, agent, batch, history, export, template, models

## Ключевые файлы
| Файл | Роль |
|------|------|
| `news_agent/pipeline.py` | 5-шаговый NewsPipeline |
| `news_agent/agent.py` | ReAct AgentLoop с function calling |
| `news_agent/roles.py` | 6 ролей + системные промпты |
| `news_agent/tools.py` | Инструменты агента |
| `news_agent/storage.py` | PostStorage (JSON) |
| `news_agent/formatter.py` | Форматтер (md/html/telegram/json) |

## Тесты
- `tests/test_news_agent.py` — роли, инструменты, pipeline, агент

---

# День 7 — Диалоговые сессии

## Требования
Управление диалоговыми сессиями с персистентным хранилищем и отслеживанием контекстного окна.

## Что реализовано
- `ConversationSession` — единичный диалог с историей
- `SessionStorage` — файловое хранилище в `sessions/`
- `DialogTokenTracker` — отслеживание роста токенов по ходам
- Индикаторы контекста: 🟢 < 50%, 🟡 < 80%, 🔴 ≥ 80%
- Команды: `chat`, `chat new`, `chat list`, `chat load`, `chat delete`, `info`

## Ключевые файлы
| Файл | Роль |
|------|------|
| `news_agent/session_manager.py` | ConversationSession + SessionStorage |

## Тесты
- `tests/test_sessions.py` — сессии, хранилище, токены, контекст

---

# День 8 — Работа с токенами

## Требования
Точный подсчёт и управление токенами: размер запроса, рост стоимости, overflow detection.

## Что реализовано
- `TokenCounter` — точный подсчёт через tiktoken, fallback chars÷4
- `DialogTokenTracker` — отслеживание на каждом ходе, таблица роста
- `TokenBudget` — детекция overflow до вызова API
- `count_request_breakdown()` — детальная разбивка токенов
- Интеграция в AgentLoop с max_completion_tokens

## Ключевые файлы
| Файл | Роль |
|------|------|
| `news_agent/token_counter.py` | TokenCounter, DialogTokenTracker, TokenBudget |

## Тесты
- `tests/test_token_counter.py` — подсчёт, трекер, бюджет
- `demos/demo_tokens.py` — 5 сценариев (mock, без API)

---

# День 9 — Сжатие контекста

## Требования
Sliding window compression: суммаризация старых сообщений через LLM.

## Что реализовано
- `ContextCompressor` — сжатие при превышении порога сообщений
- Параметры: `summarize_every=10`, `keep_last_n=6`
- Интеграция в `ConversationSession`: `enable_compression()`, `maybe_compress()`
- Команды чата: `compress`, `compress on/off`, `compress compare`
- Экономия токенов: 80-90% для длинных диалогов

## Ключевые файлы
| Файл | Роль |
|------|------|
| `news_agent/context_compressor.py` | ContextCompressor |

## Тесты
- `tests/test_context_compressor.py` — компрессия, статистика

---

# День 10 — Три стратегии управления контекстом

## Требования
Три стратегии управления контекстом (не компрессия): sliding window, sticky facts, branching.

## Что реализовано
1. **SlidingWindowStrategy** — хранить последние N сообщений
2. **StickyFactsStrategy** — извлечение ключ-значение фактов через LLM
3. **BranchingStrategy** — ветвление диалогов с чекпоинтами

Бенчмарк: 20 промптов (сбор ТЗ на фитнес-приложение) × 3 стратегии.

## Ключевые файлы
| Файл | Роль |
|------|------|
| `news_agent/context_strategies.py` | 3 стратегии с единым интерфейсом |
| `news_agent/strategy_logger.py` | Логирование и сравнение стратегий |

## Тесты
- `tests/test_context_strategies.py` — все стратегии, фабрика, сериализация
- `demos/run_strategy_benchmark.py` — автоматический прогон

---

# День 12 — Персонализация ассистента

## Требования
Профильная система персонализации поверх трёхслойной памяти.

## Что реализовано
- `UserProfile` — профиль с именем, ролью, стилем, форматом
- `ProfileManager` — управление профилями, сериализация
- `PersonalizedAgent` — агент с профилем, адаптивные ответы
- 2 встроенных профиля: `system_analyst` (формальный, UML) и `journalist` (живой, нарративный)

## Ключевые файлы
| Файл | Роль |
|------|------|
| `memory_agent/profile.py` | UserProfile, ProfileManager |
| `memory_agent/personalized_agent.py` | PersonalizedAgent |

## Тесты
- `tests/test_personalization.py` — профили, менеджер, агент (39 тестов)
- `demos/demo_personalization.py`, `demos/demo_telegram_article.py`

---

# День 13 — FSM-агент написания статей

## Требования
Конечный автомат (FSM) для написания статей с поддержкой паузы/возобновления.

## Что реализовано
- 4 фазы: planning → execution → validation → done
- 9 шагов: choose_topic → analyze → outline → intro → body → conclusion → check → score → edit
- `TaskState` — состояние с паузой, `TaskStateStorage` — JSON-хранилище
- `ArticleFSMAgent` — оркестратор
- Pause: Ctrl+C или `--pause-after`, Resume: `--resume <task_id>`

## Ключевые файлы
| Файл | Роль |
|------|------|
| `news_agent/fsm_state.py` | TaskPhase, TaskState, TaskStateStorage |
| `news_agent/fsm_agent.py` | ArticleFSMAgent |

## Тесты
- `tests/test_article_fsm.py` — переходы, сериализация, полный прогон (45 тестов)
- `demos/demo_article_fsm.py` — CLI с паузой/возобновлением

---

# День 14 — Инварианты и ограничения

## Требования
Жёсткие ограничения для агента журналиста, которые нельзя нарушить.

## Что реализовано
- **Слой 1 (pre-check):** поиск по ключевым словам без вызова LLM
- **Слой 2 (LLM reasoning):** инварианты в системном промпте, формат отказа `ОТКАЗ:[INV-XXX]`
- `Invariant` — определение ограничения (id, правило, ключевые слова)
- `InvariantStore` — загрузка из `invariants.json`

## Ключевые файлы
| Файл | Роль |
|------|------|
| `journalist_agent/invariants.py` | Invariant, InvariantStore |
| `journalist_agent/invariants.json` | Правила инвариантов |
| `journalist_agent/agent.py` | JournalistAgent с двухслойной защитой |

## Тесты
- `tests/test_journalist_agent.py` — хранилище, детекция нарушений (24 теста)
- `demos/run_journalist_agent.py` — 6 сценариев

---

# День 15 — JournalistFSMAgent + архитектурный анализ

## Требования
FSM-агент журналиста с 4 типами контента и полный архитектурный ревью.

## Что реализовано
**4 типа контента × FSM:**
| Тип | Шаги |
|---|---|
| ARTICLE | planning → research → drafting → editing → validation → published |
| NEWS_RESEARCH | brief → search → analysis → fact_check → published |
| REVIEW | criteria → immersion → draft_review → scoring → published |
| NOTE | idea_capture → drafting → published |

**Защиты:** guards (нельзя пропустить шаг), пауза/resume, инварианты

**Архитектурная оценка:** 7/10 — хорошая учебная база, нужен hardening для прода

## Ключевые файлы
| Файл | Роль |
|------|------|
| `journalist_agent/fsm_agent.py` | JournalistFSMAgent |
| `journalist_agent/workflow.py` | Типы контента, FSM-переходы, guards |
| `journalist_agent/step_prompts.py` | Промпты для каждого шага |

## Тесты
- `tests/test_journalist_fsm.py` — start, advance, pause, resume
- `demos/run_3topics.py` — 3 темы × streaming + автолог

---

# День 16 — MCP-сервер и клиент

## Требования
Локальный MCP-сервер для управления логами и памятью по протоколу JSON-RPC 2.0 (stdio).

## Что реализовано
- MCP-сервер с 6 инструментами: list_logs, read_log, search_logs, list_memory, read_memory, get_usage_stats
- MCP-клиент: handshake (initialize → initialized → tools/list → tools/call)
- Без внешних SDK — чистый JSON-RPC 2.0 через stdin/stdout

## Ключевые файлы
| Файл | Роль |
|------|------|
| `mcp_server.py` | MCP-сервер (6 инструментов) |
| `mcp_client.py` | MCP-клиент (handshake + demo) |
| `news_agent/mcp_bridge.py` | Мост MCP для AgentLoop |

---

# День 17 — Первый инструмент MCP

## Требования
Расширить MCP-сервер записью в память, подключить к агенту через function calling.

## Что реализовано
- 3 новых инструмента: save_memory, delete_memory_key, get_conversation_summary
- `MCPAgent` — агент с function calling → MCP-инструменты
- LLM сам решает какой инструмент вызвать и с какими аргументами

## Ключевые файлы
| Файл | Роль |
|------|------|
| `mcp_agent.py` | MCPAgent с function calling |
| `demos/demo_mcp_agent.py` | 5 сценариев |

## Тесты
- `tests/test_mcp_server_day17.py` — 9 инструментов end-to-end
