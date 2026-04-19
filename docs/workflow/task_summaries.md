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

## Ключевые файлы
| Файл | Роль |
|------|------|
| `news_agent/token_counter.py` | TokenCounter + overflow detection |

## Тесты
- `tests/test_token_counter.py` — подсчёт, лимиты, overflow

---

# День 9 — Компрессия контекста

## Требования
Сжатие диалогового контекста при приближении к лимиту окна.

## Что реализовано
- `ContextCompressor` — LLM-based summarization старых сообщений
- Progressive compression: compress older messages first
- Порог срабатывания: 80% от context window

## Ключевые файлы
| Файл | Роль |
|------|------|
| `news_agent/context_compressor.py` | Компрессия контекста |

## Тесты
- `tests/test_context_compressor.py` — сжатие, пороги

---

# День 10 — Стратегии управления контекстом

## Требования
Несколько стратегий управления контекстным окном.

## Что реализовано
- Стратегии: `truncate`, `summarize`, `sliding_window`
- Конфигурируемые пороги и размер окна
- Автоматический выбор стратегии по размеру контекста

## Ключевые файлы
| Файл | Роль |
|------|------|
| `news_agent/context_strategies.py` | Стратегии контекста |

## Тесты
- `tests/test_context_strategies.py` — все стратегии

---

# День 11 — Memory Agent

## Требования
Агент с 3-слойной памятью: short-term, working, long-term.

## Что реализовано
- `MemoryAgent` — трёхуровневая память
- short-term: текущий диалог
- working: переменные сессии
- long-term: персистентное хранилище (JSON)
- Автоматическое перемещение между слоями

## Ключевые файлы
| Файл | Роль |
|------|------|
| `memory_agent/agent.py` | MemoryAgent с 3-layer memory |

## Тесты
- `tests/test_memory_agent.py` — все слои памяти

---

# День 12 — Персонализация

## Требования
Персонализированные ответы на основе профиля пользователя.

## Что реализовано
- `PersonalizedAgent` — расширение MemoryAgent с user profile
- Стиль ответов адаптируется к предпочтениям
- Профиль обновляется автоматически из контекста

## Ключевые файлы
| Файл | Роль |
|------|------|
| `memory_agent/personalized_agent.py` | Персонализация |

## Тесты
- `tests/test_personalization.py` — профиль, стиль

---

# День 13 — Journalist FSM Agent

## Требования
Агент-журналист на конечном автомате (FSM) с 4 типами статей.

## Что реализовано
- `JournalistFSMAgent` — FSM-based workflow
- 4 типа: ARTICLE, NEWS_RESEARCH, REVIEW, INTERVIEW
- Состояния: PLANNING → RESEARCH → WRITING → EDITING → DONE
- Guards: проверки перед переходом, pause/resume

## Ключевые файлы
| Файл | Роль |
|------|------|
| `journalist_agent/fsm_agent.py` | FSM Agent |
| `journalist_agent/workflow.py` | Workflow definitions |
| `journalist_agent/step_prompts.py` | Промпты по шагам |

## Тесты
- `tests/test_journalist_fsm.py` — FSM, переходы, guards
- `tests/test_article_fsm.py` — ARTICLE workflow

---

# День 14 — Инварианты и streaming

## Требования
Инварианты FSM + streaming-вывод для всех LLM-вызовов.

## Что реализовано
- `invariants.json` — конфигурация инвариантов FSM
- Streaming: `<thinking>` tag handling, real-time output
- Проверка инвариантов на каждом переходе

## Ключевые файлы
| Файл | Роль |
|------|------|
| `journalist_agent/invariants.py` | Проверка инвариантов |
| `journalist_agent/invariants.json` | Конфигурация |

## Тесты
- `tests/test_journalist_agent.py` — инварианты, streaming

---

# День 15 — 3 темы × FSM

## Требования
Демонстрация FSM-агента на трёх разных темах с автоматическим логированием.

## Что реализовано
- `run_3topics.py` — запуск 3 статей последовательно
- Автологирование в `journalist_tasks_demo/`
- CLI-аргументы: `--task`, `--model`, `--max-tokens`, `--no-log`

## Ключевые файлы
| Файл | Роль |
|------|------|
| `demos/run_3topics.py` | 3 темы × FSM demo |

---

# День 16 — MCP-сервер и клиент

## Требования
MCP (Model Context Protocol) сервер и клиент для управления логами и памятью.

## Что реализовано
- `mcp_server.py` — 9 инструментов (list_logs, read_log, search_logs, ...)
- `mcp_client.py` — подключение к серверу, вызов инструментов
- JSON-RPC совместимый протокол

## Ключевые файлы
| Файл | Роль |
|------|------|
| `mcp_server.py` | MCP Server (9 tools) |
| `mcp_client.py` | MCP Client |

---

# День 17 — MCP Agent (function calling)

## Требования
Агент с function calling через MCP.

## Что реализовано
- `MCPAgent` — function calling loop с MCP Bridge
- Tool definitions → LLM выбирает инструмент → выполнение → результат
- Инструменты: save_memory, delete_memory_key, get_conversation_summary

## Ключевые файлы
| Файл | Роль |
|------|------|
| `mcp_agent.py` | MCPAgent |

---

# День 18 — Планировщик задач

## Требования
Планировщик периодических задач через MCP.

## Что реализовано
- `MCPSchedulerAgent` — агент с 7+ tool definitions
- Задачи: chat_collector, chat_backup, summary_generator, reminder
- `MCPSchedulerBridge` — мост к scheduler server
- Tool routing: scheduler tools → sched_bridge, logs → logs_bridge

## Ключевые файлы
| Файл | Роль |
|------|------|
| `mcp_scheduler_agent.py` | MCPSchedulerAgent |
| `mcp_scheduler_server.py` | Scheduler MCP Server |
| `scheduler/` | Scheduler core |

---

# День 19 — Расписание + daemon

## Требования
Фоновый daemon для автоматического выполнения задач.

## Что реализовано
- `scheduler_daemon.py` — background daemon
- Периодическое выполнение: cron-like intervals
- Auto-restart, logging

## Ключевые файлы
| Файл | Роль |
|------|------|
| `scheduler_daemon.py` | Scheduler Daemon |

---

# День 20 — MCP-оркестратор

## Требования
Координация нескольких MCP-серверов.

## Что реализовано
- `MCPOrchestratorAgent` — мульти-сервер оркестрация
- `MCPRegistry` — регистрация серверов
- `MCPRouter` — маршрутизация вызовов к нужному серверу

## Ключевые файлы
| Файл | Роль |
|------|------|
| `mcp_orchestrator_agent.py` | Orchestrator Agent |
| `mcp_registry.py` | MCPRegistry |
| `mcp_router.py` | MCPRouter |

## Тесты
- `tests/test_mcp_orchestrator.py` — оркестрация, маршрутизация

---

# День 21 — IntentRouter

## Требования
Маршрутизация запросов к нужному агенту.

## Что реализовано
- `IntentRouter` — keyword matching + LLM fallback
- Категории: news, journalist, memory, scheduler, rag, general

## Ключевые файлы
| Файл | Роль |
|------|------|
| `intent_router.py` | IntentRouter |

---

# День 22 — RAG Agent (dual-mode)

## Требования
RAG-агент с двумя режимами и бенчмарком.

## Что реализовано
- `RagAgent` — dual-mode: ask_without_rag / ask_with_rag
- Streaming вывод с `<thinking>` tag handling
- `compare()` — параллельный запуск обоих режимов
- `benchmark.py` — 10 контрольных вопросов, keyword/source metrics
- `answer_comparison.py` — build_comparison, compare_all, modes comparison

## Ключевые файлы
| Файл | Роль |
|------|------|
| `rag/rag_agent.py` | RagAgent |
| `rag/benchmark.py` | 10 контрольных вопросов + evaluate |
| `rag/answer_comparison.py` | Сравнение ответов |
| `demos/demo_rag_agent.py` | Streaming demo |

## Тесты
- `tests/test_rag_agent.py` — dual-mode, compare, answer comparison, benchmark

---

# День 23 — Query Rewrite + Reranking

## Требования
Улучшение retrieval через query rewrite и reranking.

## Что реализовано
- `QueryRewriter` — LLM-based перефразирование запроса
- `LLMReranker` — повторное ранжирование результатов через LLM
- `compare_modes()` — 4 конфигурации: baseline / rewrite / rerank / combined
- Similarity threshold filter

## Ключевые файлы
| Файл | Роль |
|------|------|
| `rag/query_rewrite.py` | QueryRewriter |
| `rag/reranker.py` | LLMReranker |

---

# День 24 — RAG с цитатами

## Требования
Inline-цитаты с верификацией и anti-hallucination guard.

## Что реализовано
- `ask_with_citations()` — RAG + цитаты + SourceRef
- `citation_parser.py` — парсинг и нормализация цитат
- Auto-извлечение цитат из top-чанков (fallback)
- Confidence threshold — отклоняет нерелевантные ответы
- Quote verification через normalized text matching

## Ключевые файлы
| Файл | Роль |
|------|------|
| `rag/citation_parser.py` | Парсер цитат |
| `rag/rag_agent.py` | ask_with_citations |

## Тесты
- `tests/test_rag_citations.py` — парсинг, верификация, fallback

---

# День 25 — Fast Mode RAG + MathReranker

## Требования
Ускоренный RAG без LLM-reranker.

## Что реализовано
- `rag_chat_agent.py` — интерактивный RAG-чат
- MathReranker — TF-IDF + BM25 + Score Fusion (без LLM)
- Значительно быстрее LLM-reranker при сопоставимом качестве

## Ключевые файлы
| Файл | Роль |
|------|------|
| `rag/rag_chat_agent.py` | RAG Chat Agent |
| `rag/math_reranker.py` | MathReranker |
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
