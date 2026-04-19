# TinyAI — Краткая сводка проекта

> Версия документа: 19 апреля 2026 · Ветка: 24task

---

## 1. Что такое TinyAI

Учебный проект из 25 задач: от базового промптинга до полноценной RAG-системы с MCP-оркестрацией, планировщиком и бенчмарками. Язык — Python 3.9+, LLM — Cloud.ru `zai-org/GLM-4.7` (reasoning), OpenAI GPT-5.x для сравнений.

**Стек:** OpenAI SDK 1.12+ · FAISS · tiktoken · Cloud.ru Foundation Models API · MCP (Model Context Protocol)

---

## 2. Архитектура

Система состоит из **6 агентов**, связанных общим LLM-слоем и MCP-инфраструктурой:

```
                        ┌─────────────────────────────┐
                        │          LLM API             │
                        │  Cloud.ru / OpenAI           │
                        └──────┬──────────┬────────────┘
                               │          │
            ┌──────────────────┼──────────┼──────────────────┐
            │                  │          │                  │
    ┌───────▼──────┐  ┌───────▼───┐  ┌───▼────────┐  ┌─────▼──────┐
    │  NewsAgent   │  │ Journalist │  │ MemoryAgent│  │  RagAgent  │
    │  (pipeline,  │  │ FSMAgent   │  │ (3-слоя    │  │ (dual-mode │
    │   ReAct)     │  │ (4 типа)   │  │  памяти)   │  │  + цитаты) │
    └──────────────┘  └────────────┘  └────────────┘  └────────────┘
            │                  │
    ┌───────▼──────────────────▼───────┐
    │        MCP Orchestrator          │
    │  MCPRouter → MCPRegistry         │
    │  MCPSchedulerAgent (daemon)      │
    └──────────────────────────────────┘
```

### Ключевые паттерны

| Паттерн | Где используется | Описание |
|---|---|---|
| **Pipeline** | NewsAgent | 5 ролей: planner → researcher → writer → editor → seo |
| **ReAct** | NewsAgent, MCPAgent | Цикл: think → act (tool_call) → observe → ... |
| **FSM** | JournalistFSMAgent | Конечный автомат: guards, pause/resume, инварианты |
| **RAG** | RagAgent | Retrieval → Rewrite → Rerank → Generate |
| **MCP** | MCPAgent, Orchestrator | Стандартный протокол вызова инструментов |
| **3-Layer Memory** | MemoryAgent | short-term → working → long-term |

---

## 3. Все 25 задач

| # | Тема | Ключевые файлы | Статус |
|---|---|---|---|
| 2 | Форматы ответов, 6 режимов | `llm_cli.py` | ✅ |
| 3 | Миграция на Cloud.ru | `llm_cli.py` | ✅ |
| 6 | News Agent CLI + pipeline | `news_agent/` | ✅ |
| 7 | Диалоговые сессии | `news_agent/session_manager.py` | ✅ |
| 8 | Подсчёт токенов | `news_agent/token_counter.py` | ✅ |
| 9 | Компрессия контекста | `news_agent/context_compressor.py` | ✅ |
| 10 | Стратегии контекста | `news_agent/context_strategies.py` | ✅ |
| 11 | Memory Agent | `memory_agent/agent.py` | ✅ |
| 12 | Персонализация | `memory_agent/personalized_agent.py` | ✅ |
| 13 | Journalist FSM | `journalist_agent/fsm_agent.py` | ✅ |
| 14 | Инварианты + streaming | `journalist_agent/invariants.py` | ✅ |
| 15 | 3 темы × FSM, run_3topics | `demos/run_3topics.py` | ✅ |
| 16 | MCP-сервер + клиент | `mcp_server.py`, `mcp_client.py` | ✅ |
| 17 | MCPAgent (function calling) | `mcp_agent.py` | ✅ |
| 18 | Планировщик задач | `mcp_scheduler_agent.py`, `scheduler/` | ✅ |
| 19 | Расписание + daemon | `scheduler_daemon.py` | ✅ |
| 20 | MCP-оркестратор | `mcp_orchestrator_agent.py` | ✅ |
| 21 | IntentRouter | `intent_router.py` | ✅ |
| 22 | RAG Agent (dual-mode) | `rag/rag_agent.py`, `rag/benchmark.py` | ✅ |
| 23 | Query Rewrite + Reranking | `rag/query_rewrite.py`, `rag/reranker.py` | ✅ |
| 24 | RAG с цитатами | `rag/citation_parser.py` | ✅ |
| 25 | Fast Mode RAG | `rag/rag_chat_agent.py`, MathReranker | ✅ |

---

## 4. RAG-система

### Пайплайн

```
Вопрос → Query Rewrite (опц.) → FAISS поиск → Similarity Filter
       → LLM Reranking (опц.) → Context Injection → LLM Generate → Ответ
```

### Стратегии чанкинга

- **fixed_size** — фиксированные чанки по 512 токенов с overlap 64
- **structure** — разбиение по Markdown-заголовкам (семантические границы)

### Ключевые метрики (бенчмарк 10 вопросов)

| Метрика | Без RAG | С RAG |
|---|---|---|
| Keyword precision | ~40% | ~75% |
| Source precision | — | ~80% |
| Avg latency | 30–90 сек | 30–60 сек |
| Token overhead | baseline | ×1.3–1.8 |

### Режимы RAG

1. `ask_without_rag` — только LLM
2. `ask_with_rag` — retrieval + LLM
3. `ask_with_citations` — RAG + inline-цитаты + верификация
4. `compare` — оба режима одновременно
5. `compare_modes` — 4 конфигурации: baseline / rewrite / rerank / combined

---

## 5. MCP-инфраструктура

### Инструменты

**MCP-сервер** (9 инструментов): `list_logs`, `read_log`, `search_logs`, `list_memory`, `read_memory`, `get_usage_stats`, `save_memory`, `delete_memory_key`, `get_conversation_summary`

**Планировщик** (7 инструментов): `scheduler_add_task`, `scheduler_list_tasks`, `scheduler_remove_task`, `scheduler_trigger_task`, `scheduler_get_task_runs`, `scheduler_get_summary`, `scheduler_list_snapshots`, `scheduler_list_backups`

### Оркестрация

`MCPOrchestratorAgent` координирует несколько MCP-серверов через `MCPRegistry` (регистрация серверов) и `MCPRouter` (маршрутизация вызовов). `IntentRouter` определяет тип запроса (keyword matching + LLM fallback) и направляет к нужному агенту.

---

## 6. Тестирование

### Статистика

- **22 тестовых файла**, ~800+ тестов
- **4 smoke-теста** (ручная интеграция с API)
- **Маркеры pytest:** `unit`, `integration`, `api`, `e2e`, `slow`
- **Время прохождения unit:** <1 секунда (mocked)

### Покрытие по модулям

| Область | Тестовые файлы | Тестов |
|---|---|---|
| Sessions/Context | test_sessions, test_context_strategies, test_context_compressor, test_token_counter | ~200 |
| Journalist/FSM | test_article_fsm, test_journalist_fsm, test_journalist_agent | ~190 |
| RAG | test_rag_agent, test_rag_citations, test_rag_chunker, test_rag | ~140 |
| News/MCP | test_news_agent, test_mcp_orchestrator, test_pipeline_executor | ~120 |
| CLI/Config | test_llm_cli, test_core, test_model_parameters | ~80 |
| Analytics | test_usage_tracker, test_analytics, test_new_features | ~120 |

### Запуск

```bash
# Все offline-тесты
pytest tests/ -v -m "not api"

# Только RAG
pytest tests/test_rag_agent.py -v

# С реальным API
pytest tests/ -v -m api
```

---

## 7. Модели и тарифы

### Cloud.ru (бесплатно)

| Модель | Особенности |
|---|---|
| `zai-org/GLM-4.7` | Reasoning (thinking chains), основная |
| `zai-org/GLM-4.7-Flash` | Быстрая, для rewrite/rerank |

### OpenAI (платно)

| Модель | Input $/1M | Output $/1M |
|---|---|---|
| GPT-5 Nano | $0.20 | $1.25 |
| GPT-5.4 Mini | $0.75 | $4.50 |
| GPT-5.4 | $2.50 | $15.00 |

*Цены актуальны на 21 марта 2026.*

---

## 8. Быстрый старт

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # добавить CLOUD_API_KEY

# CLI
python3 llm_cli.py

# RAG demo (streaming, 3 вопроса)
python3 demos/demo_rag_agent.py

# Полный бенчмарк (10 вопросов)
python3 demos/demo_rag_agent.py --full

# Journalist FSM
python3 demos/run_3topics.py --model zai-org/GLM-4.7

# MCP Agent
python3 demos/demo_mcp_agent.py

# Тесты
pytest tests/ -v
```

---

## 9. Зависимости

```
openai>=2.29.0          # LLM клиент
python-dotenv==1.0.1    # .env переменные
pytest==8.0.0           # Тестирование
tiktoken>=0.7.0         # Подсчёт токенов
faiss-cpu>=1.7.4        # Векторный поиск
numpy>=1.24.0           # Математика для embeddings
```

---

## 10. Известные ограничения и направления развития

### Ограничения

- **Синхронный код** — все API-вызовы блокирующие; нет asyncio
- **God-object `llm_cli.py`** — 1000+ строк, требует декомпозиции
- **Нет CI/CD** — тесты запускаются вручную
- **Бенчмарк keyword-based** — грубая метрика, не ловит парафразы

### Roadmap

1. **P0:** Декомпозиция `llm_cli.py` на ModelManager + SessionManager + CostCalculator
2. **P1:** Async-поддержка для параллельных embedding-запросов
3. **P1:** Улучшенный бенчмарк: semantic similarity + faithfulness proxy
4. **P2:** RAG hybrid search (BM25 + vector)
5. **P2:** OpenTelemetry для трассировки
6. **P3:** Multi-agent coordination с общим планировщиком
