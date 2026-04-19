# Промт для рефакторинга, ревью и улучшения проекта TinyAI

> **Задача:** Провести полный цикл code review, рефакторинга, улучшения документации, тестов и архитектуры проекта TinyAI. Документация — на русском языке. В конце провести исследование дальнейших улучшений.

> **Статус:** Фаза 1 выполнена (апрель 2026). Создан `core/`, `mcp_package/`, `pyproject.toml`, `Makefile`, `conftest.py`, новые тесты, обновлена документация. Подробности: `docs/ROADMAP.md`, `docs/RESEARCH.md`.

---

## Контекст проекта

**TinyAI** — учебный Python-проект, 25+ дней разработки, прошедший путь от базового LLM-промптинга до MCP-интеграции, RAG-системы, мульти-агентного оркестратора и планировщика задач.

**Стек:**
- Python 3.9+
- Cloud.ru Foundation Models API (OpenAI-compatible)
- Модели: `zai-org/GLM-4.7` (reasoning), `zai-org/GLM-4.7-Flash`
- OpenAI Python SDK
- MCP (Model Context Protocol) — stdio transport
- SQLite (scheduler), JSON-файлы (память, сессии, посты)
- pytest, tiktoken

**Корневые модули:**
```
llm_cli.py, chat_cli.py, intent_router.py
mcp_server.py, mcp_client.py, mcp_agent.py
mcp_router.py, mcp_registry.py
mcp_pipeline.py, mcp_pipeline_bridge.py, mcp_pipeline_server.py
mcp_orchestrator_agent.py, mcp_scheduler_agent.py, mcp_scheduler_server.py
scheduler_daemon.py, pipeline_templates.py
demo_day17_full.py, demo_day20_full.py
```

**Пакеты:**
```
news_agent/          — агент новостей, pipeline, роли, инструменты, FSM, сессии, токены
journalist_agent/    — FSM-агент журналиста, инварианты, workflow
memory_agent/        — трёхслойная память, профили, персонализация
rag/                 — RAG: chunker, embedder, indexer, reranker, query_rewrite, citations
pipeline/            — DB-слой pipeline
scheduler/           — планировщик задач (SQLite)
```

**Тесты** (23 файла в `tests/`), **демо** (20+ файлов в `demos/`), **бенчмарки** (`benchmarks/`).

---

## ЧАСТЬ 1 — CODE REVIEW

### 1.1 Статический анализ и стиль

1. Пройди по всем `.py` файлам проекта и выяви:
   - Нарушения PEP 8 (именование, длина строк, импорты)
   - Неиспользуемые импорты и переменные
   - Дублирование кода (одинаковая логика в разных модулях)
   - Мёртвый код (функции/методы, которые нигде не вызываются)
   - Magic numbers и hardcoded-строки, которые должны быть константами
   - Отсутствие type hints там, где они уместны

2. Обрати особое внимание на файлы-"свалки" в корне проекта:
   - `_write_rag_agent.py` — зачем underscore-префикс? нужен ли файл?
   - `demo_day17_full.py`, `demo_day20_full.py` — почему демо лежат в корне, а не в `demos/`?
   - Множество `mcp_*.py` файлов в корне — нет ли смысла вынести в пакет `mcp/`?

### 1.2 Архитектурные проблемы

1. **God-объекты:** Есть ли классы, которые делают слишком много? (> 500 строк, > 15 методов)
2. **Circular imports:** Проверь наличие циклических зависимостей между пакетами
3. **Coupling:** Как связаны `news_agent`, `journalist_agent`, `memory_agent`, `rag`? Есть ли чёткие интерфейсы между ними?
4. **Configuration management:** Конфигурация разбросана по коду или есть центральный конфиг?
5. **Error handling:** Как обрабатываются ошибки API? Есть ли retry-логика? Есть ли единый способ логирования ошибок?
6. **Concurrency:** MCP использует stdio, scheduler — SQLite. Есть ли race conditions?

### 1.3 Безопасность (OWASP)

1. `.env` файл — проверь, что он в `.gitignore` и нет случайных коммитов с секретами
2. Десериализация JSON из файлов без валидации — есть ли риск?
3. Subprocess-вызовы (stdio MCP) — правильно ли экранируются аргументы?
4. SQLite в scheduler — есть ли SQL-инъекции (параметризованные запросы vs f-strings)?

---

## ЧАСТЬ 2 — РЕФАКТОРИНГ

### 2.1 Реструктуризация корневой директории

**Предложи конкретный план переноса файлов:**

```
# Текущее (проблема: корень захламлён):
mcp_server.py, mcp_client.py, mcp_agent.py, mcp_router.py,
mcp_registry.py, mcp_pipeline.py, mcp_pipeline_bridge.py,
mcp_pipeline_server.py, mcp_orchestrator_agent.py,
mcp_scheduler_agent.py, mcp_scheduler_server.py

# Предлагаемое:
mcp/
  __init__.py
  server.py          # бывший mcp_server.py
  client.py          # бывший mcp_client.py
  agent.py           # бывший mcp_agent.py
  router.py
  registry.py
  pipeline/
    __init__.py
    pipeline.py
    bridge.py
    server.py
    templates.py
  scheduler/
    __init__.py
    agent.py
    server.py
```

Выполни рефакторинг с сохранением обратной совместимости через `__init__.py` реэкспорты или обнови все импорты.

### 2.2 Устранение дублирования

Найди и объедини:
1. Логика подсчёта токенов — возможно есть в `news_agent/token_counter.py` и дублируется в других местах
2. Логика работы с OpenAI API (заголовки, base_url, retry) — должна быть в одном месте, например `core/llm_client.py`
3. Чтение/запись JSON-файлов — есть ли единый util?
4. Логирование — используется ли `logging` модуль или везде `print()`?

### 2.3 Конфигурация

Создай центральный конфиг `config.py` или `core/config.py`:
```python
# Пример структуры
from dataclasses import dataclass
import os

@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    default_model: str
    default_temperature: float
    max_retries: int

@dataclass  
class AppConfig:
    llm: LLMConfig
    scheduler_db_path: str
    memory_data_path: str
    sessions_path: str
    posts_path: str
    logs_path: str

def load_config() -> AppConfig:
    ...
```

### 2.4 Типизация

Добавь `py.typed` маркер и постепенно внедри type hints:
1. Все публичные функции и методы должны иметь аннотации параметров и возвращаемого значения
2. Используй `TypedDict` для словарей с известной структурой (сообщения OpenAI API, JSON-конфиги)
3. Введи Protocol-интерфейсы для агентов (чтобы `NewsAgent`, `JournalistFSMAgent`, `MemoryAgent` реализовывали общий интерфейс)

```python
# Пример Protocol
from typing import Protocol, AsyncIterator

class BaseAgent(Protocol):
    def run(self, task: str) -> str: ...
    def run_stream(self, task: str) -> AsyncIterator[str]: ...
    def reset(self) -> None: ...
```

---

## ЧАСТЬ 3 — ДОКУМЕНТАЦИЯ (на русском языке)

### 3.1 README.md — полная перезапись

Перепиши `README.md` на русском языке со следующими разделами:

```markdown
# TinyAI

## Описание проекта
## Архитектура (ASCII или mermaid-диаграмма)
## Быстрый старт (5 минут до первого запуска)
## Структура директорий (с пояснением каждого модуля)
## Конфигурация (.env переменные)
## Запуск отдельных компонентов
  - CLI-чат
  - News Agent
  - Journalist FSM Agent
  - Memory Agent  
  - RAG-система
  - MCP-сервер и клиент
  - Планировщик
## Тестирование
## Известные ограничения
## Дорожная карта
```

### 3.2 Docstrings для всех публичных модулей, классов и функций

Формат — Google Style docstrings на русском языке:

```python
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Разбивает текст на перекрывающиеся чанки для RAG-индексации.
    
    Args:
        text: Исходный текст для разбивки.
        chunk_size: Максимальный размер чанка в токенах.
        overlap: Количество токенов перекрытия между соседними чанками.
        
    Returns:
        Список строк-чанков.
        
    Raises:
        ValueError: Если chunk_size <= overlap.
    """
```

**Приоритет по пакетам:**
1. `rag/` — chunker, embedder, indexer, search, reranker
2. `memory_agent/` — memory, agent, profile
3. `journalist_agent/` — fsm_agent, invariants, workflow
4. `news_agent/` — agent, pipeline, tools, session_manager
5. Корневые: `mcp_server.py`, `scheduler_daemon.py`

### 3.3 docs/ — создай/обнови файлы

**Создай следующие документы на русском:**

#### `docs/ARCHITECTURE.md` — полная перезапись
```markdown
# Архитектура TinyAI

## Общая схема компонентов
[Mermaid-диаграмма с компонентами и их связями]

## Слои системы
1. LLM-слой (API-клиент, модели, конфиг)
2. Агентный слой (news, journalist, memory)
3. Инфраструктурный слой (RAG, MCP, scheduler, pipeline)
4. Слой персистентности (JSON-файлы, SQLite)
5. CLI/интерфейсный слой

## Поток данных для каждого сценария
### Сценарий 1: Генерация новостного поста
### Сценарий 2: FSM-агент журналиста
### Сценарий 3: RAG-запрос с переформулировкой
### Сценарий 4: MCP-вызов через агента
```

#### `docs/MODULES.md` — справочник по модулям
Для каждого модуля: назначение, основные классы/функции, зависимости, пример использования.

#### `docs/TESTING.md` — руководство по тестированию (обнови существующий)
```markdown
## Запуск тестов
## Структура тестов (что тестирует каждый файл)
## Мок-стратегия (как мокируются API-вызовы)
## Coverage (как замерить покрытие)
## Написание новых тестов (соглашения)
```

#### `docs/SETUP.md` — детальная инструкция по настройке
```markdown
## Требования (Python версия, зависимости)
## Установка
## Настройка .env (все переменные с описанием)
## Проверка установки
## Типичные проблемы и решения
```

#### `docs/CHANGELOG.md` — история изменений
Восстанови по дням разработки (День 1–20+) ключевые добавленные возможности.

---

## ЧАСТЬ 4 — ТЕСТЫ

### 4.1 Аудит существующих тестов

Для каждого из 23 тестовых файлов определи:
- Реальное покрытие (unit/integration/e2e)
- Используют ли они реальный API или моки
- Насколько тесты хрупкие (зависят от порядка, внешнего состояния)
- Есть ли тесты, которые всегда падают или всегда пропускаются

### 4.2 Улучшение тестовой инфраструктуры

1. **Создай `tests/conftest.py`** с общими фикстурами:
```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_llm_client():
    """Мок OpenAI-совместимого клиента."""
    with patch('openai.OpenAI') as mock:
        mock.return_value.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="тестовый ответ"))]
        )
        yield mock

@pytest.fixture
def temp_data_dir(tmp_path):
    """Временная директория для тестовых данных."""
    return tmp_path

@pytest.fixture
def sample_article_text():
    return "Тестовая статья о технологиях искусственного интеллекта..."
```

2. **Добавь маркеры pytest** в `pytest.ini`:
```ini
[pytest]
markers =
    unit: чистые юнит-тесты без внешних зависимостей
    integration: тесты с реальными файловыми операциями
    e2e: end-to-end тесты, требующие API-ключ
    slow: медленные тесты (> 5 секунд)
```

3. **Создай `tests/factories.py`** — фабрики тестовых данных:
```python
def make_message(role="user", content="тест") -> dict: ...
def make_conversation(n=3) -> list[dict]: ...
def make_journalist_task(topic="AI") -> dict: ...
def make_rag_document(text=None) -> dict: ...
```

### 4.3 Недостающие тесты — создай:

| Файл | Что тестировать |
|---|---|
| `tests/test_rag_chunker.py` | chunk_text: размер, перекрытие, граничные случаи |
| `tests/test_rag_embedder.py` | embed_text: размер вектора, мок API |
| `tests/test_rag_search.py` | semantic_search: top-k, пороговая фильтрация |
| `tests/test_memory_agent.py` | short/working/long-term память, TTL, сброс |
| `tests/test_scheduler.py` | создание задач, выполнение, retry, отмена |
| `tests/test_config.py` | загрузка конфига из .env, defaults |
| `tests/test_intent_router.py` | классификация намерений |
| `tests/test_pipeline_templates.py` | загрузка и валидация шаблонов |

### 4.4 Покрытие кода

Настрой и запусти coverage:
```bash
pip install pytest-cov
pytest --cov=. --cov-report=html --cov-report=term-missing \
       -m "unit or integration" tests/
```
Цель: покрытие > 70% для core-модулей (news_agent, rag, memory_agent).

---

## ЧАСТЬ 5 — КАЧЕСТВО КОДА

### 5.1 Инструменты линтинга

Добавь в проект:

**`pyproject.toml`** (или `setup.cfg`):
```toml
[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "N", "UP"]
ignore = ["E501"]  # строки длиннее 100 — warning, не error

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

**`Makefile`** для удобства разработки:
```makefile
.PHONY: lint test test-unit test-cov format

lint:
	ruff check .
	mypy news_agent/ rag/ memory_agent/ journalist_agent/

format:
	ruff format .

test:
	pytest tests/ -v

test-unit:
	pytest tests/ -v -m unit

test-cov:
	pytest tests/ --cov=. --cov-report=html -m "unit or integration"
	open htmlcov/index.html

run-cli:
	python llm_cli.py

run-journalist:
	python demos/run_3topics.py --model zai-org/GLM-4.7
```

### 5.2 Pre-commit хуки

Создай `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-json
      - id: check-yaml
      - id: detect-private-key
```

---

## ЧАСТЬ 6 — ИССЛЕДОВАНИЕ ДАЛЬНЕЙШИХ УЛУЧШЕНИЙ

### 6.1 Технические долги (Technical Debt)

Проведи аудит и составь список с приоритетами:

| Приоритет | Область | Проблема | Оценка сложности |
|---|---|---|---|
| Критично | | | |
| Высокий | | | |
| Средний | | | |
| Низкий | | | |

**Что анализировать:**
- Места с `# TODO`, `# FIXME`, `# HACK` комментариями
- Функции длиннее 50 строк без разбивки
- Классы с > 10 зависимостями
- Отсутствие retry/backoff при API-вызовах
- Отсутствие timeout при HTTP-вызовах
- Неструктурированное логирование (`print()` вместо `logging`)

### 6.2 Функциональные улучшения

Исследуй и предложи с обоснованием:

**A. RAG-система:**
- [ ] Гибридный поиск (BM25 + векторный) — увеличит точность retrieval
- [ ] Persistent vector store (ChromaDB, Qdrant) вместо JSON — масштабируемость
- [ ] Re-ranking с cross-encoder — повышение quality@k
- [ ] Metadata filtering — поиск по дате, источнику, тегам
- [ ] Streaming RAG ответов

**B. Агентная система:**
- [ ] Multi-agent coordination (агенты работают параллельно)
- [ ] Agent memory sharing (общая долгосрочная память между агентами)
- [ ] Tool use самоисправление (retry если tool вернул ошибку)
- [ ] Планирование задач через LLM (не только FSM)
- [ ] Human-in-the-loop паузы в FSM

**C. MCP-интеграция:**
- [ ] HTTP/SSE transport (вместо только stdio) — для remote MCP серверов
- [ ] MCP аутентификация и авторизация инструментов
- [ ] Динамическое обнаружение MCP-серверов

**D. Производительность:**
- [ ] Async/await для параллельных LLM-вызовов (asyncio + openai async client)
- [ ] Кэширование эмбеддингов (не пересчитывать одинаковые тексты)
- [ ] Батчинг embeddings API вызовов
- [ ] Connection pooling для API-клиента

**E. Наблюдаемость:**
- [ ] OpenTelemetry трейсинг LLM-вызовов
- [ ] Метрики latency, token usage, error rate (Prometheus-совместимые)
- [ ] Structured logging (JSON-формат) для production
- [ ] Dashboard для мониторинга агентов

### 6.3 Инфраструктурные улучшения

**A. Деплой:**
- Докеризация проекта (`Dockerfile`, `docker-compose.yml`)
- CI/CD pipeline (GitHub Actions: lint → test → build)
- Возможность запуска как systemd-сервис (scheduler_daemon)

**B. Конфигурация:**
- Pydantic Settings для валидации конфига
- Поддержка нескольких окружений (dev/staging/prod через `.env.{env}`)
- Secrets management (не просто .env файл)

**C. База данных:**
- Миграции для SQLite (Alembic или простые SQL-скрипты)
- Возможность переключения на PostgreSQL

### 6.4 Исследовательские направления

Предложи 3–5 экспериментов, которые можно провести в рамках этого проекта:

1. **Сравнение стратегий RAG:**
   - Гипотеза: гибридный поиск (BM25 + векторный) даёт +15–20% к recall@5 по сравнению с чисто векторным
   - Метрика: recall@5, precision@5, MRR на тестовой выборке из `rag_data/`
   - Реализация: добавить BM25 в `rag/search.py`, сравнить с `benchmarks/`

2. **Влияние размера чанков на качество ответов:**
   - Гипотеза: чанки 256 токенов с overlap=64 дают лучшее качество для коротких фактических вопросов, чем 512/128
   - Метрика: faithfulness (ответ опирается на контекст), relevance
   - Реализация: параметризованный бенчмарк в `benchmarks/`

3. **Эффективность компрессии контекста:**
   - Гипотеза: суммаризация при превышении 80% context window не ухудшает качество ответов относительно полного контекста
   - Метрика: coherence score (оценивается LLM-судьёй)

4. **Сравнение FSM vs свободное планирование:**
   - Гипотеза: FSM-агент журналиста более предсказуем, но менее гибок чем ReAct-агент с free-form planning
   - Метрика: success rate, шаги до завершения, human preference score

5. **Персонализация и качество генерации:**
   - Гипотеза: агент с user profile (аналитик vs журналист) получает значимо более высокие оценки от соответствующей аудитории
   - Метрика: A/B тест с оценщиком LLM

---

## ЧАСТЬ 7 — ПЛАН ВЫПОЛНЕНИЯ

Выполняй задачи в следующем порядке (не пропускай шаги):

### Фаза 1 — Анализ (не вносить изменений, только читать и составлять список)
- [ ] Прочитать все `.py` файлы в корне проекта
- [ ] Прочитать `__init__.py` каждого пакета
- [ ] Прочитать по одному ключевому файлу из каждого пакета
- [ ] Составить список всех найденных проблем с указанием файла и строки

### Фаза 2 — Рефакторинг структуры
- [ ] Создать пакет `mcp/` и переместить mcp-файлы
- [ ] Переместить демо из корня в `demos/`
- [ ] Обновить все импорты
- [ ] Убедиться что тесты проходят: `pytest tests/ -x`

### Фаза 3 — Документация
- [ ] Перезаписать README.md на русском
- [ ] Написать docstrings для `rag/` (приоритет 1)
- [ ] Написать docstrings для `memory_agent/` (приоритет 2)
- [ ] Написать docstrings для `journalist_agent/` (приоритет 3)
- [ ] Создать/обновить все `docs/*.md` файлы

### Фаза 4 — Тесты
- [ ] Создать `tests/conftest.py`
- [ ] Создать `tests/factories.py`
- [ ] Написать недостающие тесты (начиная с `test_rag_chunker.py`)
- [ ] Запустить coverage, добиться > 70%

### Фаза 5 — Инструменты качества
- [ ] Добавить `pyproject.toml` с ruff + mypy конфигом
- [ ] Создать `Makefile`
- [ ] Настроить `.pre-commit-config.yaml`
- [ ] Исправить все критичные linting-ошибки

### Фаза 6 — Исследование
- [ ] Составить приоритизированный список технических долгов
- [ ] Написать `docs/ROADMAP.md` с функциональными улучшениями
- [ ] Описать 5 экспериментов в `docs/RESEARCH.md`

---

## Критерии готовности (Definition of Done)

- [ ] `pytest tests/ -m "unit or integration"` — все тесты зелёные
- [ ] `ruff check .` — 0 ошибок
- [ ] Coverage ≥ 70% для `rag/`, `memory_agent/`, `journalist_agent/`
- [ ] Все публичные функции/классы в пакетах имеют docstrings на русском
- [ ] README.md читается и понятен новому разработчику за 10 минут
- [ ] `docs/ARCHITECTURE.md` содержит актуальную mermaid-диаграмму
- [ ] `docs/ROADMAP.md` содержит приоритизированный список улучшений
- [ ] `docs/RESEARCH.md` содержит 5 описанных экспериментов
- [ ] Нет файлов с секретами (API-ключами) в git-истории
- [ ] `Makefile` работает: `make lint`, `make test`, `make test-cov`

---

## Дополнительные указания

1. **Не ломай рабочий код** — каждое изменение структуры должно сопровождаться обновлением импортов и проверкой тестов
2. **Сохраняй обратную совместимость** — если публичный API модуля меняется, добавь deprecation warning
3. **Документация — на русском языке** — docstrings, README, все docs/*.md
4. **Комментарии в коде — на русском** — для нового кода; старый код не трогать если нет другой причины
5. **Не удаляй demos/** — они служат живой документацией
6. **Бенчмарки должны работать без реального API** — добавь режим mock-data
7. При нахождении security-проблем — **исправляй немедленно**, не откладывай в backlog
