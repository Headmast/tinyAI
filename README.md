# TinyAI — News Agent + Диалоговый чат

Консольный AI-инструмент: **генерация новостных постов** через pipeline и **диалоговый чат** с историей сессий.

> **Стек:** Python 3.10+ · OpenAI API · Cloud.ru (GLM-4.7) · потоковый вывод · JSON-логирование

---

## Возможности

| Функция | Описание |
|---|---|
| **News Pipeline** | 5-шаговый конвейер: Planner → Researcher → Writer → Editor → SEO |
| **ReAct Agent** | Автономный агент с function calling и инструментами |
| **Batch** | Пакетная генерация постов из файла тем |
| **Chat Sessions** | Диалог с полной историей контекста, автосохранение |
| **Context Tracker** | Визуальный индикатор использования контекстного окна |
| **Session Resume** | Загрузка и продолжение сохранённых диалогов |
| **JSON Logging** | Полное логирование сессий в sessions/ |
| **Multi-model** | GLM-4.7-Flash, GLM-4.7, GPT-5-nano, GPT-5.4, GPT-5.4-mini |

---

## Быстрый старт

### 1. Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 2. Настройка .env

```env
CLOUD_API_KEY=ваш-ключ-cloud-ru      # для GLM-4.7-Flash (бесплатно)
OPENAI_API_KEY=sk-proj-ваш-ключ      # для GPT-5.x (платно)
```

> Достаточно одного ключа. GLM-4.7-Flash — **бесплатная** модель Cloud.ru.

### 3. Запуск

```bash
python3 llm_cli.py
```

---

## Генерация новостных постов

```
news-agent> generate Взлёт SpaceX Starship
news-agent> generate -t analysis Влияние ИИ на рынок труда
news-agent> agent Создай пост о GPT-5
news-agent> batch topics.txt
news-agent> history 5
news-agent> export abc12345 md
news-agent> template list
```

**Типы постов:** breaking · analysis · digest · social · press

**Форматы экспорта:** md · html · telegram · json · plain

---

## Диалоговый чат с историей

### Ключевая концепция: API vs Веб-чат

| | Веб-чат (ChatGPT) | API (прямой) |
|---|---|---|
| История | Сервер хранит автоматически | Нужно передавать явно каждый раз |
| Запрос | Только новое сообщение | Весь массив messages[] при каждом вызове |
| Память | Автоматическая | Реализуем сами через sessions/ |

### Команды сессий

```
news-agent> chat                        # новая сессия
news-agent> chat new Python обучение    # именованная сессия
news-agent> chat list                   # список всех сессий
news-agent> chat load a1b2c3d4          # загрузить по ID
news-agent> chat resume a1b2c3d4        # продолжить закрытую
news-agent> chat info a1b2c3d4          # детали + использование контекста
news-agent> chat delete a1b2c3d4        # удалить сессию
```

### Внутри чата

```
you> Объясни замыкания в Python
you> Покажи пример с декоратором
you> info                                # использование контекста
you> close                               # закрыть сессию
```

### Индикатор контекста (после каждого ответа)

```
Контекст: 1,234 / 128,000 токенов [████░░░░░░░░░░░░░░░░] 0.96%
   Токены ответа: prompt ~310 | completion ~42
```

🟢 < 50% · 🟡 50–79% · 🔴 >= 80% (рекомендуется новая сессия)

---

## Модели

| Модель | Провайдер | Цена prompt |
|---|---|---|
| zai-org/GLM-4.7-Flash | Cloud.ru | Бесплатно |
| zai-org/GLM-4.7 | Cloud.ru | Бесплатно |
| gpt-5-nano | OpenAI | $0.0002/1K |
| gpt-5.4 | OpenAI | $0.0025/1K |
| gpt-5.4-mini | OpenAI | $0.00075/1K |

```
news-agent> models          # список
news-agent> model gpt-5.4   # переключить
```

---

## Тестирование

```bash
pytest -v                       # все тесты
pytest test_sessions.py -v      # session manager
pytest test_llm_cli.py -v       # CLI функции
```

| Файл теста | Что проверяет |
|---|---|
| test_sessions.py | ConversationSession, SessionStorage, lifecycle |
| test_llm_cli.py | Модели, режимы, execute_mode, chat-функции |
| test_model_parameters.py | Параметры моделей |
| test_temperature.py | Диапазоны температуры |

---

## Структура проекта

```
tinyAI/
├── llm_cli.py                  # Главный CLI
├── requirements.txt
├── pytest.ini
│
├── news_agent/
│   ├── __init__.py             # Пакет v2.0.0
│   ├── session_manager.py      # ConversationSession + SessionStorage
│   ├── pipeline.py             # 5-шаговый NewsPipeline
│   ├── agent.py                # ReAct AgentLoop
│   ├── storage.py              # PostStorage
│   ├── formatter.py            # OutputFormatter
│   ├── roles.py                # Системные промпты
│   └── tools.py                # Инструменты агента
│
├── test_sessions.py            # Тесты session manager
├── test_llm_cli.py             # Тесты CLI
├── test_model_parameters.py
├── test_temperature.py
│
├── sessions/                   # Диалоговые сессии (не в VCS)
├── posts/                      # Сгенерированные посты (не в VCS)
├── logs/                       # Логи (не в VCS)
│
├── README.md
├── TASK7_README.md             # Диалоговые сессии и контекст
├── TASK6_README.md             # ReAct-агент
├── TASK3_README.md             # News pipeline
├── TASK2_README.md             # Режимы и форматирование
├── MODELS_INFO.md              # Справочник моделей
└── TEMPERATURE_GUIDE.md        # Руководство по температуре
```

---

## Решение проблем

**command not found: python3** — установите Python: `brew install python`

**CLOUD_API_KEY not found** — убедитесь что .env существует и содержит ключ

**AuthenticationError** — проверьте ключ и баланс у провайдера

**TypeError: proxies** — `python3 -m pip install -r requirements.txt --force-reinstall`

---

## Документация заданий

| Файл | Тема |
|---|---|
| TASK7_README.md | Диалоговые сессии, API vs веб-чат, контекстное окно |
| TASK6_README.md | ReAct-агент, function calling, инструменты |
| TASK3_README.md | News pipeline, 5 ролей, SEO |
| TASK2_README.md | Режимы форматирования, метапромптинг |
| MODELS_INFO.md | Сравнение моделей и цен |
| TEMPERATURE_GUIDE.md | Влияние температуры на качество |

---

> **Важно:** Никогда не коммитьте .env с API ключами в git!
