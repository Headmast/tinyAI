# Сравнение стратегий chunking

## 1. Статистика чанков

| Метрика | fixed_size | structure |
|---------|-----------|-----------|
| Кол-во чанков | 184 | 754 |
| Avg токенов | 480.6 | 110.9 |
| Median токенов | 510.0 | 81.0 |
| Min токенов | 73 | 2 |
| Max токенов | 512 | 994 |
| Std dev токенов | 88.1 | 121.4 |
| Всего токенов | 88,425 | 83,593 |
| Всего символов | 229,754 | 219,833 |
| Источников | 27 | 27 |
| Секций | 0 | 746 |

## 2. Retrieval Precision@5

| Метрика | fixed_size | structure |
|---------|-----------|-----------|
| Precision@5 | 100.0% | 93.3% |
| Hits / Total | 15/15 | 14/15 |
| Avg similarity | 0.5235 | 0.5921 |

## 3. Детализация по запросам

### fixed_size

| # | Запрос | Hit | Score | Top Source |
|---|--------|-----|-------|------------|
| 1 | Как работает memory_agent и управление памятью? | ✅ | 0.6213 | docs/ARCHITECTURE.md |
| 2 | Какие инструменты и tools у MCP Agent? | ✅ | 0.6290 | docs/workflow/task_summaries.md |
| 3 | Как считаются токены и работает TokenCounter? | ✅ | 0.4916 | docs/tasks/TASK8_README.md |
| 4 | Стратегии контекста SlidingWindow StickyFacts Bran | ✅ | 0.5274 | docs/tasks/TASK10_PLAN.md |
| 5 | FSM Agent и конечный автомат ArticleFSMAgent | ✅ | 0.5113 | docs/tasks/TASK13_README.md |
| 6 | Оркестрация нескольких MCP серверов MCPRegistry MC | ✅ | 0.5501 | docs/tasks/TASK20_README.md |
| 7 | Pipeline composition и композиция пайплайнов | ✅ | 0.3198 | docs/tasks/TASK19_README.md |
| 8 | Температура модели и влияние на генерацию | ✅ | 0.4436 | docs/TEMPERATURE_GUIDE.md |
| 9 | Модели OpenAI GPT цены стоимость pricing | ✅ | 0.6403 | docs/MODELS_INFO.md |
| 10 | Инварианты журналиста JournalistAgent constraints | ✅ | 0.5678 | docs/tasks/TASK14_README.md |
| 11 | Персонализация UserProfile профили пользователей | ✅ | 0.4732 | docs/tasks/TASK12_README.md |
| 12 | Компрессия контекста ContextCompressor sliding win | ✅ | 0.4971 | docs/workflow/task_summaries.md |
| 13 | Новостной агент news agent pipeline генерация пост | ✅ | 0.5848 | docs/tasks/TASK6_README.md |
| 14 | Тестирование pytest тесты unit test | ✅ | 0.5357 | docs/TEST_SUMMARY.md |
| 15 | Сессии разговора ConversationSession SessionStorag | ✅ | 0.4591 | docs/ARCHITECTURE.md |

### structure

| # | Запрос | Hit | Score | Top Source |
|---|--------|-----|-------|------------|
| 1 | Как работает memory_agent и управление памятью? | ✅ | 0.6843 | docs/ARCHITECTURE.md |
| 2 | Какие инструменты и tools у MCP Agent? | ✅ | 0.6786 | docs/tasks/TASK17_README.md |
| 3 | Как считаются токены и работает TokenCounter? | ✅ | 0.5928 | docs/tasks/TASK8_README.md |
| 4 | Стратегии контекста SlidingWindow StickyFacts Bran | ✅ | 0.5987 | docs/workflow/task_summaries.md |
| 5 | FSM Agent и конечный автомат ArticleFSMAgent | ✅ | 0.5912 | docs/ARCHITECTURE.md |
| 6 | Оркестрация нескольких MCP серверов MCPRegistry MC | ✅ | 0.6191 | docs/tasks/TASK20_README.md |
| 7 | Pipeline composition и композиция пайплайнов | ✅ | 0.3748 | docs/tasks/TASK19_README.md |
| 8 | Температура модели и влияние на генерацию | ✅ | 0.4955 | docs/TEMPERATURE_GUIDE.md |
| 9 | Модели OpenAI GPT цены стоимость pricing | ✅ | 0.5893 | docs/MODELS_INFO.md |
| 10 | Инварианты журналиста JournalistAgent constraints | ❌ | 0.6305 | docs/workflow/task_summaries.md |
| 11 | Персонализация UserProfile профили пользователей | ✅ | 0.6026 | docs/workflow/task_summaries.md |
| 12 | Компрессия контекста ContextCompressor sliding win | ✅ | 0.5806 | docs/tasks/TASK9_README.md |
| 13 | Новостной агент news agent pipeline генерация пост | ✅ | 0.6436 | docs/workflow/task_summaries.md |
| 14 | Тестирование pytest тесты unit test | ✅ | 0.6246 | docs/tasks/TASK20_README.md |
| 15 | Сессии разговора ConversationSession SessionStorag | ✅ | 0.5746 | docs/ARCHITECTURE.md |

## 4. Анализ

### Fixed-size chunking:
- Создаёт **184** чанков стабильного размера (~481 токенов)
- Стандартное отклонение: 88.1 — **равномерные** чанки
- Не учитывает структуру документа (section = N/A)

### Structure-based chunking:
- Создаёт **754** чанков с переменным размером (2–994 токенов)
- Стандартное отклонение: 121.4 — **неравномерные** но семантически целостные чанки
- Сохраняет структуру: 746 уникальных секций

### Вывод: **Fixed-size** показывает лучшую retrieval precision (равномерные чанки обеспечивают более стабильное покрытие документов).