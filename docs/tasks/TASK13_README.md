# День 13. FSM-агент написания статей

## Задача

Реализовать состояние задачи как конечный автомат (FSM):

- **этап задачи** — текущая фаза
- **текущий шаг** — позиция внутри фазы
- **ожидаемое действие** — что именно делает агент на этом шаге

Паузировать на любом этапе и продолжить без повторных объяснений.

## Состояния

```
planning → execution → validation → done
```

| Фаза | Шаги | Описание |
|------|------|---------|
| `planning` | choose_topic → analyze_topic → create_outline | Агент выбирает тему, анализирует, строит план |
| `execution` | write_intro → write_body → write_conclusion | Написание статьи по плану |
| `validation` | check_structure → score_quality → final_edit | Проверка и финальная редактура |
| `done` | — | Статья готова |

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    ArticleFSMAgent                           │
│                                                             │
│  ┌─────────────┐    ┌──────────────────────────────────┐   │
│  │  TaskState   │    │         FSM Loop                  │   │
│  │             │    │                                   │   │
│  │ - phase      │───▶│  while phase != DONE:             │   │
│  │ - step       │    │    step = get_current_step_info() │   │
│  │ - expected_  │    │    result = execute_step(LLM)     │   │
│  │   action     │    │    record_step_result(result)     │   │
│  │ - article_   │    │    save(state)                    │   │
│  │   data       │    │    advance_step()                 │   │
│  │ - paused     │    │                                   │   │
│  └─────────────┘    └──────────────────────────────────┘   │
│          │                                                   │
│  ┌───────▼──────────────────────────────────────────────┐   │
│  │              TaskStateStorage                         │   │
│  │   tasks/index.json + tasks/<task_id>.json             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Новые модули

### `news_agent/fsm_state.py`

| Класс/объект | Назначение |
|---|---|
| `TaskPhase` | Enum: PLANNING / EXECUTION / VALIDATION / DONE |
| `PHASE_STEPS` | Шаги каждой фазы с описанием и expected_action |
| `PHASE_TRANSITIONS` | Граф переходов между фазами |
| `StepResult` | Результат выполнения одного шага |
| `TaskState` | Полное состояние задачи: фаза, шаг, данные, пауза |
| `TaskStateStorage` | JSON-хранилище: save / load / list / find_latest_paused |

### `news_agent/fsm_agent.py`

| Класс/функция | Назначение |
|---|---|
| `ArticleFSMAgent` | Агент: запуск, пауза, возобновление, вызов LLM по шагам |
| `_build_step_prompt()` | Строит промпт для текущего шага из накопленных данных |
| `_build_full_article()` | Собирает финальный текст из article_data |
| `STEP_PROMPTS` | Шаблоны промптов для каждого из 9 шагов |

## Паузировка и возобновление

### Как работает пауза

1. `Ctrl+C` во время выполнения → graceful pause (перехватывается через `signal.SIGINT`)
2. `--pause-after <phase>` → автоматическая пауза после завершения фазы
3. Ошибка LLM API → автоматическая пауза с причиной

### Что сохраняется при паузе

- Текущая фаза и шаг
- Все уже написанные фрагменты статьи (`article_data`)
- Причина паузы
- История выполненных шагов (`step_results`)

### Возобновление

При `--resume <task_id>` агент:
1. Загружает состояние из `tasks/<task_id>.json`
2. Снимает флаг `paused`
3. Продолжает с **того же шага**, где остановился
4. Не переспрашивает и не повторяет уже выполненные шаги

## Запуск

### Новая задача (тема на выбор агента)

```bash
python3 demo_article_fsm.py
```

### Новая задача с конкретной темой

```bash
python3 demo_article_fsm.py --topic "Будущее генеративных нейросетей"
```

### Запуск с паузой после planning

```bash
python3 demo_article_fsm.py --pause-after planning
# → выведет task_id

python3 demo_article_fsm.py --resume <task_id>
# → продолжит с execution
```

### Список всех задач

```bash
python3 demo_article_fsm.py --list
```

### Статус конкретной задачи

```bash
python3 demo_article_fsm.py --status <task_id>
```

## Тесты

```bash
python3 -m pytest test_article_fsm.py -v
```

45 тестов покрывают:

- `TaskPhase` — значения, переходы, шаги
- `StepResult` — создание, сериализация
- `TaskState` — создание, advance_step, pause/resume, сериализация, форматтеры
- `TaskStateStorage` — save/load, индекс, поиск паузированных
- `_build_full_article` / `_build_step_prompt` — вспомогательные функции
- `ArticleFSMAgent` — инициализация, парсинг JSON, выполнение шагов, retry, пауза, полный прогон

## Пример использования в коде

```python
from openai import OpenAI
from news_agent.fsm_state import TaskStateStorage
from news_agent.fsm_agent import ArticleFSMAgent

client = OpenAI(base_url="https://foundation-models.api.cloud.ru/v1", api_key="...")
storage = TaskStateStorage()
agent = ArticleFSMAgent(client=client, model="zai-org/GLM-4.7", storage=storage)

# Запуск новой задачи
state = agent.run(topic="AI в здравоохранении")
print(state.phase)  # done

# Или с паузой и возобновлением
agent.pause(state, reason="перерыв")
state2 = agent.run(task_id=state.task_id)  # продолжит с того же места
```

## Структура хранилища

```
tasks/
    index.json              — индекс всех задач
    <task_id>.json          — полное состояние задачи
```

## Файлы

| Файл | Описание |
|------|---------|
| `news_agent/fsm_state.py` | TaskPhase, TaskState, TaskStateStorage |
| `news_agent/fsm_agent.py` | ArticleFSMAgent, промпты, вспомогательные функции |
| `demo_article_fsm.py` | CLI-демо: new / resume / list / status / pause-after |
| `test_article_fsm.py` | 45 тестов |
| `TASK13_README.md` | Этот файл |
