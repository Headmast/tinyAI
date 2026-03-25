# День 8 — Работа с токенами

## Что реализовано

### Новый модуль: `news_agent/token_counter.py`

Ядро всей токен-логики. Три класса:

#### `TokenCounter`
Точный подсчёт токенов через `tiktoken` (OpenAI) с автоматическим fallback на эвристику `chars÷4`.

```python
from news_agent.token_counter import TokenCounter

counter = TokenCounter(model="gpt-5.4")
print(counter.is_exact)        # True — tiktoken доступен
print(counter.method_label)    # "tiktoken" или "~chars÷4"

n = counter.count_text("Hello, world!")            # 4 токена
d = counter.count_messages(messages)               # dict с деталями
r = counter.count_response("Some response text")   # completion tokens

# Полная разбивка запроса:
breakdown = counter.count_request_breakdown(messages, response_text)
# {
#   "prompt_tokens": 156,
#   "completion_tokens": 87,
#   "total_tokens": 243,
#   "history_tokens": 120,   # токены предыдущих ходов
#   "current_msg_tokens": 36, # токены последнего сообщения
#   "overhead": 15,           # overhead на структуру messages[]
#   "per_message": [34, 22, 52, ...],
# }
```

Почему `cl100k_base` для GLM-моделей: это кодировка GPT-4, наиболее близкая по словарю к современным многоязычным моделям. Погрешность <15% против кириллического текста (vs 40% у chars÷4).

#### `DialogTokenTracker`
Отслеживает рост токенов по ходу диалога — снимок (`TurnSnapshot`) после каждого хода.

```python
from news_agent.token_counter import DialogTokenTracker

tracker = DialogTokenTracker(model="gpt-5-nano", max_tokens=128_000)

for role, text in dialog_turns:
    messages.append({"role": role, "content": text})
    snap = tracker.add_turn(role, text, messages)
    # snap.total_tokens, snap.growth, snap.growth_pct

print(tracker.format_growth_table())
print(tracker.format_context_bar())
# 🟢 [████░░░░░░░░░░░░░░░░░░░░] 2.1%  2,734 / 128,000 токенов

cost = tracker.cumulative_cost(prompt_price=0.0002, completion_price=0.00125)
```

#### `TokenBudget`
Управление бюджетом контекстного окна — детектирует overflow **до** отправки запроса.

```python
from news_agent.token_counter import TokenBudget

budget = TokenBudget(context_limit=128_000, max_completion=4_000)

eff = budget.effective_max_completion(prompt_tokens=125_000)  # → 3_000
is_ov, msg = budget.check_overflow(prompt_tokens=130_000)
# is_ov=True, msg="OVERFLOW: prompt 130,000 ≥ лимит 128,000 токенов."
```

---

### Обновления `AgentLoop` (`news_agent/agent.py`)

```python
agent = AgentLoop(
    client=client,
    model="zai-org/GLM-4.7-Flash",
    max_completion_tokens=4_000,   # ← управление длиной ответа (было хардкод)
    context_limit=128_000,          # ← размер контекстного окна
)
result = agent.run(task)
```

**Новое поведение при каждой итерации:**
```
--- Итерация 3 ---
  ⚠️  Токены: WARNING: prompt 98,432 токенов (76.9% окна)...
  📊 Токены: prompt=98,432 (76.9%) | completion=312 | max_completion=3_500
```

**Итоговая таблица в конце работы агента:**
```
─────────────────────────────────────────────────────────────────
  ИТОГ ТОКЕНОВ ПО ИТЕРАЦИЯМ  (tiktoken)
  Iter    Prompt  Completion     Total  Msgs
  ───────────────────────────────────────────────────────
     1       456         234       690     2
     2     1 102         189     1 291     4
     3     1 987         312     2 299     7
```

Агент **автоматически останавливается** при реальном overflow контекста.

---

### Обновления чата (`llm_cli.py`)

- `_chat_send` использует `TokenCounter` вместо `chars÷4`
- В каждом ответе показывается метод подсчёта: `[tiktoken]` / `[~chars÷4]`
- Новая команда `tokens` внутри чата:

```
you> tokens

  Рост токенов по ходу диалога  [tiktoken]:
  ──────────────────────────────────────────────────────────────────
    Turn  Role       MsgTok       Δ    Total    Window
  ──────────────────────────────────────────────────────────────────
       1  user           12  +   19       19    0.01%
       2  assistant      45  +   52       71    0.06%
       3  user           18  +   25       96    0.07%
  ──────────────────────────────────────────────────────────────────
  Итого: 96 / 128,000 токенов  (0.07%)  осталось: 127,904
```

---

### `demo_tokens.py`

Автономная демонстрация (без API), запускается сразу:

```bash
python3 demo_tokens.py
```

**5 сценариев:**
1. Разбивка одного запроса (системный промпт / запрос / ответ / overhead)
2. Длинный диалог (10 ходов) — таблица роста токенов + стоимость
3. Управление `max_completion_tokens` — сравнение сценариев
4. Переполнение контекста с искусственным лимитом 800 токенов
5. Проекция стоимости при масштабировании (100 сессий/день)

---

## Тесты

```bash
python3 -m pytest test_token_counter.py -v
# 56 passed
```

| Класс                      | Тестов |
|----------------------------|--------|
| `TestTokenCounterProperties` | 4    |
| `TestCountText`             | 6     |
| `TestCountMessages`         | 7     |
| `TestCountRequestBreakdown` | 5     |
| `TestTokenCounterFallback`  | 4     |
| `TestDialogTokenTracker`    | 14    |
| `TestTokenBudget`           | 10    |
| `TestIntegration`           | 4     |
| **Итого**                   | **56** |

---

## Ключевые выводы по работе с токенами

### Рост стоимости в диалоге

Критически важно понимать: каждый API-вызов в диалоге включает **всю предыдущую историю**. Это означает:

```
Ход 1: prompt = сообщение_1
Ход 2: prompt = сообщение_1 + сообщение_2  ← растёт
Ход 3: prompt = сообщение_1 + ... + сообщение_3
```

10-ходовой диалог стоит не в 10 раз дороже одного ответа, а значительно дороже.

### Что ломается при переполнении

| Поведение модели          | Последствие                                    |
|--------------------------|------------------------------------------------|
| `context_length_exceeded` | API возвращает ошибку, запрос не выполнен     |
| Тихое усечение промпта   | Модель «забывает» начало диалога              |
| Деградация качества       | Модель теряет нить рассуждений                |

### Стратегии решения

- **Truncation** — удалить старые сообщения (теряем контекст)
- **Summarization** — сжать историю через отдельный LLM-вызов
- **RAG** — хранить историю в векторной БД, вставлять только релевантное
- **max_completion_tokens** — ограничить длину ответа для экономии

---

## Изменённые файлы

| Файл | Изменение |
|------|-----------|
| `news_agent/token_counter.py` | **новый** — TokenCounter, DialogTokenTracker, TokenBudget |
| `news_agent/agent.py` | max_completion_tokens/context_limit params, per-iteration breakdown, overflow stop |
| `news_agent/session_manager.py` | estimate_tokens() через TokenCounter вместо chars÷4 |
| `news_agent/__init__.py` | экспорт новых классов, версия 4.0.0 |
| `llm_cli.py` | TokenCounter в _chat_send, DialogTokenTracker в чате, команда `tokens` |
| `demo_tokens.py` | **новый** — автономная демонстрация 5 сценариев |
| `test_token_counter.py` | **новый** — 56 тестов |
| `requirements.txt` | добавлен `tiktoken>=0.7.0` |
