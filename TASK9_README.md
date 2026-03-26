# День 9 — Управление контекстом: сжатие истории

## Что реализовано

### Новый модуль: `news_agent/context_compressor.py`

Ядро механизма компрессии. Два класса:

#### `CompressionStats`
Статистика одного акта сжатия.

```python
from news_agent.context_compressor import CompressionStats

stats = CompressionStats(
    compression_index=1,
    messages_compressed=10,
    tokens_before=1200,
    tokens_after=350,
    summary_length=280,
)

print(stats.tokens_saved)        # 850
print(stats.compression_ratio)   # 0.29
print(stats.format())
# Сжатие #1: 10 сообщ. → 1 summary  |  ~850 токенов сэкономлено  |  сжатие 71%
```

#### `ContextCompressor`
Основной механизм скользящей компрессии.

```python
from news_agent.context_compressor import ContextCompressor

compressor = ContextCompressor(
    summarize_every=10,   # суммаризировать каждые 10 сообщений
    keep_last_n=6,        # хранить последние 6 сообщений «как есть»
    model="zai-org/GLM-4.7-Flash",
)

# Проверка необходимости сжатия
if compressor.needs_compression(session.messages):

    # Запуск суммаризации через LLM
    stats = compressor.compress(client, session.messages)
    print(stats.format())

# Получить сжатые сообщения для API
api_messages = compressor.get_compressed_messages(session.messages)
# → [system] + [📋 summary_msg] + [последние 6 сообщений]

# Сравнение токенов с/без сжатия
cmp = compressor.get_comparison(session.messages)
# {
#   "full_tokens": 4500,
#   "compressed_tokens": 800,
#   "tokens_saved": 3700,
#   "savings_pct": 82.2,
#   "compression_count": 3,
#   ...
# }
```

---

### Алгоритм скользящей компрессии

```
История до сжатия (20 сообщений, summarize_every=10, keep_last_n=6):
┌──────────────────────────────────────────────────────────────┐
│  u1 a1 u2 a2 u3 a3 u4 a4 u5 a5 │ u6 a6 u7 a7 u8 a8 u9 a9 … │
│   ← 10 сообщений на суммаризацию →   ← живой хвост 6 шт →  │
└──────────────────────────────────────────────────────────────┘
                    ↓  LLM генерирует summary
История после сжатия, отправляемая в API:
┌──────────────────────────────────────────────────────────────┐
│  [SYSTEM]  [📋 SUMMARY: u1-a5]  [u6] [a6] [u7] [a7] [u8] … │
└──────────────────────────────────────────────────────────────┘
```

**Ключевые свойства:**
- `summary_covers_up_to` — индекс в chat_messages[], до которого уже суммировано
- При повторной суммаризации новые summaries объединяются: `Summary 1 --- Summary 2`
- `_chat_only()` фильтрует только `user`/`assistant` — system-сообщения учитываются отдельно

---

### Интеграция в `ConversationSession`

```python
from news_agent.session_manager import ConversationSession

session = ConversationSession(model="zai-org/GLM-4.7-Flash")

# Включить компрессию
session.enable_compression(summarize_every=10, keep_last_n=6)

# Проверить и запустить сжатие (вызывается после каждого обмена)
stats = session.maybe_compress(client)
if stats:
    print(stats.format())

# get_messages_for_api() автоматически возвращает сжатую версию
messages = session.get_messages_for_api()

# Выключить (summaries сохраняются)
session.disable_compression()

# Сводка
info = session.get_compression_info()
# {
#   "enabled": True,
#   "compression_count": 2,
#   "messages_summarized": 20,
#   "total_tokens_saved": 1850,
#   "summarize_every": 10,
#   "keep_last_n": 6,
# }
```

**Новые поля в `ConversationSession`:**

| Поле | Тип | Описание |
|---|---|---|
| `compression_enabled` | `bool` | Флаг активности компрессии |
| `compressor` | `Optional[ContextCompressor]` | Экземпляр компрессора |

**Обратная совместимость:** старые JSON-файлы сессий без полей компрессии загружаются корректно — `compression_enabled=False`, `compressor=None`.

---

### Новые команды в `llm_cli.py`

**Внутри чата (`chat`, `chat new`, `chat load`):**

```
compress                 — показать статус компрессии и статистику
compress on              — включить компрессию (каждые 10 сообщений)
compress off             — выключить компрессию (summaries сохраняются)
compress compare         — сравнить токены с/без сжатия
```

Пример вывода `compress compare`:
```
───────────────────────────────────────────────────────
  СРАВНЕНИЕ: токены с компрессией vs без
───────────────────────────────────────────────────────
  Без компрессии:      4,520 токенов  (32 сообщения)
  Со сжатием:            780 токенов  (8 сообщений)
───────────────────────────────────────────────────────
  Экономия:            3,740 токенов  (82.7%)
  Сжатий выполнено:    2
───────────────────────────────────────────────────────
```

Пример автоматического триггера (после 10-го сообщения):
```
🔄 Сжатие #1: 10 сообщ. → 1 summary  |  ~920 токенов сэкономлено  |  сжатие 68%
  Последнее summary:
    Пользователь спрашивал о Python и работе с файлами. Важные моменты:
    использован pathlib, нужна обработка исключений FileNotFoundError...
```

---

### Сериализация

Состояние компрессора полностью сохраняется в JSON-файле сессии:

```json
{
  "session_id": "a1b2c3d4",
  "compression_enabled": true,
  "compressor": {
    "summarize_every": 10,
    "keep_last_n": 6,
    "model": "zai-org/GLM-4.7-Flash",
    "summaries": [
      "Краткое изложение первой части диалога...",
      "Краткое изложение второй части..."
    ],
    "summary_covers_up_to": 20,
    "compression_count": 2,
    "total_tokens_saved": 1850
  }
}
```

---

## Тесты: `test_context_compressor.py`

58 тестов, разбитых по 8 классам:

| Класс | Что покрывает |
|---|---|
| `TestCompressionStats` | tokens_saved, compression_ratio, format() |
| `TestContextCompressorCreation` | Параметры по умолчанию и кастомные |
| `TestNeedsCompression` | Пороговая логика (мало/ровно/много сообщений, после предыдущего сжатия) |
| `TestGetCompressedMessages` | Структура вывода: system + summary + recent |
| `TestCompress` | Режим с моком LLM: счётчики, тексты, вторая суммаризация, gpt-5-nano |
| `TestGetComparison` | Процент экономии, ключи ответа |
| `TestFormatStats` | Форматирование сводки и превью |
| `TestContextCompressorSerialization` | to_dict / from_dict / JSON |
| `TestConversationSessionCompression` | enable/disable, maybe_compress, API output, backward compat |

```
python3 -m pytest test_context_compressor.py -v
# 58 passed in 0.43s
```

---

## Сравнение: качество и расход токенов

### Расход токенов

| Метрика | Без компрессии | Со сжатием | Выигрыш |
|---|---|---|---|
| Запрос №10 | ~2,000 | ~2,000 | 0% |
| Запрос №20 | ~4,500 | ~800 | **~82%** |
| Запрос №30 | ~6,800 | ~900 | **~87%** |
| Запрос №50 | ~11,000 | ~950 | **~91%** |

*Оценочные данные для сообщений длиной ~100 токенов каждое.*

### Качество ответов

- **Без сжатия:** модель видит полный контекст → идеальная точность, но рост стоимости линейный.
- **Со сжатием:** модель видит краткое изложение → факты сохранены, нюансы могут потеряться. Для аналитических или длинных диалогов рекомендуется `keep_last_n=8–12`.

### Рекомендации

| Сценарий | summarize_every | keep_last_n |
|---|---|---|
| Лёгкий чат / FAQ | 10 | 6 |
| Технические детали | 10 | 10 |
| Длинное эссе / работа | 20 | 12 |

---

## Ключевые концепции

### Почему сжатие экономит токены?

Каждый API-запрос передаёт **весь** массив `messages[]` с начала диалога. При 30 обменах это 60+ сообщений — тысячи токенов prompt'а. Компрессия заменяет старые сообщения одним коротким summary (~200–400 токенов), сохраняя суть.

### Трейдофф

```
Полная история:  ТОЧНОСТЬ ↑↑  |  СТОИМОСТЬ ↑↑↑  |  ДЛИНА ДИАЛОГА ограничена
С компрессией:   ТОЧНОСТЬ ↑   |  СТОИМОСТЬ ↑    |  ДЛИНА ДИАЛОГА неограничена
```

### Когда не использовать

- При коротких диалогах (< 10 сообщений) — компрессия не запустится.
- Когда каждая деталь критична (например, работа с кодом построчно) — лучше `keep_last_n=20`.
