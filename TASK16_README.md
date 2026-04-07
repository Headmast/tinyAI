# TASK16 — Локальный MCP-клиент для управления логами и памятью

## Что создано

| Файл | Роль |
|------|------|
| `mcp_server.py` | MCP-сервер: предоставляет инструменты для работы с логами и памятью |
| `mcp_client.py` | MCP-клиент: подключается к серверу, получает список инструментов, вызывает их |

---

## Как запустить

```bash
# Запускается одной командой — клиент сам поднимает сервер
python mcp_client.py
```

Ожидаемый вывод:
```
████ TinyAI MCP Client — День 16 ████

Шаг 1: Запускаем MCP-сервер
  Сервер запущен (pid=XXXXX)

Шаг 2: Отправляем initialize
  Сервер: tinyai-logs-server v1.0.0
  Протокол: 2024-11-05
  Capabilities: ['tools']

Шаг 3: Отправляем notifications/initialized
  Рукопожатие завершено!

Шаг 4: Запрашиваем tools/list
ДОСТУПНЫЕ ИНСТРУМЕНТЫ MCP (6 шт.):
  1. list_logs
  2. read_log
  3. search_logs
  4. list_memory
  5. read_memory
  6. get_usage_stats
```

---

## Как это работает — полная схема

```
┌─────────────────────────────────────────────────────────────┐
│                       mcp_client.py                         │
│                                                             │
│  MCPClient                                                  │
│    └── MCPConnection                                        │
│          ├── subprocess.Popen(mcp_server.py)  ← запускает  │
│          ├── .stdin.write(json)               ← пишет      │
│          └── .stdout.readline()               ← читает     │
└──────────────────────┬──────────────────────────────────────┘
                  stdin│stdout
                       │   (JSON-RPC 2.0, одна строка = одно сообщение)
┌──────────────────────▼──────────────────────────────────────┐
│                       mcp_server.py                         │
│                                                             │
│  main() читает stdin построчно                              │
│    └── handle_request()                                     │
│          ├── "initialize"      → serverInfo + capabilities  │
│          ├── "tools/list"      → список инструментов        │
│          └── "tools/call"      → вызывает нужный handler    │
│                ├── list_logs()                              │
│                ├── read_log()                               │
│                ├── search_logs()                            │
│                ├── list_memory()                            │
│                ├── read_memory()                            │
│                └── get_usage_stats()                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Протокол MCP под капотом

MCP (Model Context Protocol) — открытый стандарт от Anthropic.  
Транспорт: **stdio** (stdin/stdout) или HTTP+SSE.  
Формат: **JSON-RPC 2.0** — каждое сообщение на отдельной строке.

### Шаг 1 — initialize (рукопожатие)

Клиент сообщает серверу версию протокола и свои capabilities.  
Сервер отвечает своими capabilities.

```
Клиент → Сервер:
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": { "name": "tinyai-mcp-client", "version": "1.0.0" }
  }
}

Сервер → Клиент:
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": { "tools": {} },
    "serverInfo": { "name": "tinyai-logs-server", "version": "1.0.0" }
  }
}
```

### Шаг 2 — notifications/initialized (финал рукопожатия)

Клиент подтверждает готовность. Это уведомление — `id` нет, ответа нет.

```
Клиент → Сервер:
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

### Шаг 3 — tools/list (получение инструментов)

```
Клиент → Сервер:
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list"
}

Сервер → Клиент:
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "list_logs",
        "description": "Возвращает список всех файлов логов...",
        "inputSchema": {
          "type": "object",
          "properties": {
            "filter": { "type": "string", "description": "..." }
          },
          "required": []
        }
      },
      ...
    ]
  }
}
```

### Шаг 4 — tools/call (вызов инструмента)

```
Клиент → Сервер:
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "search_logs",
    "arguments": { "query": "нейрон", "max_results": 3 }
  }
}

Сервер → Клиент:
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      { "type": "text", "text": "Найдено 2 совпадения..." }
    ]
  }
}
```

---

## Доступные инструменты сервера

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `list_logs` | `filter?` (string) | Список всех файлов логов с размером и датой |
| `read_log` | `filename*`, `last_n?` | Содержимое конкретного лога |
| `search_logs` | `query*`, `max_results?` | Поиск текста по всем логам |
| `list_memory` | — | Список файлов памяти агента |
| `read_memory` | `filename*` | Содержимое файла памяти |
| `get_usage_stats` | — | Токены, стоимость, число разговоров |

`*` — обязательный параметр, `?` — необязательный

---

## Вызов отдельного инструмента из кода

```python
from mcp_client import MCPClient

with MCPClient() as client:
    # Список логов
    print(client.call_tool("list_logs"))

    # Прочитать последние 3 сообщения из лога
    print(client.call_tool("read_log", {
        "filename": "conversation_20260317_192513.json",
        "last_n": 3
    }))

    # Поиск по логам
    print(client.call_tool("search_logs", {"query": "GPT"}))

    # Статистика
    print(client.call_tool("get_usage_stats"))
```

---

## Почему без официального SDK

MCP SDK (`pip install mcp`) требует Python ≥ 3.10, в проекте используется Python 3.9.  
Реализация протокола вручную позволяет:
- работать на любой версии Python
- видеть точно, что происходит внутри MCP
- нет зависимости от сторонних пакетов (только stdlib)

Протокол MCP полностью открытый, его спецификация:
https://spec.modelcontextprotocol.io/

---

## Структура файлов

```
tinyAI/
├── mcp_server.py       ← MCP-сервер (инструменты для логов/памяти)
├── mcp_client.py       ← MCP-клиент (подключение + список инструментов)
├── TASK16_README.md    ← эта инструкция
├── logs/               ← файлы логов разговоров
└── memory_data/        ← файлы долгосрочной памяти
```
