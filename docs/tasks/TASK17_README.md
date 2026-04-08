# TASK17 — Первый инструмент MCP (День 17)

## Что создано / обновлено

| Файл | Роль |
|------|------|
| `mcp_server.py` | MCP-сервер: расширен 3 новыми инструментами (save_memory, delete_memory_key, get_conversation_summary) |
| `mcp_agent.py` | MCPAgent: диалоговый агент с function calling для MCP-инструментов |
| `demo_mcp_agent.py` | Демо: агент автоматически вызывает MCP-инструменты для работы с историей/памятью |
| `news_agent/mcp_bridge.py` | MCPBridge: обновлён реестр MCP-инструментов |
| `test_mcp_server_day17.py` | Интеграционный тест: все 9 инструментов |

---

## Как запустить

```bash
# 1. Тест MCP-сервера (без API-ключа)
python3 test_mcp_server_day17.py

# 2. Демо агента с MCP-инструментами (нужен CLOUD_API_KEY в .env)
python3 demo_mcp_agent.py
```

---

## MCP-инструменты (9 шт.)

### Из Дня 16 (чтение):
| Инструмент | Описание |
|------------|----------|
| `list_logs` | Список файлов логов разговоров |
| `read_log` | Прочитать конкретный лог |
| `search_logs` | Поиск текста по всем логам |
| `list_memory` | Список файлов памяти |
| `read_memory` | Прочитать файл памяти |
| `get_usage_stats` | Статистика токенов и стоимости |

### Новые в Дне 17 (запись + аналитика):
| Инструмент | Описание |
|------------|----------|
| `save_memory` | Записать данные в файл памяти (создание/обновление) |
| `delete_memory_key` | Удалить ключ из файла памяти |
| `get_conversation_summary` | Краткая сводка разговора (сообщения, период, темы) |

---

## Архитектура

```
┌──────────────────────────────────────────────────────┐
│                  demo_mcp_agent.py                   │
│                                                      │
│  MCPAgent                                            │
│    ├── OpenAI API  (function calling)                │
│    │     LLM решает какой инструмент вызвать         │
│    │     → tool_calls: [{name, arguments}]           │
│    │                                                 │
│    └── MCPBridge (news_agent/mcp_bridge.py)          │
│          ├── subprocess.Popen(mcp_server.py)         │
│          ├── MCP handshake (initialize)              │
│          └── tools/call → JSON-RPC → результат       │
└────────────────────────┬─────────────────────────────┘
                    stdin│stdout
                         │   JSON-RPC 2.0
┌────────────────────────▼─────────────────────────────┐
│                   mcp_server.py                      │
│                                                      │
│  handle_request()                                    │
│    ├── "initialize"     → capabilities               │
│    ├── "tools/list"     → 9 инструментов             │
│    └── "tools/call"     → dispatch → handler         │
│          ├── tool_list_logs()                        │
│          ├── tool_read_log()                         │
│          ├── tool_search_logs()                      │
│          ├── tool_list_memory()                      │
│          ├── tool_read_memory()                      │
│          ├── tool_get_usage_stats()                  │
│          ├── tool_save_memory()          ← День 17   │
│          ├── tool_delete_memory_key()    ← День 17   │
│          └── tool_get_conversation_summary() ← 17   │
└──────────────────────────────────────────────────────┘
```

---

## Как работает MCPAgent

1. Пользователь отправляет сообщение
2. MCPAgent передаёт его в LLM через OpenAI API с `tools=MCP_TOOL_DEFINITIONS`
3. LLM возвращает `tool_calls` — какие MCP-инструменты вызвать и с какими аргументами
4. MCPAgent вызывает инструменты через MCPBridge → MCP-сервер (JSON-RPC)
5. Результаты возвращаются LLM как `role: tool` сообщения
6. LLM формирует финальный ответ пользователю на основе данных из инструментов

```
User: "Покажи статистику использования"
  → LLM: tool_call(get_usage_stats, {})
    → MCP Server: tools/call → JSON-RPC → handler
      → Ответ: "Разговоров: 12, Токенов: 50,000, Стоимость: $0.15"
  → LLM: "По данным системы, за все время было 12 разговоров..."
```

---

## Сценарий демонстрации

| # | Задача | Ожидаемые MCP-вызовы |
|---|--------|---------------------|
| 1 | Список логов | `list_logs` |
| 2 | Файлы памяти | `list_memory` + `read_memory` |
| 3 | Запомнить предпочтение | `save_memory` |
| 4 | Статистика | `get_usage_stats` |
| 5 | Поиск по истории | `search_logs` |
