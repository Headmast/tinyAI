# План: Файловый ассистент с MCP-инструментами

## Суть
Создать AI-ассистента, который автономно работает с файлами проекта через выделенный MCP-сервер.
Ассистент читает, ищет, анализирует и модифицирует файлы на основе высокоуровневых целей.
Реализует 3 сценария: поиск использований символа, генерация CHANGELOG, проверка серверов на инварианты безопасности.

## Архитектура
- **mcp_files_server.py** — MCP-сервер с 6 файловыми инструментами (stdio transport, JSON-RPC 2.0)
- **file_assistant.py** — Агентный LLM-цикл (паттерн MCPAgent), использует файловые инструменты через MCPStdioClient
- **demos/demo_file_assistant.py** — Демо с 3 сценариями
- **tests/test_mcp_files_server.py** — Тесты серверных инструментов через subprocess

## Фаза 1: MCP-сервер для файлов (`mcp_files_server.py`)

По паттерну `mcp_server.py` / `mcp_git_server.py`:
- Список TOOLS, словарь TOOL_HANDLERS, диспетчер handle_request(), основной цикл stdin
- BASE_DIR = Path(__file__).parent (корень проекта)
- Защита от path traversal: resolve() + startswith()

### 6 инструментов

| Инструмент | Назначение |
|---|---|
| `list_project_files(path?, pattern?)` | Список файлов/каталогов. Пропускает `__pycache__`, `.git`, `node_modules`, `*.pyc` |
| `read_file(path, max_lines?)` | Чтение файла с ограничением строк. Защита от path traversal |
| `search_in_files(query, path?, file_pattern?, max_results?)` | Grep-like поиск по файлам. Регистро-независимый |
| `get_file_info(path)` | Метаданные: размер, строки, дата изменения |
| `write_file(path, content)` | Создать/перезаписать файл. Создаёт родительские каталоги |
| `apply_diff(path, old_text, new_text)` | Замена текста в файле (str_replace). Проверяет уникальность |

### Безопасность
- Все пути проверяются через `resolve()` + `startswith(BASE_DIR)`
- `write_file` и `apply_diff` запрещены для `.git/`, `.env`, бинарных файлов

## Фаза 2: Агент-ассистент (`file_assistant.py`)

Класс `FileAssistant` по паттерну `mcp_agent.py`:
- Использует `MCPStdioClient(server_path="mcp_files_server.py")`
- Агентный цикл `run(task)` до 15 итераций
- LLM выбирает инструменты → вызов через MCP → накопление результатов → финальный ответ

### Системный промпт
Инструктирует агента:
- Планировать перед действием (сначала `list_project_files`, потом `read_file`)
- Использовать `search_in_files` для поиска паттернов по кодовой базе
- При генерации файлов — `write_file`
- При модификации — `apply_diff`
- Отчитываться: что сделано, какие файлы затронуты, итог

## Фаза 3: Демо (`demos/demo_file_assistant.py`)

3 сценария (последовательно, каждый независим):

### Сценарий 1: Найти все использования MCPRegistry
- Задача: "Найди все файлы, где используется класс MCPRegistry. Покажи, как именно он используется."
- Ожидание: агент ищет "MCPRegistry" → читает найденные файлы → отчёт

### Сценарий 2: Сгенерировать CHANGELOG.md
- Задача: "Проанализируй структуру проекта и сгенерируй файл CHANGELOG.md"
- Ожидание: агент перечисляет файлы → читает ключевые → генерирует и записывает CHANGELOG.md

### Сценарий 3: Проверка MCP-серверов на path traversal
- Задача: "Проверь все MCP-серверы проекта на наличие защиты от path traversal"
- Ожидание: агент ищет mcp_*_server.py → читает каждый → анализирует → отчёт

## Фаза 4: Тесты (`tests/test_mcp_files_server.py`)

Subprocess-паттерн: `_send_requests()` хелпер + тесты:
- `test_initialize`, `test_tools_list`
- `test_list_project_files`, `test_read_file`
- `test_read_file_path_traversal` (отклоняет `../../etc/passwd`)
- `test_search_in_files`, `test_write_file`, `test_apply_diff`

## Верификация
1. `pytest tests/test_mcp_files_server.py -v` — все тесты проходят
2. `python demos/demo_file_assistant.py` — 3 сценария работают с 2+ файлами каждый
3. CHANGELOG.md создаётся с реальным содержимым
4. Сценарий 3 корректно определяет наличие/отсутствие защиты path traversal

## Решения
- `MCPStdioClient` (не MCPBridge) — проще, паттерн из `dev_assistant.py`
- 6 инструментов — минимум, но покрывает все сценарии
- `write_file` для новых файлов, `apply_diff` для изменений
- Файл плана `PLAN_FILE_ASSISTANT.md` НЕ удаляется при реализации
