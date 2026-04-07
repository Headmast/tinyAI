"""
Определения инструментов (tools) для function calling API.
Содержит схемы OpenAI tools + диспетчер вызовов.
"""

import json
from typing import Any, Callable, Dict, List, Optional

from news_agent.mcp_bridge import MCPBridge, MCP_TOOLS


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "save_post",
            "description": (
                "Сохраняет готовый новостной пост на диск. "
                "Возвращает post_id сохранённого поста."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Заголовок поста",
                    },
                    "content": {
                        "type": "string",
                        "description": "Полный текст поста",
                    },
                    "post_type": {
                        "type": "string",
                        "enum": ["breaking", "analysis", "digest", "social", "press"],
                        "description": "Тип поста",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Теги для поста",
                    },
                    "meta_description": {
                        "type": "string",
                        "description": "Краткое описание для SEO (опционально)",
                    },
                },
                "required": ["title", "content", "post_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_post_history",
            "description": (
                "Загружает последние N постов из хранилища. "
                "Используй для проверки на дублирование тем."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Количество постов для загрузки (по умолчанию 5)",
                        "default": 5,
                    },
                    "post_type": {
                        "type": "string",
                        "enum": ["breaking", "analysis", "digest", "social", "press"],
                        "description": "Фильтр по типу поста (опционально)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_topic",
            "description": (
                "Анализирует тему: определяет ключевые слова, тематику, "
                "сложность и рекомендуемый формат поста."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Тема для анализа",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "format_post",
            "description": (
                "Конвертирует текст поста в указанный формат "
                "(markdown, html, telegram, json, plain)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Текст поста для конвертации",
                    },
                    "target_format": {
                        "type": "string",
                        "enum": ["markdown", "html", "telegram", "json", "plain"],
                        "description": "Целевой формат",
                    },
                    "title": {
                        "type": "string",
                        "description": "Заголовок поста (опционально)",
                    },
                },
                "required": ["content", "target_format"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_post_by_id",
            "description": "Загружает конкретный пост по его ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "Уникальный идентификатор поста",
                    },
                },
                "required": ["post_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_duplicate",
            "description": (
                "Проверяет, есть ли в хранилище пост с похожей темой. "
                "Возвращает похожие посты если они есть."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Тема для проверки на дублирование",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Порог схожести от 0 до 1 (по умолчанию 0.7)",
                        "default": 0.7,
                    },
                },
                "required": ["topic"],
            },
        },
    },
    # ── MCP-инструменты (проксируются через mcp_bridge → mcp_server) ──────────
    {
        "type": "function",
        "function": {
            "name": "list_logs",
            "description": (
                "[MCP] Возвращает список всех файлов логов разговоров с метаданными. "
                "Используй чтобы узнать, какие логи чатов сохранены в системе."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Необязательный фильтр по имени файла (подстрока)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_log",
            "description": (
                "[MCP] Читает содержимое конкретного файла лога разговора. "
                "Возвращает сообщения с временными метками и статистикой токенов."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Имя файла лога (например conversation_20260317_192513.json)",
                    },
                    "last_n": {
                        "type": "integer",
                        "description": "Вернуть только последние N сообщений",
                    },
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": (
                "[MCP] Ищет заданный текст во всех логах разговоров. "
                "Используй для поиска предыдущих разговоров по теме."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Текст для поиска (без учёта регистра)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Максимальное число результатов (по умолчанию 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_memory",
            "description": (
                "[MCP] Возвращает список файлов долгосрочной памяти агента "
                "с кратким содержимым каждого."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": (
                "[MCP] Читает конкретный файл долгосрочной памяти агента. "
                "Используй для просмотра сохранённого профиля пользователя."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Имя файла памяти (например long_term.json)",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_usage_stats",
            "description": (
                "[MCP] Возвращает накопленную статистику использования: "
                "токены, стоимость, число разговоров."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


class ToolDispatcher:
    """
    Диспетчер вызовов инструментов.
    Связывает имена функций с их реализациями.
    """

    def __init__(self, storage=None, verbose: bool = False):
        self._storage = storage
        self._mcp = MCPBridge(verbose=verbose)
        self._handlers: Dict[str, Callable] = {
            "save_post": self._save_post,
            "load_post_history": self._load_post_history,
            "analyze_topic": self._analyze_topic,
            "format_post": self._format_post,
            "get_post_by_id": self._get_post_by_id,
            "check_duplicate": self._check_duplicate,
        }

    def close(self) -> None:
        """Закрывает MCP-соединение. Вызывать при завершении работы агента."""
        self._mcp.close()

    def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Вызывает нужный инструмент и возвращает строковый результат."""
        # MCP-инструменты проксируются напрямую в MCP-сервер
        if tool_name in MCP_TOOLS:
            return self._mcp.call_tool(tool_name, arguments)

        handler = self._handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Инструмент '{tool_name}' не найден"}, ensure_ascii=False)
        try:
            result = handler(**arguments)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _save_post(
        self,
        title: str,
        content: str,
        post_type: str,
        tags: Optional[List[str]] = None,
        meta_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._storage:
            return {"error": "Storage не инициализирован"}
        post_id = self._storage.save({
            "title": title,
            "content": content,
            "post_type": post_type,
            "tags": tags or [],
            "meta_description": meta_description or "",
        })
        return {"success": True, "post_id": post_id, "message": f"Пост сохранён с ID: {post_id}"}

    def _load_post_history(self, n: int = 5, post_type: Optional[str] = None) -> Dict[str, Any]:
        if not self._storage:
            return {"error": "Storage не инициализирован"}
        posts = self._storage.list_posts(n=n, post_type=post_type)
        return {
            "count": len(posts),
            "posts": [
                {
                    "id": p["id"],
                    "title": p["title"],
                    "post_type": p["post_type"],
                    "created_at": p["created_at"],
                    "tags": p.get("tags", []),
                }
                for p in posts
            ],
        }

    def _analyze_topic(self, topic: str) -> Dict[str, Any]:
        words = topic.lower().split()
        word_count = len(words)
        has_numbers = any(w.replace(",", "").replace(".", "").isdigit() for w in words)

        if word_count <= 5:
            suggested_type = "social"
        elif word_count <= 10 and has_numbers:
            suggested_type = "breaking"
        elif word_count > 15:
            suggested_type = "analysis"
        else:
            suggested_type = "breaking"

        keywords = [w for w in words if len(w) > 4][:5]

        return {
            "topic": topic,
            "word_count": word_count,
            "suggested_type": suggested_type,
            "keywords": keywords,
            "complexity": "high" if word_count > 12 else "medium" if word_count > 6 else "low",
            "has_numbers": has_numbers,
        }

    def _format_post(
        self, content: str, target_format: str, title: Optional[str] = None
    ) -> str:
        from news_agent.formatter import OutputFormatter
        formatter = OutputFormatter()
        return formatter.convert(content=content, fmt=target_format, title=title)

    def _get_post_by_id(self, post_id: str) -> Dict[str, Any]:
        if not self._storage:
            return {"error": "Storage не инициализирован"}
        post = self._storage.load(post_id)
        if not post:
            return {"error": f"Пост с ID '{post_id}' не найден"}
        return post

    def _check_duplicate(self, topic: str, threshold: float = 0.7) -> Dict[str, Any]:
        if not self._storage:
            return {"duplicates": [], "is_duplicate": False}
        duplicates = self._storage.find_by_keywords(topic, min_overlap=threshold)
        return {
            "is_duplicate": len(duplicates) > 0,
            "duplicates": [
                {"id": p["id"], "title": p["title"], "overlap_score": p.get("overlap_score", 0)}
                for p in duplicates
            ],
        }
