"""
Центральная конфигурация TinyAI.

Единая точка загрузки переменных окружения и создания LLM-клиентов.
Устраняет дублирование load_dotenv() и os.getenv() по всему проекту.

Использование:
    from core.config import get_config, get_llm_client

    config = get_config()
    client = get_llm_client()           # Cloud.ru клиент
    client = get_llm_client("openai")   # OpenAI клиент (для embeddings)
    client = get_llm_client("ollama")   # Локальный Ollama клиент
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Загружаем .env один раз при импорте модуля
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# --- Константы ---

CLOUD_BASE_URL = "https://foundation-models.api.cloud.ru/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

DEFAULT_MODEL = "zai-org/GLM-4.7"
DEFAULT_FAST_MODEL = "zai-org/GLM-4.7-Flash"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_RETRIES = 3


@dataclass(frozen=True)
class AppConfig:
    """Конфигурация приложения из переменных окружения."""

    cloud_api_key: str = ""
    openai_api_key: str = ""
    base_url: str = CLOUD_BASE_URL
    default_model: str = DEFAULT_MODEL
    default_fast_model: str = DEFAULT_FAST_MODEL
    default_temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = DEFAULT_MAX_RETRIES

    # Пути данных
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    logs_path: Path = field(default_factory=lambda: _PROJECT_ROOT / "logs")
    sessions_path: Path = field(default_factory=lambda: _PROJECT_ROOT / "sessions")
    posts_path: Path = field(default_factory=lambda: _PROJECT_ROOT / "posts")
    memory_data_path: Path = field(default_factory=lambda: _PROJECT_ROOT / "memory_data")
    scheduler_db_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "scheduler_data" / "scheduler.db"
    )
    pipeline_db_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "pipeline_data" / "content.db"
    )
    rag_data_path: Path = field(default_factory=lambda: _PROJECT_ROOT / "rag_data")

    @property
    def has_cloud_key(self) -> bool:
        return bool(self.cloud_api_key)

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def primary_api_key(self) -> str:
        """Возвращает первый доступный API-ключ (Cloud.ru приоритет)."""
        return self.cloud_api_key or self.openai_api_key


_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Возвращает синглтон конфигурации, загруженный из переменных окружения."""
    global _config
    if _config is None:
        _config = AppConfig(
            cloud_api_key=os.getenv("CLOUD_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("BASE_URL", CLOUD_BASE_URL),
            default_model=os.getenv("DEFAULT_MODEL", DEFAULT_MODEL),
            default_fast_model=os.getenv("DEFAULT_FAST_MODEL", DEFAULT_FAST_MODEL),
        )
    return _config


def get_llm_client(provider: str = "cloud") -> "OpenAI":
    """Создаёт OpenAI-совместимый клиент для указанного провайдера.

    Args:
        provider: "cloud" для Cloud.ru (по умолчанию), "openai" для OpenAI.

    Returns:
        Экземпляр openai.OpenAI с настроенными base_url и api_key.

    Raises:
        ValueError: Если API-ключ для провайдера не найден.
    """
    from openai import OpenAI

    config = get_config()

    if provider == "ollama":
        ollama_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
        return OpenAI(
            api_key="ollama",
            base_url=ollama_url,
        )

    if provider == "openai":
        if not config.has_openai_key:
            raise ValueError(
                "OPENAI_API_KEY не найден. Задайте в .env для работы с OpenAI API."
            )
        return OpenAI(
            api_key=config.openai_api_key,
            base_url=OPENAI_BASE_URL,
        )

    # Cloud.ru (default)
    api_key = config.primary_api_key
    if not api_key:
        raise ValueError(
            "API-ключ не найден. Задайте CLOUD_API_KEY или OPENAI_API_KEY в .env"
        )
    return OpenAI(
        api_key=api_key,
        base_url=config.base_url,
    )
