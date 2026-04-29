"""
Support Assistant — AI-ассистент для поддержки пользователей.

Компоненты:
  - RAG-поиск по FAQ (support_data/faq.md) из индекса rag_data_support/
  - MCP Support Server — контекст пользователя и тикетов
  - LLM — генерация ответа с учётом FAQ и пользовательского контекста

Использование:
    from support_assistant import SupportAssistant

    assistant = SupportAssistant()
    answer = assistant.answer("Почему не работает авторизация?", user_id="user_1")
    print(answer)
    assistant.close()
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from core.config import get_config, get_llm_client
from mcp_stdio_client import MCPStdioClient
from rag import SearchResult
from rag.embedder import OpenAIEmbedder
from rag.search import search as rag_search

PROJECT_ROOT = Path(__file__).parent
DEFAULT_SUPPORT_INDEX = PROJECT_ROOT / "rag_data_support"
MCP_SUPPORT_SERVER = PROJECT_ROOT / "mcp_support_server.py"

SYSTEM_PROMPT = (
    "Ты — AI-ассистент службы поддержки платформы TinyAI. "
    "Твоя задача — помогать пользователям решать проблемы, отвечать на вопросы о продукте. "
    "Используй предоставленную документацию (FAQ) для точных ответов. "
    "Если предоставлен контекст пользователя или тикета — учитывай его при ответе "
    "(тариф пользователя, историю обращений, конкретную проблему из тикета). "
    "Будь вежливым, конкретным и полезным. "
    "Если информации недостаточно для ответа — честно скажи об этом и предложи обратиться в поддержку. "
    "Отвечай на русском языке."
)


class SupportAssistant:
    """
    Ассистент поддержки: RAG по FAQ + MCP (пользователи/тикеты) + LLM.

    Используется командой /support в chat_cli.py.
    """

    def __init__(
        self,
        index_dir: Path = DEFAULT_SUPPORT_INDEX,
        model: str | None = None,
        verbose: bool = False,
    ):
        self.index_dir = index_dir
        config = get_config()
        # Используем быструю модель для поддержки (Flash — быстрее и дешевле)
        self.model = model or config.default_fast_model
        self.verbose = verbose
        self._mcp_client: Optional[MCPStdioClient] = None
        # Кэшируем embedder — избегаем пересоздания при каждом поиске (LRU query cache)
        self._embedder = OpenAIEmbedder()
        # Кэшируем LLM-клиент (OpenAI для gpt-* моделей, Cloud.ru для остальных)
        provider = "openai" if self.model.startswith("gpt-") else "cloud"
        self._llm_client = get_llm_client(provider)

    def _get_mcp_client(self) -> MCPStdioClient:
        """Lazy init MCP-клиента для support-сервера."""
        if self._mcp_client is None:
            self._mcp_client = MCPStdioClient(
                server_path=MCP_SUPPORT_SERVER,
                client_name="support-assistant",
            )
        return self._mcp_client

    def _search_faq(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Поиск по RAG-индексу FAQ. Использует кэшированный embedder."""
        if not (self.index_dir / "chunks.db").exists():
            return []
        return rag_search(
            query=query,
            top_k=top_k,
            strategy="structure",
            index_dir=self.index_dir,
            similarity_threshold=0.2,
            embedder=self._embedder,
        )

    def _get_user_context(self, user_id: str) -> str:
        """Получает профиль и тикеты пользователя через MCP."""
        try:
            client = self._get_mcp_client()
            user_info = client.call_tool("get_user", {"user_id": user_id})
            tickets_info = client.call_tool("get_user_tickets", {"user_id": user_id})
            return f"## Профиль пользователя\n{user_info}\n\n## Тикеты пользователя\n{tickets_info}"
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  MCP Support недоступен: {e}", file=sys.stderr)
            return "Контекст пользователя недоступен."

    def _get_ticket_context(self, ticket_id: str) -> str:
        """Получает данные тикета через MCP."""
        try:
            client = self._get_mcp_client()
            ticket_info = client.call_tool("get_ticket", {"ticket_id": ticket_id})
            return f"## Данные тикета\n{ticket_info}"
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  MCP Support недоступен: {e}", file=sys.stderr)
            return "Контекст тикета недоступен."

    def answer(
        self,
        question: str,
        user_id: str | None = None,
        ticket_id: str | None = None,
    ) -> str:
        """
        Отвечает на вопрос пользователя.

        1. Ищет релевантные FAQ-фрагменты в RAG
        2. Получает контекст пользователя/тикета через MCP (если указан)
        3. Формирует промпт и вызывает LLM
        4. Возвращает ответ с источниками
        """
        # RAG search
        results = self._search_faq(question)

        # Build FAQ context
        faq_parts = []
        sources = []
        for r in results:
            section = r.chunk.metadata.get("section", "")
            sources.append({"section": section, "score": r.score})
            faq_parts.append(
                f"--- FAQ: {section} (score: {r.score:.3f}) ---\n{r.chunk.text}"
            )
        faq_context = "\n\n".join(faq_parts) if faq_parts else "FAQ не найдено по данному запросу."

        # User/ticket context via MCP
        extra_context_parts = []
        if user_id:
            extra_context_parts.append(self._get_user_context(user_id))
        if ticket_id:
            extra_context_parts.append(self._get_ticket_context(ticket_id))
        extra_context = "\n\n".join(extra_context_parts)

        # Build user prompt
        user_prompt_parts = [f"## FAQ (документация)\n{faq_context}"]
        if extra_context:
            user_prompt_parts.append(f"## Контекст\n{extra_context}")
        user_prompt_parts.append(f"## Вопрос пользователя\n{question}")
        user_prompt = "\n\n".join(user_prompt_parts)

        if self.verbose:
            print(f"  📚 Найдено FAQ-фрагментов: {len(results)}")
            for s in sources:
                print(f"     {s['section']} (score: {s['score']:.3f})")
            if user_id:
                print(f"  👤 Контекст пользователя: {user_id}")
            if ticket_id:
                print(f"  🎫 Контекст тикета: {ticket_id}")

        # LLM call (используем кэшированный клиент)
        try:
            is_reasoning = "nano" in self.model or self.model.startswith("o")

            # Reasoning-модели (nano, o-series) не поддерживают system role —
            # встраиваем системный промпт в user message
            if is_reasoning:
                messages = [
                    {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}"},
                ]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]

            kwargs = {
                "model": self.model,
                "messages": messages,
                # Reasoning-модели (nano, o-series) тратят токены на "думание",
                # поэтому нужен больший лимит (reasoning + output)
                "max_completion_tokens": 4096 if is_reasoning else 1024,
            }
            # GPT-5 Nano не поддерживает temperature ≠ 1
            if not is_reasoning:
                kwargs["temperature"] = 0.3
            response = self._llm_client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            answer = msg.content or ""

            # Reasoning-модели (GPT-5 Nano, o-series) могут вернуть ответ в reasoning
            if not answer:
                answer = getattr(msg, "reasoning_content", "") or ""
            if not answer:
                answer = getattr(msg, "reasoning", "") or ""
        except Exception as e:
            answer = f"Ошибка LLM: {e}"

        # Format output
        output_parts = [answer]
        if sources:
            output_parts.append("\n📚 Источники FAQ:")
            for i, s in enumerate(sources, 1):
                output_parts.append(
                    f"  [{i}] {s['section']} (score: {s['score']:.3f})"
                )

        return "\n".join(output_parts)

    def close(self):
        """Закрывает MCP-соединение."""
        if self._mcp_client:
            self._mcp_client.close()
            self._mcp_client = None
