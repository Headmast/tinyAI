"""
Developer Assistant — отвечает на вопросы о проекте используя RAG + MCP Git.

Компоненты:
  - RAG-поиск по документации (README + docs/) из индекса rag_data_dev/
  - MCP Git Server — контекст текущей ветки, статуса файлов
  - LLM — генерация финального ответа с источниками

Использование:
    from dev_assistant import DevAssistant

    assistant = DevAssistant()
    answer = assistant.answer_help("Как устроен RAG в проекте?")
    print(answer)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from core.config import get_config, get_llm_client
from rag import SearchResult
from rag.search import search as rag_search

PROJECT_ROOT = Path(__file__).parent
DEFAULT_DEV_INDEX = PROJECT_ROOT / "rag_data_dev"
MCP_GIT_SERVER = PROJECT_ROOT / "mcp_git_server.py"


class _GitMCPClient:
    """Минимальный MCP-клиент для git-сервера (одноразовый вызов)."""

    def __init__(self, server_path: Path = MCP_GIT_SERVER):
        self._server_path = server_path
        self._proc: Optional[subprocess.Popen] = None
        self._msg_id = 0

    def _ensure_started(self):
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            [sys.executable, str(self._server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Handshake
        self._send_recv("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "dev-assistant", "version": "1.0"},
        })
        # Notification
        self._proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        self._proc.stdin.flush()

    def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """Вызывает MCP-инструмент и возвращает текстовый результат."""
        self._ensure_started()
        resp = self._send_recv("tools/call", {"name": name, "arguments": arguments or {}})
        result = resp.get("result", {})
        content = result.get("content", [{}])
        return content[0].get("text", "") if content else ""

    def close(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._proc = None

    def _send_recv(self, method: str, params: dict | None = None) -> dict:
        self._msg_id += 1
        msg = {"jsonrpc": "2.0", "id": self._msg_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        if not raw:
            raise RuntimeError("Git MCP server closed unexpectedly")
        return json.loads(raw)


class DevAssistant:
    """
    Ассистент разработчика: RAG-поиск по документации + git-контекст + LLM.

    Используется командой /help в chat_cli.py.
    """

    def __init__(
        self,
        index_dir: Path = DEFAULT_DEV_INDEX,
        model: str | None = None,
        verbose: bool = False,
    ):
        self.index_dir = index_dir
        config = get_config()
        self.model = model or config.default_model
        self.verbose = verbose
        self._git_client: Optional[_GitMCPClient] = None

    def _get_git_client(self) -> _GitMCPClient:
        if self._git_client is None:
            self._git_client = _GitMCPClient()
        return self._git_client

    def _get_git_context(self) -> str:
        """Получает git-контекст через MCP."""
        try:
            client = self._get_git_client()
            branch = client.call_tool("git_current_branch")
            status = client.call_tool("git_status")
            return f"{branch}\n\n{status}"
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Git MCP недоступен: {e}", file=sys.stderr)
            return "Git-контекст недоступен."

    def _search_docs(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Поиск по RAG-индексу документации."""
        if not (self.index_dir / "chunks.db").exists():
            return []
        return rag_search(
            query=query,
            top_k=top_k,
            strategy="structure",
            index_dir=self.index_dir,
            similarity_threshold=0.2,
        )

    def answer_help(self, question: str) -> str:
        """
        Отвечает на вопрос о проекте.

        1. Ищет релевантные документы в RAG
        2. Получает git-контекст (ветка, статус)
        3. Формирует промпт и вызывает LLM
        4. Возвращает форматированный ответ с источниками
        """
        # RAG search
        results = self._search_docs(question)

        # Git context
        git_context = self._get_git_context()

        # Build context from RAG results
        doc_context_parts = []
        sources = []
        for r in results:
            src = r.chunk.metadata.get("source", "unknown")
            section = r.chunk.metadata.get("section", "")
            label = f"{src}"
            if section:
                label += f" / {section}"
            sources.append({"source": src, "section": section, "score": r.score})
            doc_context_parts.append(
                f"--- Источник: {label} (score: {r.score:.3f}) ---\n{r.chunk.text}"
            )

        doc_context = "\n\n".join(doc_context_parts) if doc_context_parts else "Документация не найдена."

        # System prompt
        system_prompt = (
            "Ты — ассистент разработчика проекта TinyAI. "
            "Отвечай на вопросы о проекте, используя предоставленную документацию и git-контекст. "
            "Если информация есть в документации — цитируй источники. "
            "Если информации недостаточно — честно скажи об этом. "
            "Отвечай на том же языке, на котором задан вопрос."
        )

        user_prompt = (
            f"## Git-контекст\n{git_context}\n\n"
            f"## Документация проекта\n{doc_context}\n\n"
            f"## Вопрос\n{question}"
        )

        if self.verbose:
            print(f"  📚 Найдено источников: {len(results)}")
            for s in sources:
                print(f"     {s['source']} (score: {s['score']:.3f})")
            print(f"  🌿 Git: {git_context.splitlines()[0] if git_context else 'n/a'}")

        # LLM call
        client = get_llm_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=2048,
                temperature=0.3,
            )
            answer = response.choices[0].message.content or ""

            # Fallback: extract from reasoning if content is empty (GLM quirk)
            if not answer and hasattr(response.choices[0].message, "reasoning"):
                answer = response.choices[0].message.reasoning or ""
        except Exception as e:
            answer = f"Ошибка LLM: {e}"

        # Format output
        output_parts = [answer]

        if sources:
            output_parts.append("\n📚 Источники:")
            for i, s in enumerate(sources, 1):
                section_str = f" / {s['section']}" if s['section'] else ""
                output_parts.append(
                    f"  [{i}] {s['source']}{section_str} (score: {s['score']:.3f})"
                )

        return "\n".join(output_parts)

    def close(self):
        """Закрывает MCP-соединение."""
        if self._git_client:
            self._git_client.close()
            self._git_client = None
