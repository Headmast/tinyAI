from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error, parse, request


def normalize_service_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


@dataclass(frozen=True)
class LocalLLMClientConfig:
    base_url: str = "http://127.0.0.1:8000"
    timeout: float = 30.0
    default_model: str = "qwen3:8b"


class LocalLLMHttpClient:
    def __init__(self, config: Optional[LocalLLMClientConfig] = None) -> None:
        self.config = config or LocalLLMClientConfig()
        self.base_url = normalize_service_base_url(self.config.base_url)

    def health(self) -> Dict[str, Any]:
        return self._request_json("GET", "/health")

    def list_models(self) -> Dict[str, Any]:
        return self._request_json("GET", "/v1/models")

    def chat(
        self,
        prompt: str,
        *,
        system: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_completion_tokens: int = 1024,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "history": history or [],
            "model": model or self.config.default_model,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        if system:
            payload["system"] = system
        return self._request_json("POST", "/chat", payload)

    def chat_completions(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_completion_tokens: int = 1024,
    ) -> Dict[str, Any]:
        payload = {
            "model": model or self.config.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        return self._request_json("POST", "/v1/chat/completions", payload)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = self.base_url + path
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(url, data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"error": {"message": raw}}
            message = parsed.get("error", {}).get("message", raw)
            raise RuntimeError(f"HTTP {exc.code}: {message}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Connection failed: {exc.reason}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP client for TinyAI local LLM service")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--model", default="qwen3:8b")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health")
    subparsers.add_parser("models")

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("prompt")
    chat_parser.add_argument("--system", default="")
    chat_parser.add_argument("--temperature", type=float, default=0.3)
    chat_parser.add_argument("--max-completion-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = LocalLLMHttpClient(
        LocalLLMClientConfig(
            base_url=args.base_url,
            timeout=args.timeout,
            default_model=args.model,
        )
    )

    if args.command == "health":
        print(json.dumps(client.health(), ensure_ascii=False, indent=2))
        return
    if args.command == "models":
        print(json.dumps(client.list_models(), ensure_ascii=False, indent=2))
        return

    result = client.chat(
        args.prompt,
        system=args.system,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()