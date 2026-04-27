from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib import error, request
from urllib.parse import parse_qs, unquote, urlparse


DEFAULT_UPSTREAM_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
DEFAULT_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen3:8b")
_PROJECT_ROOT = Path(__file__).parent
_DASHBOARD_ROOT = _PROJECT_ROOT / "dashboard_ui"
_LOGS_DIR = _PROJECT_ROOT / "logs"
_TELEMETRY_FILE = _LOGS_DIR / "dashboard_telemetry.json"


def iso_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def safe_snippet(text: str, max_len: int = 150) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def parse_int_field(payload: Dict[str, Any], name: str, default: int) -> int:
    value = payload.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field '{name}' must be an integer.") from exc


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Грубая оценка токенов без внешних зависимостей."""
    total_chars = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            total_chars += sum(len(str(item)) for item in content)
        else:
            total_chars += len(str(content))
    return max(1, total_chars // 4)


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    upstream_base_url: str = DEFAULT_UPSTREAM_BASE_URL
    model: str = DEFAULT_MODEL
    request_timeout: float = 120.0
    max_requests_per_minute: int = 30
    max_context_tokens: int = 12_000
    max_completion_tokens: int = 2_048
    dashboard_history_limit: int = 200


@dataclass(frozen=True)
class ForwardResult:
    payload: Dict[str, Any]
    prompt_tokens: int
    completion_tokens: int


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: Deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self, now: Optional[float] = None) -> bool:
        now = now or time.time()
        with self._lock:
            while self._events and now - self._events[0] >= self.window_seconds:
                self._events.popleft()
            if len(self._events) >= self.limit:
                return False
            self._events.append(now)
            return True


class ServiceTelemetry:
    def __init__(self, persist_file: Path, history_limit: int = 200) -> None:
        self.persist_file = persist_file
        self.history_limit = history_limit
        self.started_at = time.time()
        self.active_requests = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rate_limit_hits = 0
        self.bad_request_errors = 0
        self.upstream_errors = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_limit)
        self._lock = threading.Lock()
        self.persist_file.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def begin_upstream(self) -> None:
        with self._lock:
            self.active_requests += 1

    def end_upstream(self) -> None:
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)

    def record_request(
        self,
        endpoint: str,
        client_ip: str,
        model: str,
        status_code: int,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        prompt_preview: str,
        response_preview: str,
        error_message: str = "",
        error_type: str = "",
    ) -> None:
        total_tokens = prompt_tokens + completion_tokens
        entry = {
            "timestamp": iso_timestamp(),
            "endpoint": endpoint,
            "client_ip": client_ip,
            "model": model,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
            "prompt_preview": safe_snippet(prompt_preview),
            "response_preview": safe_snippet(response_preview),
            "error": error_message,
            "error_type": error_type,
            "success": 200 <= status_code < 300,
        }

        with self._lock:
            self.total_requests += 1
            self.total_prompt_tokens += int(prompt_tokens)
            self.total_completion_tokens += int(completion_tokens)

            if 200 <= status_code < 300:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
                if status_code == 429:
                    self.rate_limit_hits += 1
                elif error_type == "upstream_error":
                    self.upstream_errors += 1
                else:
                    self.bad_request_errors += 1

            self._history.appendleft(entry)
            self._save_locked()

    def status_payload(self, config: ServiceConfig) -> Dict[str, Any]:
        with self._lock:
            uptime = int(max(0.0, time.time() - self.started_at))
            return {
                "status": "ok",
                "model": config.model,
                "upstream_base_url": config.upstream_base_url,
                "uptime_seconds": uptime,
                "active_requests": self.active_requests,
                "limits": {
                    "max_requests_per_minute": config.max_requests_per_minute,
                    "max_context_tokens": config.max_context_tokens,
                    "max_completion_tokens": config.max_completion_tokens,
                },
            }

    def metrics_payload(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "rate_limit_hits": self.rate_limit_hits,
                "bad_request_errors": self.bad_request_errors,
                "upstream_errors": self.upstream_errors,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "history_size": len(self._history),
            }

    def history_payload(self, limit: int) -> Dict[str, Any]:
        with self._lock:
            safe_limit = max(1, min(limit, self.history_limit))
            items = list(self._history)[:safe_limit]
            return {"items": items, "count": len(items)}

    def _load(self) -> None:
        if not self.persist_file.exists():
            return
        try:
            with open(self.persist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        counters = data.get("counters", {})
        self.total_requests = int(counters.get("total_requests", 0))
        self.successful_requests = int(counters.get("successful_requests", 0))
        self.failed_requests = int(counters.get("failed_requests", 0))
        self.rate_limit_hits = int(counters.get("rate_limit_hits", 0))
        self.bad_request_errors = int(counters.get("bad_request_errors", 0))
        self.upstream_errors = int(counters.get("upstream_errors", 0))
        self.total_prompt_tokens = int(counters.get("total_prompt_tokens", 0))
        self.total_completion_tokens = int(counters.get("total_completion_tokens", 0))

        history = data.get("history", [])
        if isinstance(history, list):
            for item in history[: self.history_limit]:
                if isinstance(item, dict):
                    self._history.append(item)

    def _save_locked(self) -> None:
        payload = {
            "updated_at": iso_timestamp(),
            "counters": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "rate_limit_hits": self.rate_limit_hits,
                "bad_request_errors": self.bad_request_errors,
                "upstream_errors": self.upstream_errors,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
            },
            "history": list(self._history),
        }
        try:
            with open(self.persist_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError:
            return


class LocalLLMService:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self._limiters: Dict[str, SlidingWindowRateLimiter] = {}
        self._limiters_lock = threading.Lock()
        self.telemetry = ServiceTelemetry(
            persist_file=_TELEMETRY_FILE,
            history_limit=config.dashboard_history_limit,
        )

    def serve_forever(self) -> None:
        server = ThreadingHTTPServer((self.config.host, self.config.port), self._build_handler())
        print(
            "Local LLM service listening on "
            f"http://{self.config.host}:{self.config.port} -> {self.config.upstream_base_url} "
            f"(model={self.config.model})"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping local LLM service")
        finally:
            server.server_close()

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        service = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "TinyAI-LocalLLM/0.1"

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/health":
                    self._send_json(200, service._health_payload())
                    return
                if path == "/v1/models":
                    self._send_json(200, service._models_payload())
                    return
                if path == "/dashboard":
                    self._send_dashboard_index()
                    return
                if path.startswith("/dashboard/static/"):
                    self._send_dashboard_static(path)
                    return
                if path == "/dashboard/api/status":
                    self._send_json(200, service.telemetry.status_payload(service.config))
                    return
                if path == "/dashboard/api/metrics":
                    self._send_json(200, service.telemetry.metrics_payload())
                    return
                if path == "/dashboard/api/history":
                    query = parse_qs(parsed.query)
                    limit = 50
                    if "limit" in query:
                        try:
                            limit = int(query["limit"][0])
                        except (TypeError, ValueError):
                            limit = 50
                    self._send_json(200, service.telemetry.history_payload(limit=limit))
                    return
                self._send_json(404, {"error": {"message": "Not found"}})

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                if path == "/v1/chat/completions":
                    self._handle_chat_completions()
                    return
                if path == "/chat":
                    self._handle_simple_chat()
                    return
                self._send_json(404, {"error": {"message": "Not found"}})

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _handle_chat_completions(self) -> None:
                started_at = time.monotonic()
                payload, body_error = self._read_json_body()
                model = service.config.model
                status_code = 500
                prompt_tokens = 0
                completion_tokens = 0
                prompt_preview = ""
                response_preview = ""
                error_message = ""
                error_type = ""
                if body_error is not None:
                    status_code = body_error[0]
                    error_message = body_error[1].get("error", {}).get("message", "")
                    error_type = "bad_request"
                    self._send_json(*body_error)
                else:
                    model = str(payload.get("model", service.config.model))
                    prompt_preview = service._extract_user_prompt(payload.get("messages", []))

                    limited, limit_error = service._check_limits(self.client_address[0], payload)
                    if not limited:
                        status_code = limit_error[0]
                        error_message = limit_error[1].get("error", {}).get("message", "")
                        error_type = "rate_limit" if status_code == 429 else "bad_request"
                        self._send_json(*limit_error)
                    else:
                        try:
                            service.telemetry.begin_upstream()
                            result = service._forward_chat(payload)
                        except ValueError as exc:
                            status_code = 400
                            error_message = str(exc)
                            error_type = "bad_request"
                            self._send_json(
                                400,
                                {"error": {"message": str(exc), "type": "invalid_request_error"}},
                            )
                        except RuntimeError as exc:
                            status_code = 502
                            error_message = str(exc)
                            error_type = "upstream_error"
                            self._send_json(
                                502,
                                {"error": {"message": str(exc), "type": "upstream_error"}},
                            )
                        else:
                            status_code = 200
                            prompt_tokens = result.prompt_tokens
                            completion_tokens = result.completion_tokens
                            response_preview = service._extract_content(result.payload)
                            self._send_json(200, result.payload)
                        finally:
                            service.telemetry.end_upstream()

                service.telemetry.record_request(
                    endpoint="/v1/chat/completions",
                    client_ip=self.client_address[0],
                    model=model,
                    status_code=status_code,
                    latency_ms=(time.monotonic() - started_at) * 1000.0,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    prompt_preview=prompt_preview,
                    response_preview=response_preview,
                    error_message=error_message,
                    error_type=error_type,
                )

            def _handle_simple_chat(self) -> None:
                started_at = time.monotonic()
                status_code = 500
                prompt_tokens = 0
                completion_tokens = 0
                prompt_preview = ""
                response_preview = ""
                error_message = ""
                error_type = ""
                model = service.config.model

                payload, body_error = self._read_json_body()
                if body_error is not None:
                    status_code = body_error[0]
                    error_message = body_error[1].get("error", {}).get("message", "")
                    error_type = "bad_request"
                    self._send_json(*body_error)
                else:
                    prompt = payload.get("prompt", "")
                    history = payload.get("history", [])
                    model = str(payload.get("model", service.config.model))
                    if isinstance(prompt, str):
                        prompt_preview = prompt
                    if not isinstance(prompt, str) or not prompt.strip():
                        status_code = 400
                        error_message = "Field 'prompt' must be a non-empty string."
                        error_type = "bad_request"
                        self._send_json(
                            400,
                            {"error": {"message": "Field 'prompt' must be a non-empty string."}},
                        )
                    elif not isinstance(history, list):
                        status_code = 400
                        error_message = "Field 'history' must be a list."
                        error_type = "bad_request"
                        self._send_json(
                            400,
                            {"error": {"message": "Field 'history' must be a list."}},
                        )
                    else:
                        messages: List[Dict[str, Any]] = []
                        system_prompt = payload.get("system")
                        if isinstance(system_prompt, str) and system_prompt.strip():
                            messages.append({"role": "system", "content": system_prompt})

                        for item in history:
                            if isinstance(item, dict) and "role" in item and "content" in item:
                                messages.append({"role": item["role"], "content": item["content"]})
                        messages.append({"role": "user", "content": prompt})

                        chat_payload = {
                            "model": model,
                            "messages": messages,
                            "max_completion_tokens": payload.get(
                                "max_completion_tokens", service.config.max_completion_tokens
                            ),
                            "temperature": payload.get("temperature", 0.3),
                        }

                        limited, limit_error = service._check_limits(self.client_address[0], chat_payload)
                        if not limited:
                            status_code = limit_error[0]
                            error_message = limit_error[1].get("error", {}).get("message", "")
                            error_type = "rate_limit" if status_code == 429 else "bad_request"
                            self._send_json(*limit_error)
                        else:
                            try:
                                service.telemetry.begin_upstream()
                                result = service._forward_chat(chat_payload)
                            except ValueError as exc:
                                status_code = 400
                                error_message = str(exc)
                                error_type = "bad_request"
                                self._send_json(
                                    400,
                                    {"error": {"message": str(exc), "type": "invalid_request_error"}},
                                )
                            except RuntimeError as exc:
                                status_code = 502
                                error_message = str(exc)
                                error_type = "upstream_error"
                                self._send_json(
                                    502,
                                    {"error": {"message": str(exc), "type": "upstream_error"}},
                                )
                            else:
                                status_code = 200
                                content = service._extract_content(result.payload)
                                response_preview = content
                                prompt_tokens = result.prompt_tokens
                                completion_tokens = result.completion_tokens
                                self._send_json(
                                    200,
                                    {
                                        "reply": content,
                                        "model": result.payload.get("model", service.config.model),
                                        "usage": result.payload.get("usage", {}),
                                    },
                                )
                            finally:
                                service.telemetry.end_upstream()

                service.telemetry.record_request(
                    endpoint="/chat",
                    client_ip=self.client_address[0],
                    model=model,
                    status_code=status_code,
                    latency_ms=(time.monotonic() - started_at) * 1000.0,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    prompt_preview=prompt_preview,
                    response_preview=response_preview,
                    error_message=error_message,
                    error_type=error_type,
                )

            def _send_dashboard_index(self) -> None:
                self._send_dashboard_file("index.html")

            def _send_dashboard_static(self, path: str) -> None:
                rel_path = path.replace("/dashboard/static/", "", 1)
                self._send_dashboard_file(rel_path)

            def _send_dashboard_file(self, relative_path: str) -> None:
                try:
                    body, content_type = service.read_dashboard_asset(relative_path)
                except (FileNotFoundError, ValueError):
                    self._send_json(404, {"error": {"message": "Not found"}})
                    return
                self._send_bytes(200, body, content_type)

            def _read_json_body(self) -> Tuple[Dict[str, Any], Optional[Tuple[int, Dict[str, Any]]]]:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_length) if content_length else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    return {}, (400, {"error": {"message": "Invalid JSON body."}})
                if not isinstance(payload, dict):
                    return {}, (400, {"error": {"message": "JSON body must be an object."}})
                return payload, None

            def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self._send_bytes(status, body, "application/json; charset=utf-8")

            def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def _health_payload(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": self.config.model,
            "upstream_base_url": self.config.upstream_base_url,
            "limits": {
                "max_requests_per_minute": self.config.max_requests_per_minute,
                "max_context_tokens": self.config.max_context_tokens,
                "max_completion_tokens": self.config.max_completion_tokens,
            },
        }

    def read_dashboard_asset(self, relative_path: str) -> Tuple[bytes, str]:
        rel = unquote(relative_path).lstrip("/")
        target = (_DASHBOARD_ROOT / rel).resolve()
        root = _DASHBOARD_ROOT.resolve()
        if not str(target).startswith(str(root)):
            raise ValueError("Invalid path")
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(str(target))
        with open(target, "rb") as f:
            body = f.read()
        guessed, _ = mimetypes.guess_type(str(target))
        content_type = guessed or "application/octet-stream"
        if target.suffix.lower() == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif target.suffix.lower() == ".css":
            content_type = "text/css; charset=utf-8"
        elif target.suffix.lower() == ".html":
            content_type = "text/html; charset=utf-8"
        return body, content_type

    def _models_payload(self) -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.config.model,
                    "object": "model",
                    "owned_by": "local-llm-service",
                }
            ],
        }

    @staticmethod
    def _extract_user_prompt(messages: Any) -> str:
        if not isinstance(messages, list):
            return ""
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(str(item) for item in content)
            return str(content)
        return ""

    def _check_limits(
        self,
        client_ip: str,
        payload: Dict[str, Any],
    ) -> Tuple[bool, Tuple[int, Dict[str, Any]]]:
        limiter = self._get_limiter(client_ip)
        if not limiter.allow():
            return False, (
                429,
                {
                    "error": {
                        "message": "Rate limit exceeded.",
                        "type": "rate_limit_error",
                    }
                },
            )

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return False, (
                400,
                {
                    "error": {
                        "message": "Field 'messages' must be a non-empty list.",
                        "type": "invalid_request_error",
                    }
                },
            )

        if payload.get("stream"):
            return False, (
                400,
                {
                    "error": {
                        "message": "Streaming is not supported by this service.",
                        "type": "unsupported_feature",
                    }
                },
            )

        estimated_prompt_tokens = estimate_tokens(messages)
        if estimated_prompt_tokens > self.config.max_context_tokens:
            return False, (
                400,
                {
                    "error": {
                        "message": (
                            "Prompt exceeds max context limit: "
                            f"{estimated_prompt_tokens} > {self.config.max_context_tokens}"
                        ),
                        "type": "context_length_exceeded",
                    }
                },
            )

        requested_completion_tokens = parse_int_field(
            payload,
            "max_completion_tokens",
            self.config.max_completion_tokens,
        )
        if requested_completion_tokens > self.config.max_completion_tokens:
            return False, (
                400,
                {
                    "error": {
                        "message": (
                            "Requested max_completion_tokens exceeds service limit: "
                            f"{requested_completion_tokens} > {self.config.max_completion_tokens}"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        return True, (200, {})

    def _get_limiter(self, client_ip: str) -> SlidingWindowRateLimiter:
        with self._limiters_lock:
            limiter = self._limiters.get(client_ip)
            if limiter is None:
                limiter = SlidingWindowRateLimiter(self.config.max_requests_per_minute)
                self._limiters[client_ip] = limiter
            return limiter

    def _forward_chat(self, payload: Dict[str, Any]) -> ForwardResult:
        messages = payload.get("messages", [])
        if not isinstance(messages, list) or not messages:
            raise ValueError("Field 'messages' must be a non-empty list.")

        upstream_payload = {
            "model": payload.get("model", self.config.model),
            "messages": messages,
            "temperature": payload.get("temperature", 0.3),
            "stream": False,
            "max_completion_tokens": parse_int_field(
                payload,
                "max_completion_tokens",
                self.config.max_completion_tokens,
            ),
        }

        upstream_url = self.config.upstream_base_url.rstrip("/") + "/chat/completions"
        request_body = json.dumps(upstream_payload).encode("utf-8")
        req = request.Request(
            upstream_url,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.config.request_timeout) as response:
                upstream_response = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Upstream returned HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Upstream connection failed: {exc.reason}") from exc

        prompt_tokens = estimate_tokens(messages)
        content = self._extract_content(upstream_response)
        completion_tokens = max(1, len(content) // 4) if content else 0
        usage = upstream_response.get("usage") or {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        upstream_response["usage"] = usage
        upstream_response.setdefault("model", upstream_payload["model"])
        upstream_response.setdefault("object", "chat.completion")
        return ForwardResult(
            payload=upstream_response,
            prompt_tokens=usage.get("prompt_tokens", prompt_tokens),
            completion_tokens=usage.get("completion_tokens", completion_tokens),
        )

    @staticmethod
    def _extract_content(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(item) for item in content)
        return ""


def parse_args() -> ServiceConfig:
    parser = argparse.ArgumentParser(description="TinyAI local LLM HTTP service")
    parser.add_argument("--host", default=os.getenv("LOCAL_LLM_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LOCAL_LLM_PORT", "8000")))
    parser.add_argument(
        "--upstream-base-url",
        default=os.getenv("LOCAL_LLM_UPSTREAM_BASE_URL", DEFAULT_UPSTREAM_BASE_URL),
        help="OpenAI-compatible upstream URL, for example http://127.0.0.1:11434/v1",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LOCAL_LLM_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("LOCAL_LLM_TIMEOUT", "120")),
    )
    parser.add_argument(
        "--max-requests-per-minute",
        type=int,
        default=int(os.getenv("LOCAL_LLM_RATE_LIMIT", "30")),
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=int(os.getenv("LOCAL_LLM_MAX_CONTEXT", "12000")),
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=int(os.getenv("LOCAL_LLM_MAX_COMPLETION", "2048")),
    )
    parser.add_argument(
        "--dashboard-history-limit",
        type=int,
        default=int(os.getenv("LOCAL_LLM_DASHBOARD_HISTORY_LIMIT", "200")),
    )
    args = parser.parse_args()
    return ServiceConfig(
        host=args.host,
        port=args.port,
        upstream_base_url=args.upstream_base_url,
        model=args.model,
        request_timeout=args.timeout,
        max_requests_per_minute=args.max_requests_per_minute,
        max_context_tokens=args.max_context_tokens,
        max_completion_tokens=args.max_completion_tokens,
        dashboard_history_limit=args.dashboard_history_limit,
    )


def main() -> None:
    config = parse_args()
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
    LocalLLMService(config).serve_forever()


if __name__ == "__main__":
    main()