import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request

import pytest

from local_llm_client import LocalLLMClientConfig, LocalLLMHttpClient
from local_llm_service import LocalLLMService, ServiceConfig


class FakeUpstreamHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        payload = json.loads(raw.decode("utf-8"))
        time.sleep(0.05)
        last_message = payload["messages"][-1]["content"]
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": payload.get("model", "qwen3:8b"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"echo:{last_message}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


@pytest.fixture
def upstream_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.fixture
def local_service_server(upstream_server):
    upstream_url = f"http://127.0.0.1:{upstream_server.server_port}/v1"
    config = ServiceConfig(
        host="127.0.0.1",
        port=0,
        upstream_base_url=upstream_url,
        model="qwen3:8b",
        request_timeout=5.0,
        max_requests_per_minute=20,
        max_context_tokens=100,
        max_completion_tokens=2048,
    )
    service = LocalLLMService(config)
    server = ThreadingHTTPServer((config.host, 0), service._build_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.fixture
def client(local_service_server):
    return LocalLLMHttpClient(
        LocalLLMClientConfig(
            base_url=f"http://127.0.0.1:{local_service_server.server_port}",
            timeout=5.0,
            default_model="qwen3:8b",
        )
    )


def test_health_endpoint(client):
    payload = client.health()
    assert payload["status"] == "ok"
    assert payload["model"] == "qwen3:8b"


def test_models_endpoint(client):
    payload = client.list_models()
    assert payload["data"][0]["id"] == "qwen3:8b"


def test_chat_endpoint_returns_reply(client):
    payload = client.chat("hello")
    assert payload["reply"] == "echo:hello"
    assert payload["usage"]["total_tokens"] == 15


def test_parallel_requests_are_stable(client):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(client.chat, f"hello-{index}") for index in range(4)]
    replies = [future.result()["reply"] for future in futures]
    assert sorted(replies) == [f"echo:hello-{index}" for index in range(4)]


def test_context_limit_is_enforced(client):
    with pytest.raises(RuntimeError) as exc:
        client.chat("x" * 1000)
    assert "Prompt exceeds max context limit" in str(exc.value)


def test_rate_limit_is_enforced(upstream_server):
    service = LocalLLMService(
        ServiceConfig(
            host="127.0.0.1",
            port=0,
            upstream_base_url=f"http://127.0.0.1:{upstream_server.server_port}/v1",
            model="qwen3:8b",
            request_timeout=5.0,
            max_requests_per_minute=2,
            max_context_tokens=100,
            max_completion_tokens=2048,
        )
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), service._build_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    client = LocalLLMHttpClient(LocalLLMClientConfig(base_url=base_url, timeout=5.0))
    try:
        client.chat("one")
        client.chat("two")
        with pytest.raises(RuntimeError) as exc:
            client.chat("three")
        assert "HTTP 429" in str(exc.value)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_dashboard_index_is_served(local_service_server):
    url = f"http://127.0.0.1:{local_service_server.server_port}/dashboard"
    with request.urlopen(url, timeout=5.0) as response:
        body = response.read().decode("utf-8")
        assert response.status == 200
        assert "TinyAI Local LLM Dashboard" in body


def test_dashboard_metrics_and_history(client, local_service_server):
    client.chat("dashboard-test")

    metrics_url = f"http://127.0.0.1:{local_service_server.server_port}/dashboard/api/metrics"
    with request.urlopen(metrics_url, timeout=5.0) as response:
        metrics = json.loads(response.read().decode("utf-8"))
    assert metrics["total_requests"] >= 1
    assert metrics["total_tokens"] >= 15

    history_url = (
        f"http://127.0.0.1:{local_service_server.server_port}/dashboard/api/history?limit=5"
    )
    with request.urlopen(history_url, timeout=5.0) as response:
        history = json.loads(response.read().decode("utf-8"))
    assert history["count"] >= 1
    first = history["items"][0]
    assert first["endpoint"] == "/chat"
    assert "dashboard-test" in first["prompt_preview"]


def test_dashboard_status_has_active_requests(local_service_server):
    status_url = f"http://127.0.0.1:{local_service_server.server_port}/dashboard/api/status"
    with request.urlopen(status_url, timeout=5.0) as response:
        payload = json.loads(response.read().decode("utf-8"))

    assert payload["status"] == "ok"
    assert payload["model"] == "qwen3:8b"
    assert "active_requests" in payload
    assert "limits" in payload