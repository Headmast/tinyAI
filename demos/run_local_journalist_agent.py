from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from journalist_agent import JournalistAgent, InvariantStore


def normalize_openai_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return base_url + "/v1"


def build_client(base_url: str, timeout: float) -> OpenAI:
    return OpenAI(
        api_key="local-service",
        base_url=normalize_openai_base_url(base_url),
        timeout=timeout,
    )


def run_repl(args: argparse.Namespace) -> None:
    client = build_client(args.base_url, args.timeout)
    agent = JournalistAgent(
        client=client,
        model=args.model,
        invariant_store=InvariantStore(),
        verbose=False,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        max_history=args.max_history,
    )

    print("Journalist agent via local LLM service")
    print(f"Endpoint: {normalize_openai_base_url(args.base_url)}")
    print(f"Model:    {args.model}")
    print("Commands: /reset, /tokens, /quit")

    while True:
        try:
            user_input = input("journalist> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in {"/quit", "/exit", "quit", "exit"}:
            break
        if user_input == "/reset":
            agent.reset_dialog()
            continue
        if user_input == "/tokens":
            print(agent.get_total_tokens())
            continue

        response = agent.chat(user_input)
        status = "OK" if response.allowed else "REFUSAL"
        print(f"[{status}]\n{response.text}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JournalistAgent via local HTTP LLM service")
    parser.add_argument("--base-url", default=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.getenv("LOCAL_LLM_MODEL", "qwen3:8b"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("LOCAL_LLM_TIMEOUT", "120")))
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=int(os.getenv("LOCAL_LLM_MAX_COMPLETION", "1500")),
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-history", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run_repl(parse_args())