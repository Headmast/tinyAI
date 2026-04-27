from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from local_llm_client import LocalLLMClientConfig, LocalLLMHttpClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check TinyAI local LLM service")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--prompt", default="Сформулируй один факт о локальных LLM.")
    parser.add_argument("--max-completion-tokens", type=int, default=256)
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

    print("[1] Health")
    print(json.dumps(client.health(), ensure_ascii=False, indent=2))

    print("\n[2] Network chat")
    first = client.chat(
        args.prompt,
        max_completion_tokens=args.max_completion_tokens,
    )
    print(json.dumps(first, ensure_ascii=False, indent=2))

    print("\n[3] Parallel stability")
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [
            executor.submit(
                client.chat,
                f"{args.prompt} [{index}]",
                max_completion_tokens=args.max_completion_tokens,
            )
            for index in range(args.parallel)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    print(f"Completed {len(results)} parallel requests")

    print("\n[4] Limits")
    try:
        client.chat(
            "x" * 80_000,
            max_completion_tokens=args.max_completion_tokens,
        )
        print("Context limit check: service accepted the long prompt")
    except RuntimeError as exc:
        print(f"Context limit check: {exc}")


if __name__ == "__main__":
    main()