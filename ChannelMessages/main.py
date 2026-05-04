"""
main.py — entry point for the Telegram Channel Stats Collector.

Usage:
    python main.py

On first run, Telethon will prompt for your phone number and an SMS/app code.
Subsequent runs reuse the saved session file (.session) without re-authentication.
"""

import asyncio
import logging
import sys

from db import init_db
from collector import collect


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _print_summary(summaries: list[dict]) -> None:
    if not summaries:
        print("No broadcast channels found in your account.")
        return

    print(f"\n{'='*64}")
    print(f"  Channel Stats — {len(summaries)} channels collected")
    print(f"{'='*64}")

    header = f"{'Channel':<35} {'Msgs':>4} {'Avg chars':>9} {'Media':>5} {'Top sender'}"
    print(header)
    print("-" * 64)

    for s in sorted(summaries, key=lambda x: x["title"].lower()):
        name = (s["title"] or "")[:34]
        print(
            f"{name:<35} {s['msg_count']:>4} {s['avg_chars']:>9.1f}"
            f" {s['media_count']:>5}  {s['top_sender']}"
        )

    print("=" * 64)
    print("Results saved to the database.\n")


async def main() -> None:
    print("Initialising database …")
    await init_db()

    print(f"Connecting to Telegram …")
    try:
        summaries = await collect()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_summary(summaries)


if __name__ == "__main__":
    asyncio.run(main())
