"""
День 9 — Демонстрация компрессии контекста на космической теме.

Скрипт ведёт реальную беседу с LLM (gpt-5.4-mini через OpenAI):
  — 12 вопросов о космосе
  — после каждого ответа показывается рост контекста
  — на 10-м сообщении автоматически запускается суммаризация
  — в конце сравниваются токены с/без сжатия

Запуск:
    python3 demo_compression.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from news_agent.session_manager import ConversationSession
from news_agent.token_counter import TokenCounter
from llm_cli import stream_response  # reuse streaming + thinking support

load_dotenv()

SEP  = "═" * 68
SEP2 = "─" * 68

# ─────────────────────────────────────────────────────────────
# Настройки
# ─────────────────────────────────────────────────────────────

MODEL           = "gpt-5.4-mini"
SUMMARIZE_EVERY = 10   # суммаризация каждые N chat-сообщений
KEEP_LAST_N     = 6    # хранить N последних без сжатия

SPACE_QUESTIONS = [
    "Что такое тёмная материя и почему мы её не можем напрямую увидеть?",
    "Как работает гравитационное линзирование и какой вклад оно внесло в астрономию?",
    "Чем отличаются нейтронные звёзды от чёрных дыр по физическим свойствам?",
    "Что такое реликтовое излучение и что оно говорит о начале Вселенной?",
    "Почему Марс потерял свою атмосферу, а Земля — нет?",
    "Что такое парадокс Ферми и какие существуют гипотезы его объяснения?",
    "Как происходит процесс формирования планетарных систем из протопланетного диска?",
    "Что такое квазары и почему они светят так ярко?",
    "Объясни принцип работы космического телескопа «Джеймс Уэбб» и его главные достижения.",
    "Можно ли путешествовать быстрее скорости света — что говорит об этом физика?",
    "Что такое приливное разрушение звезды вблизи чёрной дыры?",
    "Каков современный научный консенсус о возможности жизни на спутнике Юпитера Европа?",
]

SYSTEM_PROMPT = (
    "Ты астрофизик и популяризатор космоса. "
    "Отвечай понятно, но точно. Каждый ответ — 3–5 предложений. "
    "Избегай излишних вводных слов."
)


# ─────────────────────────────────────────────────────────────
# Утилиты вывода
# ─────────────────────────────────────────────────────────────

def _header(text: str) -> None:
    print(f"\n{SEP}")
    print(f"  {text}")
    print(SEP)


def _sub(text: str) -> None:
    print(f"\n{SEP2}")
    print(f"  {text}")
    print(SEP2)


def _get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("❌ OPENAI_API_KEY не найден в .env")
        sys.exit(1)
    return OpenAI(
        api_key=key,
        timeout=180.0,
    )


# ─────────────────────────────────────────────────────────────
# Отправка запроса через stream_response (поддержка thinking)
# ─────────────────────────────────────────────────────────────

def _send(client: OpenAI, session: ConversationSession, question: str) -> tuple:
    """
    Добавляет вопрос в сессию, отправляет запрос через stream_response,
    возвращает (answer, prompt_tokens, completion_tokens).
    """
    session.add_user_message(question)
    messages = session.get_messages_for_api()

    params = {
        "model": MODEL,
        "messages": messages,
        "max_completion_tokens": 400,
    }

    answer, reasoning, api_usage = stream_response(client, params, show_thinking=False)
    session.add_assistant_message(answer)

    pt = api_usage["prompt_tokens"]   if api_usage else 0
    ct = api_usage["completion_tokens"] if api_usage else 0
    session.update_token_usage(pt, ct)
    return answer, pt, ct


# ─────────────────────────────────────────────────────────────
# Суммаризация: используем тот же stream_response без показа thinking
# ─────────────────────────────────────────────────────────────

def _summarize(client: OpenAI, compressor, messages: list, model: str) -> str:
    """Override суммаризатора через stream_response вместо обычного create()."""
    from news_agent.context_compressor import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE

    chat_only = [m for m in messages if m["role"] in ("user", "assistant")]
    to_compress = chat_only[compressor.summary_covers_up_to:
                            len(chat_only) - compressor.keep_last_n]
    if not to_compress:
        return ""

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in to_compress
    )
    summary_messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": SUMMARY_USER_TEMPLATE.format(history=history_text)},
    ]
    params = {
        "model": model,
        "messages": summary_messages,
        "max_completion_tokens": 600,
    }
    print(f"\n  🔄 Генерирую summary [{len(to_compress)} сообщений] ...")
    text, _, _ = stream_response(client, params, show_thinking=False)
    return text.strip()

# ─────────────────────────────────────────────────────────────
# Основная демонстрация
# ─────────────────────────────────────────────────────────────

def run_demo() -> None:
    client = _get_client()
    counter = TokenCounter(model=MODEL)

    _header("ДЕНЬ 9 — КОМПРЕССИЯ КОНТЕКСТА  |  Космическая беседа")
    print(f"""
  Параметры:
    Модель:                    {MODEL}
    Суммаризация каждые:       {SUMMARIZE_EVERY} chat-сообщений
    Живой хвост (keep_last_n): {KEEP_LAST_N} сообщений
    Вопросов:                  {len(SPACE_QUESTIONS)}
    """)

    # ── Создаём сессию с компрессией ────────────────────────────
    session = ConversationSession(
        model=MODEL,
        name="Космос — компрессия",
        system_prompt=SYSTEM_PROMPT,
    )
    session.enable_compression(
        summarize_every=SUMMARIZE_EVERY,
        keep_last_n=KEEP_LAST_N,
    )

    # ── Диалог ──────────────────────────────────────────────────
    for i, question in enumerate(SPACE_QUESTIONS, start=1):
        print(f"\n{SEP2}")
        chat_count = len([m for m in session.messages if m["role"] in ("user", "assistant")])
        print(f"  Вопрос {i:02d}/{len(SPACE_QUESTIONS)}  "
              f"[chat-сообщений в истории: {chat_count}  "
              f"суммаризировано: {session.compressor.summary_covers_up_to}]")
        print(f"  🧑 {question}")

        t0 = time.monotonic()
        answer, pt, ct = _send(client, session, question)
        elapsed = int((time.monotonic() - t0) * 1000)

        # ── Проверяем и запускаем компрессию ────────────────────
        if session.compressor.needs_compression(session.messages):
            summary_text = _summarize(client, session.compressor, session.messages, MODEL)
            if summary_text:
                from news_agent.token_counter import TokenCounter as TC
                _c = TC(model=MODEL)
                tokens_before = _c.count_messages(
                    session.compressor.get_compressed_messages(session.messages)
                )["total"]

                chat_only = [m for m in session.messages
                             if m["role"] in ("user", "assistant")]
                compress_end = len(chat_only) - session.compressor.keep_last_n
                compressed_n = compress_end - session.compressor.summary_covers_up_to

                session.compressor.summaries.append(summary_text)
                session.compressor.summary_covers_up_to += compressed_n
                session.compressor.compression_count += 1

                tokens_after = _c.count_messages(
                    session.compressor.get_compressed_messages(session.messages)
                )["total"]
                saved = max(0, tokens_before - tokens_after)
                session.compressor.total_tokens_saved += saved

                ratio_pct = (1.0 - tokens_after / max(tokens_before, 1)) * 100
                print(
                    f"\n  🔄 Сжатие #{session.compressor.compression_count}: "
                    f"{compressed_n} сообщ. → 1 summary  |  "
                    f"~{saved:,} токенов сэкономлено  |  сжатие {ratio_pct:.0f}%"
                )
                cmp_data = session.compressor.get_comparison(session.messages)
                print(
                    f"  📊 API-запрос: полная история {cmp_data['full_tokens']:,} → "
                    f"сжато {cmp_data['compressed_tokens']:,} токенов "
                    f"(−{cmp_data['savings_pct']:.0f}%)\n"
                )

        # ── Статистика строки ────────────────────────────────────
        api_msgs = session.get_messages_for_api()
        api_tokens = counter.count_messages(api_msgs)["total"]

        print(
            f"  {session.format_context_bar()}\n"
            f"  API-запрос: {api_tokens:,} токенов ({len(api_msgs)} сообщ. в пакете)  "
            f"| API: prompt={pt:,} compl={ct:,}  |  {elapsed}мс"
        )

        time.sleep(0.5)

    # ── Финальное сравнение ─────────────────────────────────────
    _header("ИТОГ: СРАВНЕНИЕ ТОКЕНОВ")

    cmp_data = session.compressor.get_comparison(session.messages)
    full_tokens = cmp_data["full_tokens"]
    compressed_tokens = cmp_data["compressed_tokens"]
    saved = cmp_data["tokens_saved"]
    savings_pct = cmp_data["savings_pct"]

    print(f"""
  {'Метрика':<38} {'Без сжатия':>10} {'Со сжатием':>10}
  {'─' * 62}
  {'Токенов в API-запросе':<38} {full_tokens:>10,} {compressed_tokens:>10,}
  {'Сообщений в API-запросе':<38} {len(session.messages):>10} {cmp_data['compressed_messages_count']:>10}
  {'─' * 62}
  {'Сэкономлено токенов':<38} {'':>10} {saved:>10,}
  {'Экономия':<38} {'':>10} {savings_pct:>9.1f}%
  {'─' * 62}
  Суммаризаций выполнено:          {cmp_data['compression_count']}
""")

    _sub("ДЕТАЛИ КОМПРЕССОРА")
    print(session.compressor.format_stats())
    print()
    print(session.compressor.format_summary_preview())
    print()

    _sub("НАКОПЛЕННЫЕ ТОКЕНЫ ЗА ВСЮ СЕССИЮ (реальные от API)")
    tu = session.token_usage
    print(f"""
  prompt_tokens:     {tu['prompt_tokens']:>10,}
  completion_tokens: {tu['completion_tokens']:>10,}
  total_tokens:      {tu['total_tokens']:>10,}
""")

    print(f"\n{SEP}")
    print("  Демонстрация завершена.")
    print(SEP)


if __name__ == "__main__":
    run_demo()

