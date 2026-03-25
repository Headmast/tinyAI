"""
День 8 — Демонстрация работы с токенами.

Показывает:
  1. Подсчёт токенов текущего запроса / истории / ответа
  2. Рост стоимости и токенов по мере диалога
  3. Что происходит при переполнении контекстного окна
  4. Управление длиной ответа через max_completion_tokens

Скрипт работает автономно — без реального API-запроса.
Все "ответы" симулируются заранее заготовленными данными.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List

from news_agent.token_counter import (
    TokenCounter,
    DialogTokenTracker,
    TokenBudget,
)


SEP = "═" * 68
SEP2 = "─" * 68


def _section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _subsection(title: str) -> None:
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


def demo_single_request_breakdown() -> None:
    """
    Часть 1: Подсчёт токенов одного запроса.

    Показывает разбивку по:
      — системному промпту
      — сообщению пользователя
      — ответу модели
      — overhead (структура сообщений)
    """
    _section("ЧАСТЬ 1 — Подсчёт токенов одного запроса")

    counter = TokenCounter(model="zai-org/GLM-4.7-Flash")
    print(f"\n  Метод подсчёта:  {counter.method_label}")
    print(f"  Точный счётчик:  {'Да (tiktoken)' if counter.is_exact else 'Нет (chars÷4 fallback)'}")

    system_prompt = (
        "Ты профессиональный журналист. Пиши новостные посты чётко и лаконично. "
        "Используй факты, избегай эмоциональных суждений."
    )
    user_message = "Напиши короткий новостной пост о росте цен на нефть до $90 за баррель."
    assistant_response = (
        "**Нефть достигла $90 за баррель**\n\n"
        "Мировые цены на нефть марки Brent поднялись до $90 за баррель впервые "
        "за три месяца на фоне сокращения добычи странами ОПЕК+. "
        "Аналитики предупреждают о возможном давлении на инфляцию в ключевых экономиках.\n\n"
        "#нефть #ОПЕК #экономика"
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_message},
    ]

    sys_tokens   = counter.count_text(system_prompt)
    user_tokens  = counter.count_text(user_message)
    resp_tokens  = counter.count_response(assistant_response)
    msg_info     = counter.count_messages(messages)
    breakdown    = counter.count_request_breakdown(messages, assistant_response)

    print(f"\n  {'Компонент':<30} {'Символов':>9}  {'Токенов':>9}")
    print(f"  {'─' * 52}")
    print(f"  {'Системный промпт':<30} {len(system_prompt):>9,}  {sys_tokens:>9,}")
    print(f"  {'Запрос пользователя':<30} {len(user_message):>9,}  {user_tokens:>9,}")
    print(f"  {'Overhead (роли + структура)':<30} {'':>9}  {msg_info['overhead']:>9,}")
    print(f"  {'─' * 52}")
    print(f"  {'Итого prompt':<30} {'':>9}  {breakdown['prompt_tokens']:>9,}")
    print(f"  {'Ответ модели':<30} {len(assistant_response):>9,}  {resp_tokens:>9,}")
    print(f"  {'─' * 52}")
    print(f"  {'TOTAL':<30} {'':>9}  {breakdown['total_tokens']:>9,}")

    print(f"\n  Ответ модели ({resp_tokens} токенов):")
    for line in textwrap.wrap(assistant_response[:200], width=64):
        print(f"    {line}")


def demo_dialog_growth() -> None:
    """
    Часть 2: Рост токенов по мере диалога.

    Симулирует многоходовой разговор и показывает, как токены
    накапливаются с каждым ходом.  Диалог длинный — 10 ходов.
    """
    _section("ЧАСТЬ 2 — Рост токенов в длинном диалоге (10 ходов)")

    model = "gpt-5-nano"
    counter = TokenCounter(model=model)
    max_ctx = 4096
    tracker = DialogTokenTracker(model=model, max_tokens=max_ctx)

    PROMPT_PRICE = 0.0002
    COMPLETION_PRICE = 0.00125

    turns: List[tuple] = [
        ("user",      "Привет! Объясни, что такое нейронная сеть, простыми словами."),
        ("assistant", "Нейронная сеть — это математическая модель, вдохновлённая работой мозга. "
                      "Она состоит из слоёв «нейронов», которые передают сигналы друг другу и "
                      "обучаются на примерах, постепенно улучшая свои предсказания."),
        ("user",      "А чем отличается глубокое обучение от обычного машинного обучения?"),
        ("assistant", "Глубокое обучение (Deep Learning) — это подмножество машинного обучения, "
                      "где используются нейронные сети с большим числом слоёв (отсюда «глубокое»). "
                      "Обычное ML опирается на ручное выделение признаков, а Deep Learning "
                      "учится извлекать их самостоятельно из сырых данных — изображений, текста, звука."),
        ("user",      "Какие задачи лучше всего решает Deep Learning?"),
        ("assistant", "Deep Learning особенно хорош в: распознавании изображений (CV), "
                      "обработке естественного языка (NLP), синтезе речи и музыки, "
                      "игровом ИИ (AlphaGo, AlphaStar) и медицинской диагностике. "
                      "Там, где данных много и задача требует сложного представления."),
        ("user",      "Что такое трансформер и почему он стал основой современных LLM?"),
        ("assistant", "Трансформер — архитектура нейронной сети, предложенная Google в 2017 году. "
                      "Ключевое новшество — механизм self-attention: каждый токен в тексте "
                      "«смотрит» на все остальные токены и взвешивает их важность. "
                      "Это позволяет обрабатывать длинные контексты параллельно, что намного "
                      "эффективнее рекуррентных сетей (LSTM/GRU). GPT, BERT, Claude, Gemini — "
                      "все они построены на трансформерах."),
        ("user",      "Сколько параметров у GPT-4 и что означают «параметры»?"),
        ("assistant", "Параметры — это числовые веса (матрицы чисел), которые нейросеть "
                      "обновляет в процессе обучения. Чем их больше, тем сложнее модель. "
                      "У GPT-4 параметры публично не раскрыты, но по оценкам — около "
                      "1 триллиона. GPT-3 имел 175 млрд. Больше ≠ лучше: качество зависит "
                      "от архитектуры, данных и методов обучения."),
    ]

    messages: List[Dict[str, Any]] = []

    _subsection("Пошаговый рост токенов")
    print(f"\n  Модель: {model}  |  Контекстное окно: {max_ctx:,} токенов")
    print(f"  Метод: {counter.method_label}\n")

    for role, text in turns:
        messages.append({"role": role, "content": text})
        snap = tracker.add_turn(role, text, messages)
        cost = tracker.cost_at_turn(snap.turn, PROMPT_PRICE, COMPLETION_PRICE)
        cum_cost = tracker.cumulative_cost(PROMPT_PRICE, COMPLETION_PRICE)
        print(
            f"  Ход {snap.turn:>2} [{role:9}]  msg={snap.message_tokens:>4} tok  "
            f"total={snap.total_tokens:>4,}  window={snap.growth_pct:>5.1f}%  "
            f"стоим.=${cum_cost:.6f}"
        )

    print()
    print(tracker.format_growth_table())

    print(f"\n  Итоговая суммарная стоимость: ${tracker.cumulative_cost(PROMPT_PRICE, COMPLETION_PRICE):.6f}")
    print(f"  {tracker.format_context_bar()}")


def demo_max_tokens_control() -> None:
    """
    Часть 3: Управление длиной ответа через max_completion_tokens.

    Показывает, как ограничение max_completion_tokens влияет
    на то, сколько токенов модель может использовать для ответа.
    """
    _section("ЧАСТЬ 3 — Управление длиной ответа (max_completion_tokens)")

    counter = TokenCounter(model="gpt-5.4")
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "Ты эксперт по искусственному интеллекту."},
        {"role": "user",   "content": "Расскажи подробно об истории развития ИИ."},
    ]

    msg_info = counter.count_messages(messages)
    prompt_tokens = msg_info["total"]

    context_limit = 128_000
    scenarios = [
        ("Очень короткий",   50),
        ("Короткий",        200),
        ("Средний",        1000),
        ("Длинный",        4000),
        ("Без ограничений", 8000),
    ]

    budget = TokenBudget(context_limit=context_limit, max_completion=8000)

    print(f"\n  Контекстное окно: {context_limit:,} токенов")
    print(f"  Токены промпта:   {prompt_tokens:,}")
    print(f"\n  {'Сценарий':<20} {'max_completion':>14}  {'effective':>9}  {'осталось окна':>13}")
    print(f"  {'─' * 62}")

    for name, max_tok in scenarios:
        b = TokenBudget(context_limit=context_limit, max_completion=max_tok)
        effective = b.effective_max_completion(prompt_tokens)
        remaining = context_limit - prompt_tokens - effective
        print(f"  {name:<20} {max_tok:>14,}  {effective:>9,}  {remaining:>13,}")

    print(f"\n  ПРИМЕР: промпт уже занял 120 000 токенов из 128 000")
    heavy_prompt = 120_000
    b_heavy = TokenBudget(context_limit=context_limit, max_completion=8000)
    eff = b_heavy.effective_max_completion(heavy_prompt)
    is_ov, msg = b_heavy.check_overflow(heavy_prompt)
    print(f"  {b_heavy.format_budget_line(heavy_prompt)}")
    if msg:
        print(f"  {'🔴' if is_ov else '⚠️ '} {msg}")

    print(f"\n  ПРИМЕР: промпт превысил лимит (130 000 токенов)")
    over_prompt = 130_000
    is_ov2, msg2 = b_heavy.check_overflow(over_prompt)
    print(f"  {b_heavy.format_budget_line(over_prompt)}")
    if msg2:
        print(f"  {'🔴' if is_ov2 else '⚠️ '} {msg2}")


def demo_overflow() -> None:
    """
    Часть 4: Что ломается при переполнении контекстного окна.

    Демонстрирует нарастание токенов до искусственно малого лимита (800 токенов)
    и что произошло бы при реальном API-вызове.
    """
    _section("ЧАСТЬ 4 — Переполнение контекстного окна")

    model = "gpt-5.4"
    TINY_LIMIT = 800
    counter = TokenCounter(model=model)
    tracker = DialogTokenTracker(model=model, max_tokens=TINY_LIMIT)
    budget = TokenBudget(context_limit=TINY_LIMIT, max_completion=200)

    print(f"\n  Используем искусственный лимит: {TINY_LIMIT} токенов")
    print(f"  (реальное окно модели — 128 000 токенов)\n")

    long_turns = [
        ("user",
         "Объясни принцип работы трансформерной архитектуры нейронных сетей с примерами."),
        ("assistant",
         "Трансформер обрабатывает последовательности токенов через механизм внимания (attention). "
         "Для каждого токена вычисляются три вектора: Query, Key, Value. "
         "Attention Score = softmax(QK^T / √d_k) × V. Это позволяет учитывать весь контекст "
         "одновременно, в отличие от рекуррентных сетей."),
        ("user",
         "Как именно работает multi-head attention и зачем нужны несколько голов?"),
        ("assistant",
         "Multi-head attention — это несколько параллельных механизмов внимания (голов). "
         "Каждая голова обучается фокусироваться на разных аспектах: синтаксис, семантика, "
         "кореференция и т.д. Результаты конкатенируются и проецируются обратно. "
         "Типично 8–32 головы. Это даёт модели возможность одновременно отслеживать "
         "разные виды зависимостей в тексте."),
        ("user",
         "Расскажи подробно про позиционное кодирование (positional encoding) в трансформерах."),
        ("assistant",
         "Поскольку attention не имеет встроенного понятия порядка, "
         "позиционное кодирование добавляет информацию о позиции каждого токена. "
         "В оригинальной работе Vaswani (2017): PE(pos,2i) = sin(pos/10000^(2i/d_model)), "
         "PE(pos,2i+1) = cos(pos/10000^(2i/d_model)). "
         "Современные модели используют RoPE (Rotary Position Embedding) или ALiBi, "
         "которые лучше обобщаются на длинные контексты."),
        ("user",
         "Объясни разницу между encoder-only, decoder-only и encoder-decoder архитектурами."),
    ]

    messages: List[Dict[str, Any]] = []
    overflowed = False

    print(f"  {'Ход':>4}  {'Role':9}  {'Prompt tok':>10}  {'Status':>18}")
    print(f"  {'─' * 50}")

    for role, text in long_turns:
        messages.append({"role": role, "content": text})
        snap = tracker.add_turn(role, text, messages)
        is_ov, ov_msg = budget.check_overflow(snap.total_tokens)
        eff_max = budget.effective_max_completion(snap.total_tokens)

        if is_ov:
            status = "🔴 OVERFLOW"
        elif eff_max < 50:
            status = "⚠️  критически мало"
        else:
            status = f"✅ ок (max={eff_max})"

        print(f"  {snap.turn:>4}  {role:9}  {snap.total_tokens:>10,}  {status:>18}")

        if is_ov and not overflowed:
            overflowed = True

    print()
    print(tracker.format_growth_table())

    print(f"\n  Что произошло бы при реальном API-вызове:")
    print(f"  ─────────────────────────────────────────")
    if overflowed:
        print(f"  🔴 API вернул бы ошибку: 'context_length_exceeded'")
        print(f"     или обрезал промпт без предупреждения (зависит от модели)")
        print(f"     Модель потеряла бы начало диалога — забыла бы контекст")
    else:
        print(f"  ✅ Диалог уложился в лимит")
    print(f"\n  Стратегии управления переполнением:")
    print(f"  • Обрезка старых сообщений (truncation)")
    print(f"  • Суммаризация истории через отдельный LLM-вызов")
    print(f"  • Векторный поиск (RAG) — только релевантный контекст")
    print(f"  • Увеличение контекстного окна (моделью с большим лимитом)")


def demo_cost_projection() -> None:
    """
    Часть 5: Проекция стоимости при масштабировании.

    Показывает, как быстро растёт стоимость при увеличении
    числа пользователей и длины диалогов.
    """
    _section("ЧАСТЬ 5 — Проекция стоимости при масштабировании")

    models_pricing = {
        "gpt-5-nano":   (0.0002,  0.00125),
        "gpt-5.4-mini": (0.00075, 0.0045),
        "gpt-5.4":      (0.0025,  0.015),
        "GLM-4.7-Flash": (0.0,    0.0),
    }

    avg_prompt_tokens = 2_000
    avg_completion_tokens = 500
    turns_per_session = 10
    sessions_per_day = 100

    print(f"\n  Допущения:")
    print(f"    Средний промпт:    {avg_prompt_tokens:,} токенов")
    print(f"    Средний ответ:     {avg_completion_tokens:,} токенов")
    print(f"    Ходов на сессию:   {turns_per_session}")
    print(f"    Сессий в день:     {sessions_per_day:,}")

    print(f"\n  {'Модель':<18} {'За сессию':>12}  {'В день':>10}  {'В месяц':>12}")
    print(f"  {'─' * 58}")

    for model_name, (pp, cp) in models_pricing.items():
        tokens_per_session_p = avg_prompt_tokens * turns_per_session
        tokens_per_session_c = avg_completion_tokens * turns_per_session
        cost_per_session = (tokens_per_session_p * pp + tokens_per_session_c * cp) / 1000
        cost_per_day = cost_per_session * sessions_per_day
        cost_per_month = cost_per_day * 30
        label = "$0.000000" if pp == 0.0 else f"${cost_per_session:.4f}"
        print(
            f"  {model_name:<18} {label:>12}  ${cost_per_day:>8.2f}  ${cost_per_month:>10.2f}"
        )

    print(f"\n  💡 Ключевой вывод: prompt-токены в длинных диалогах")
    print(f"     накапливаются экспоненциально — каждый ход включает")
    print(f"     ВСЮ предыдущую историю. Это главная причина роста стоимости.")


def main() -> None:
    print(SEP)
    print("  ДЕНЬ 8 — РАБОТА С ТОКЕНАМИ")
    print("  Демонстрация подсчёта, роста и управления токенами")
    print(SEP)

    demo_single_request_breakdown()
    demo_dialog_growth()
    demo_max_tokens_control()
    demo_overflow()
    demo_cost_projection()

    print(f"\n{SEP}")
    print("  ИТОГ")
    print(SEP)
    print("""
  Что реализовано:

  1. TokenCounter (news_agent/token_counter.py)
     — tiktoken для точного подсчёта (GPT/GLM → cl100k_base)
     — fallback: chars÷4 при отсутствии tiktoken
     — count_text / count_messages / count_response / count_request_breakdown

  2. DialogTokenTracker
     — снимок (TurnSnapshot) после каждого хода
     — таблица роста токенов с процентом окна
     — расчёт стоимости per-turn и суммарно

  3. TokenBudget
     — effective_max_completion: адаптирует лимит под реальный остаток
     — check_overflow: детектирует overflow ДО отправки запроса
     — format_budget_line: однострочное резюме бюджета

  4. Обновления агентов:
     — AgentLoop: max_completion_tokens + context_limit параметры
     — Per-iteration breakdown: prompt / completion / max_completion
     — Таблица токенов по итерациям в финале
     — Автостоп при overflow контекстного окна

  5. Обновления чата (_run_chat_loop):
     — TokenCounter вместо chars÷4
     — DialogTokenTracker встроен в цикл
     — Команда 'tokens' — таблица роста токенов
     — Метка [tiktoken] / [~chars÷4] в каждом ответе
""")


if __name__ == "__main__":
    main()
