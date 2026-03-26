"""
News Agent CLI — AI-агент для генерации новостных постов + диалоговый чат с историей.

Команды генерации постов:
  generate <тема>          — создать пост через 5-шаговый pipeline
  generate -t <тип> <тема> — с указанием типа (breaking/analysis/digest/social/press)
  agent <тема>             — автономный ReAct-агент
  batch <файл>             — пакетная генерация из файла тем
  history [n]              — последние n постов (по умолчанию 10)
  export <id> <формат>     — экспортировать пост (md/html/telegram/json/plain)
  template list            — список типов постов
  template show <тип>      — показать описание типа

Диалоговый чат (с сохранением истории):
  chat                     — начать новую сессию
  chat new [имя]           — начать именованную сессию
  chat list                — список сохранённых сессий
  chat load <id>           — загрузить и продолжить сессию
  chat resume <id>         — продолжить незаконченную сессию (alias: load)
  chat delete <id>         — удалить сессию
  chat info [id]           — информация о сессии и использование контекста

Внутри чата:
  info                     — показать использование контекстного окна
  tokens                   — таблица роста токенов по ходу диалога
  compress                 — статус компрессии истории
  compress on              — включить компрессию (суммаризация каждые 10 сообщений)
  compress off             — выключить компрессию
  compress compare         — сравнить токены с/без компрессии
  close / q                — закрыть сессию и вернуться в главное меню

Ручные запросы к API:
  api                      — простой режим (только выбор модели)
  api advanced             — расширенный режим (настройка всех параметров под модель)

Статистика:
  stats                    — накопленная статистика (запросы, токены, стоимость)
  stats reset              — сбросить статистику (подтверждение обязательно)

Модели:
  models                   — список доступных моделей
  model <name>             — переключить модель
  quit / exit / q          — выход
"""

import os
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from news_agent.storage import PostStorage
from news_agent.pipeline import NewsPipeline
from news_agent.agent import AgentLoop
from news_agent.roles import POST_TYPE_GUIDES
from news_agent.session_manager import ConversationSession, SessionStorage, MODEL_CONTEXT_SIZES
from news_agent.usage_tracker import UsageTracker, RequestTimer
from news_agent.token_counter import TokenCounter, DialogTokenTracker
from news_agent.context_compressor import ContextCompressor

load_dotenv()


def get_available_models():
    return {
        "zai-org/GLM-4.7-Flash": {
            "name": "GLM-4.7-Flash",
            "prompt_price": 0.0,
            "completion_price": 0.0,
            "description": "Быстрая облачная модель от Cloud.ru",
            "provider": "cloud_ru",
            "id": 1
        },
        "zai-org/GLM-4.7": {
            "name": "GLM-4.7",
            "prompt_price": 0.0,
            "completion_price": 0.0,
            "description": "Более мощная модель от Cloud.ru",
            "provider": "cloud_ru",
            "id": 2
        },
        "gpt-5-nano": {
            "name": "GPT-5 Nano",
            "prompt_price": 0.0002,
            "completion_price": 0.00125,
            "cached_prompt_price": 0.00002,
            "description": "Компактная версия GPT-5 от OpenAI",
            "provider": "openai",
            "id": 3
        },
        "gpt-5.4": {
            "name": "GPT-5.4",
            "prompt_price": 0.0025,
            "completion_price": 0.015,
            "cached_prompt_price": 0.00025,
            "description": "Новейшая модель GPT-5, высокая производительность",
            "provider": "openai",
            "id": 4
        },
        "gpt-5.4-mini": {
            "name": "GPT-5.4 Mini",
            "prompt_price": 0.00075,
            "completion_price": 0.0045,
            "cached_prompt_price": 0.00008,
            "description": "Облегченная версия GPT-5.4, оптимальная для большинства задач",
            "provider": "openai",
            "id": 5
        }
    }

MODEL_PARAMS_SCHEMA = {
    "zai-org/GLM-4.7-Flash": {
        "temperature":   {"type": float, "min": 0.0, "max": 1.0, "default": 0.7,  "desc": "Случайность (0=детерм., 1=макс.)"},
        "max_completion_tokens": {"type": int,   "min": 1,   "max": 128000, "default": 8000, "desc": "Макс. токенов в ответе"},
        "top_p":         {"type": float, "min": 0.0, "max": 1.0, "default": 1.0,  "desc": "Nucleus sampling (0–1)"},
        "thinking":      {"type": bool,  "default": True,  "desc": "Включить цепочку размышлений (GLM)"},
    },
    "zai-org/GLM-4.7": {
        "temperature":   {"type": float, "min": 0.0, "max": 1.0, "default": 0.7,  "desc": "Случайность"},
        "max_completion_tokens": {"type": int,   "min": 1,   "max": 128000, "default": 8000, "desc": "Макс. токенов в ответе"},
        "top_p":         {"type": float, "min": 0.0, "max": 1.0, "default": 1.0,  "desc": "Nucleus sampling"},
        "thinking":      {"type": bool,  "default": True,  "desc": "Включить цепочку размышлений (GLM)"},
    },
    "gpt-5-nano": {
        "max_completion_tokens": {"type": int,   "min": 1,   "max": 128000, "default": 8000,  "desc": "Макс. токенов в ответе"},
        "top_p":         {"type": float, "min": 0.0, "max": 1.0, "default": 1.0,  "desc": "Nucleus sampling"},
        "presence_penalty":  {"type": float, "min": -2.0, "max": 2.0, "default": 0.0, "desc": "Штраф за повторение тем (+поощряет новые)"},
        "frequency_penalty": {"type": float, "min": -2.0, "max": 2.0, "default": 0.0, "desc": "Штраф за частые токены (+уменьшает повторы)"},
    },
    "gpt-5.4": {
        "temperature":   {"type": float, "min": 0.0, "max": 2.0, "default": 0.7,  "desc": "Случайность (0=детерм., 2=макс.)"},
        "max_completion_tokens": {"type": int,   "min": 1,   "max": 128000, "default": 8000, "desc": "Макс. токенов в ответе"},
        "top_p":         {"type": float, "min": 0.0, "max": 1.0, "default": 1.0,  "desc": "Nucleus sampling"},
        "presence_penalty":  {"type": float, "min": -2.0, "max": 2.0, "default": 0.0, "desc": "Штраф за повторение тем"},
        "frequency_penalty": {"type": float, "min": -2.0, "max": 2.0, "default": 0.0, "desc": "Штраф за частые токены"},
    },
    "gpt-5.4-mini": {
        "temperature":   {"type": float, "min": 0.0, "max": 2.0, "default": 0.7,  "desc": "Случайность"},
        "max_completion_tokens": {"type": int,   "min": 1,   "max": 128000, "default": 8000, "desc": "Макс. токенов в ответе"},
        "top_p":         {"type": float, "min": 0.0, "max": 1.0, "default": 1.0,  "desc": "Nucleus sampling"},
        "presence_penalty":  {"type": float, "min": -2.0, "max": 2.0, "default": 0.0, "desc": "Штраф за повторение тем"},
        "frequency_penalty": {"type": float, "min": -2.0, "max": 2.0, "default": 0.0, "desc": "Штраф за частые токены"},
    },
}


def _get_default_params(model_name: str) -> dict:
    """Возвращает словарь параметров со значениями по умолчанию для модели."""
    schema = MODEL_PARAMS_SCHEMA.get(model_name, MODEL_PARAMS_SCHEMA["zai-org/GLM-4.7-Flash"])
    return {k: v["default"] for k, v in schema.items()}


def _set_param(params: dict, model_name: str, key: str, raw_value: str) -> str:
    """
    Устанавливает параметр с валидацией. Возвращает строку-результат.
    """
    schema = MODEL_PARAMS_SCHEMA.get(model_name, {})
    if key not in schema:
        return f"❌ Параметр '{key}' не поддерживается моделью {model_name}"
    spec = schema[key]
    try:
        if spec["type"] is bool:
            val = raw_value.lower() in ("1", "true", "yes", "да", "on")
        elif spec["type"] is int:
            val = int(raw_value)
        else:
            val = float(raw_value)
    except ValueError:
        return f"❌ Значение '{raw_value}' не является {spec['type'].__name__}"

    if spec["type"] is not bool:
        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and val < lo:
            return f"❌ Минимум {lo}"
        if hi is not None and val > hi:
            return f"❌ Максимум {hi}"
    params[key] = val
    return f"✓ {key} = {val}"


def _print_params(params: dict, model_name: str) -> None:
    """Выводит текущие параметры с описаниями."""
    schema = MODEL_PARAMS_SCHEMA.get(model_name, {})
    print(f"\n  Параметры для {model_name}:")
    print(f"  {'─' * 55}")
    for k, v in params.items():
        desc = schema.get(k, {}).get("desc", "")
        spec = schema.get(k, {})
        if spec.get("type") is not bool:
            lo = spec.get("min", "")
            hi = spec.get("max", "")
            rng = f"[{lo}–{hi}]" if lo != "" else ""
        else:
            rng = "[true/false]"
        print(f"    {k:24} = {str(v):>8}  {rng:16}  {desc}")
    print(f"  {'─' * 55}\n")


def get_client_for_model(model_name, clients):
    """Возвращает соответствующий client в зависимости от провайдера модели"""
    models = get_available_models()
    if model_name not in models:
        model_name = "zai-org/GLM-4.7-Flash"
    
    provider = models[model_name]["provider"]
    return clients[provider]

def calculate_cost(prompt_tokens, completion_tokens, model_name):
    models = get_available_models()
    if model_name not in models:
        model_name = "zai-org/GLM-4.7-Flash"
    
    model = models[model_name]
    prompt_cost = (prompt_tokens * model["prompt_price"]) / 1000
    completion_cost = (completion_tokens * model["completion_price"]) / 1000
    return prompt_cost + completion_cost

def setup_logging():
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"conversation_{timestamp}.json"
    return log_file

def log_interaction(log_file, user_input, assistant_response, usage_info, metadata=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "assistant_response": assistant_response,
        "usage": usage_info,
        "metadata": metadata or {}
    }
    
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def get_model_params(model_name, base_params):
    """Адаптирует параметры под конкретную модель"""
    models = get_available_models()
    if model_name not in models:
        model_name = "zai-org/GLM-4.7-Flash"
    
    provider = models[model_name]["provider"]
    params = base_params.copy()
    
    # gpt-5-nano не поддерживает temperature != 1
    if model_name == "gpt-5-nano" and "temperature" in params:
        del params["temperature"]
    
    return params

def get_available_modes(model_name="zai-org/GLM-4.7-Flash"):
    return [
        {
            "id": 1,
            "name": "Без ограничений",
            "params": {
                "model": model_name,
                "messages": [],
                "max_completion_tokens": 120000,
                "temperature": 0.7
            },
            "metadata": {"mode": "unrestricted"}
        },
        {
            "id": 2,
            "name": "С явным форматом ответа",
            "params": {
                "model": model_name,
                "messages": [],
                "max_completion_tokens": 120000,
                "temperature": 0.7
            },
            "system_prompt": "Структурируй свой ответ следующим образом:\n1. ОСНОВНОЙ ОТВЕТ: (подробный ответ на вопрос)\n2. КРАТКОЕ РЕЗЮМЕ: (1-2 предложения)\n3. КЛЮЧЕВЫЕ МОМЕНТЫ: (список из 2-3 пунктов)",
            "metadata": {"mode": "formatted", "format": "Structured text with sections"}
        },
        {
            "id": 3,
            "name": "С ограничением длины",
            "params": {
                "model": model_name,
                "messages": [],
                "max_completion_tokens": 100,
                "temperature": 0.7
            },
            "system_prompt": "Отвечай предельно кратко и лаконично. Максимум 50 слов.",
            "metadata": {"mode": "length_limited", "max_completion_tokens": 100, "max_words": 50}
        },
        {
            "id": 4,
            "name": "С явной инструкцией завершения",
            "params": {
                "model": model_name,
                "messages": [],
                "max_completion_tokens": 120000,
                "temperature": 0.7
            },
            "system_prompt": "После завершения ответа обязательно добавь маркер [КОНЕЦ]. Это важно для обозначения конца ответа.",
            "metadata": {"mode": "with_end_marker", "instruction": "Add [КОНЕЦ] marker"}
        },
        {
            "id": 5,
            "name": "С выводом в формате JSON",
            "params": {
                "model": model_name,
                "messages": [],
                "max_completion_tokens": 120000,
                "temperature": 0.7
            },
            "system_prompt": "Отвечай СТРОГО в формате JSON с полями:\n- \"answer\": подробный ответ на вопрос\n- \"summary\": краткое резюме в 1-2 предложения\n- \"key_points\": массив из 2-3 ключевых моментов\nВозвращай только валидный JSON, без дополнительного текста.",
            "metadata": {"mode": "json_format", "format": "JSON with answer, summary, and key_points"}
        },
        {
            "id": 6,
            "name": "С метапромптингом (двухэтапный)",
            "params": {
                "model": model_name,
                "messages": [],
                "max_completion_tokens": 120000,
                "temperature": 0.7
            },
            "metadata": {"mode": "meta_prompting", "two_stage": True}
        }
    ]

def count_words(text):
    return len(text.split())


def _build_token_comparison(
    local_prompt: int,
    local_completion: int,
    api_usage: Optional[dict],
    counter_method: str,
) -> Optional[dict]:
    """
    Строит словарь сравнения локального подсчёта токенов с реальными данными API.

    Возвращает None если API usage недоступен.
    Возвращает dict:
        has_api      — True если API вернул usage
        api_prompt   — реальные prompt-токены от API
        api_compl    — реальные completion-токены от API
        local_prompt — локальный расчёт (TokenCounter)
        local_compl  — локальный расчёт (TokenCounter)
        diff_prompt  — отклонение (local - api)
        diff_compl   — отклонение (local - api)
        pct_prompt   — % отклонения prompt
        pct_compl    — % отклонения completion
        method       — "tiktoken" или "~chars÷4"
    """
    if not api_usage:
        return {"has_api": False, "method": counter_method}

    api_p = api_usage["prompt_tokens"]
    api_c = api_usage["completion_tokens"]
    diff_p = local_prompt - api_p
    diff_c = local_completion - api_c
    pct_p = diff_p / api_p * 100 if api_p else 0.0
    pct_c = diff_c / api_c * 100 if api_c else 0.0

    return {
        "has_api":      True,
        "api_prompt":   api_p,
        "api_compl":    api_c,
        "api_total":    api_usage["total_tokens"],
        "local_prompt": local_prompt,
        "local_compl":  local_completion,
        "diff_prompt":  diff_p,
        "diff_compl":   diff_c,
        "pct_prompt":   pct_p,
        "pct_compl":    pct_c,
        "method":       counter_method,
    }


def _format_token_comparison(cmp: Optional[dict]) -> str:
    """Форматирует сравнение токенов в одну-две строки для вывода в чат."""
    if not cmp or not cmp.get("has_api"):
        return ""

    def _sign(n: float) -> str:
        return f"+{n:.1f}" if n >= 0 else f"{n:.1f}"

    p_diff = _sign(cmp["pct_prompt"])
    c_diff = _sign(cmp["pct_compl"])

    accuracy_icon = "✅" if abs(cmp["pct_prompt"]) < 5 and abs(cmp["pct_compl"]) < 5 else "📐"

    return (
        f"   {accuracy_icon} API:   prompt={cmp['api_prompt']:,}  completion={cmp['api_compl']:,}  total={cmp['api_total']:,}\n"
        f"      local [{cmp['method']}]: prompt={cmp['local_prompt']:,} ({p_diff}%)  "
        f"completion={cmp['local_compl']:,} ({c_diff}%)"
    )


def _chat_send(
    client,
    session: ConversationSession,
    user_input: str,
    max_completion_tokens: int = 8000,
) -> tuple:
    """
    Отправляет сообщение модели с полной историей диалога и возвращает ответ.

    Именно здесь реализуется «память» через API:
      Веб-чат — браузер шлёт только новое сообщение, сервер хранит историю сам.
      API     — каждый вызов stateless, мы явно передаём весь messages[] каждый раз.

    Использует TokenCounter для точного (tiktoken) или эвристического подсчёта токенов.

    Returns:
        (response_text, prompt_tokens, completion_tokens)
    """
    session.add_user_message(user_input)
    messages = session.get_messages_for_api()

    counter = TokenCounter(model=session.model)
    msg_info = counter.count_messages(messages)
    prompt_tokens = msg_info["total"]

    params: dict = {
        "model": session.model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "temperature": 0.7,
    }
    if session.model == "gpt-5-nano":
        del params["temperature"]

    response_text, reasoning, api_usage = stream_response(client, params, show_thinking=True)

    session.add_assistant_message(response_text)

    local_completion = counter.count_response(response_text + reasoning)
    session.update_token_usage(prompt_tokens, local_completion)

    token_comparison = _build_token_comparison(
        local_prompt=prompt_tokens,
        local_completion=local_completion,
        api_usage=api_usage,
        counter_method=counter.method_label,
    )

    final_prompt = api_usage["prompt_tokens"] if api_usage else prompt_tokens
    final_completion = api_usage["completion_tokens"] if api_usage else local_completion
    return response_text, final_prompt, final_completion, token_comparison


def _print_session_info(session: ConversationSession) -> None:
    """Выводит информацию о сессии и визуальный индикатор использования контекста."""
    info = session.get_context_info()
    cmp = session.get_compression_info()
    print(f"\n{'─' * 55}")
    print(f"  Сессия:    {session.name}")
    print(f"  ID:        {session.session_id}")
    print(f"  Модель:    {session.model}")
    print(f"  Статус:    {session.status}")
    print(f"  Сообщений: {info['messages_count']}")
    print(f"  Создана:   {session.created_at[:16].replace('T', ' ')}")
    print(f"  Обновлена: {session.updated_at[:16].replace('T', ' ')}")
    print(f"  {session.format_context_bar()}")
    if session.token_usage["total_tokens"] > 0:
        print(f"  Токены суммарно: {session.token_usage['total_tokens']:,}")
    cmp_status = "✅ включена" if cmp["enabled"] else "⭕ выключена"
    print(f"  Компрессия: {cmp_status}", end="")
    if cmp["compression_count"] > 0:
        print(
            f"  |  сжатий: {cmp['compression_count']}  "
            f"|  суммаризировано: {cmp['messages_summarized']}  "
            f"|  сэкономлено: ~{cmp['total_tokens_saved']:,} токенов"
        )
    else:
        print()
    print(f"{'─' * 55}\n")


def _print_compression_status(session: ConversationSession) -> None:
    """Выводит текущий статус и статистику компрессии для сессии."""
    cmp = session.get_compression_info()
    status = "✅ включена" if cmp["enabled"] else "⭕ выключена"
    print(f"\n{'─' * 55}")
    print(f"  Компрессия истории: {status}")
    if cmp["summarize_every"] is not None:
        print(f"  Суммаризация каждые: {cmp['summarize_every']} сообщений")
        print(f"  Живой хвост (keep_last_n): {cmp['keep_last_n']}")
    if cmp["compression_count"] > 0:
        print(f"  Выполнено сжатий:   {cmp['compression_count']}")
        print(f"  Суммаризировано:    {cmp['messages_summarized']} сообщений")
        print(f"  Токенов сэкономлено: ~{cmp['total_tokens_saved']:,}")
        if session.compressor:
            print(session.compressor.format_summary_preview())
    else:
        print("  Сжатий не выполнялось.")
    print(f"{'─' * 55}")
    print("  compress on  |  compress off  |  compress compare")
    print(f"{'─' * 55}\n")


def _print_compression_compare(session: ConversationSession) -> None:
    """Сравнивает токены с компрессией и без для текущей сессии."""
    if not session.compressor or not session.compressor.summaries:
        tokens_full = session.estimate_tokens()
        print(f"\n  Сжатий ещё не выполнялось.")
        print(f"  Токенов в текущей истории: {tokens_full:,}\n")
        return

    cmp_data = session.compressor.get_comparison(session.messages)
    savings_pct = cmp_data["savings_pct"]
    print(f"\n{'─' * 55}")
    print(f"  СРАВНЕНИЕ: токены с компрессией vs без")
    print(f"{'─' * 55}")
    print(f"  Без компрессии:  {cmp_data['full_tokens']:>8,} токенов  "
          f"({cmp_data['full_messages_count']} сообщений)")
    print(f"  Со сжатием:      {cmp_data['compressed_tokens']:>8,} токенов  "
          f"({cmp_data['compressed_messages_count']} сообщений)")
    print(f"{'─' * 55}")
    print(f"  Экономия:        {cmp_data['tokens_saved']:>8,} токенов  "
          f"({savings_pct:.1f}%)")
    print(f"  Сжатий выполнено: {cmp_data['compression_count']}")
    print(f"{'─' * 55}\n")


def _run_chat_loop(
    client,
    session: ConversationSession,
    session_storage: SessionStorage,
    tracker: Optional[UsageTracker] = None,
) -> None:
    """
    Интерактивный цикл диалога с сохранением истории.

    Каждое сообщение добавляется в session.messages и при каждом
    API-запросе передаётся вся история целиком — это механизм «памяти» модели.
    Если включена компрессия — старые сообщения заменяются LLM-summary
    каждые summarize_every сообщений, экономя токены.

    Внутренние команды:
        info               — показать использование контекста
        tokens             — показать таблицу роста токенов по ходу диалога
        compress           — показать статус компрессии
        compress on        — включить компрессию истории
        compress off       — выключить компрессию истории
        compress compare   — сравнить токены с/без компрессии
        close / q          — закрыть сессию и вернуться в главное меню
    """
    counter = TokenCounter(model=session.model)
    ctx_sizes = MODEL_CONTEXT_SIZES
    max_ctx = ctx_sizes.get(session.model, 128_000)
    dialog_tracker = DialogTokenTracker(model=session.model, max_tokens=max_ctx)

    method_label = counter.method_label
    cmp_label = "  🔄 компрессия ON" if session.compression_enabled else ""
    print(f"\n{'═' * 62}")
    print(f"  💬  {session.name}{cmp_label}")
    print(f"  Модель: {session.model}  |  ID: {session.session_id}")
    print(f"  Счётчик токенов: {method_label}")
    print(f"{'─' * 62}")
    print("  close/q — закрыть   |   info — статистика   |   compress — сжатие")
    print(f"{'═' * 62}\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            session_storage.save(session)
            break

        if not user_input:
            continue

        cmd_lower = user_input.lower()

        if cmd_lower in ("close", "exit", "q", "quit"):
            session.close()
            session_storage.save(session)
            info = session.get_context_info()
            print(f"\n✅ Сессия закрыта.")
            print(f"   Сообщений: {info['messages_count']}  |  {session.format_context_bar()}\n")
            break

        if cmd_lower == "info":
            _print_session_info(session)
            continue

        if cmd_lower == "tokens":
            print(f"\n  Рост токенов по ходу диалога  [{method_label}]:")
            print(dialog_tracker.format_growth_table())
            print()
            continue

        # ── Управление компрессией ────────────────────────────────
        if cmd_lower.startswith("compress"):
            parts = cmd_lower.split()
            sub = parts[1] if len(parts) > 1 else ""

            if sub == "on":
                session.enable_compression(summarize_every=10, keep_last_n=6)
                session_storage.save(session)
                print(
                    f"\n✅ Компрессия включена."
                    f" Суммаризация каждые 10 сообщений, хвост: 6.\n"
                )
            elif sub == "off":
                session.disable_compression()
                session_storage.save(session)
                print("\n⭕ Компрессия выключена. Накопленные summary сохранены.\n")
            elif sub == "compare":
                _print_compression_compare(session)
            else:
                # Без аргумента — показать статус и статистику
                _print_compression_status(session)
            continue
        # ─────────────────────────────────────────────────────────

        try:
            with RequestTimer() as t:
                _, pt, ct, token_cmp = _chat_send(client, session, user_input)

            # Компрессия: проверяем и запускаем если нужно
            if session.compression_enabled:
                cstats = session.maybe_compress(client)
                if cstats is not None:
                    print(f"\n🔄 {cstats.format()}")
                    if session.compressor:
                        print(session.compressor.format_summary_preview())
                    print()

            session_storage.save(session)

            msgs = session.get_messages_for_api()
            if len(msgs) >= 2:
                asst_msg = msgs[-1] if msgs[-1]["role"] == "assistant" else None
                dialog_tracker.add_turn("user", user_input, msgs[:-1] if asst_msg else msgs)
                if asst_msg:
                    dialog_tracker.add_turn("assistant", asst_msg.get("content", ""), msgs)

            cost = calculate_cost(pt, ct, session.model)
            tokens_exact = counter.is_exact
            if tracker is not None:
                tracker.record(
                    command="chat",
                    model=session.model,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    cost_usd=cost,
                    response_time_ms=t.elapsed_ms,
                    session_id=session.session_id,
                    tokens_estimated=not tokens_exact,
                )

            ctx_info = session.get_context_info()
            print(f"\n{session.format_context_bar()}")
            print(
                f"   prompt={pt:,} | completion={ct:,} | "
                f"${cost:.6f} | {t.elapsed_ms:.0f}мс"
            )

            cmp_line = _format_token_comparison(token_cmp)
            if cmp_line:
                print(cmp_line)
            print()

            if ctx_info["warning"]:
                print(
                    f"⚠️  Контекст заполнен на {ctx_info['percentage']:.1f}% "
                    f"— рассмотрите начало новой сессии (chat new)\n"
                )
        except Exception as e:
            if tracker is not None:
                tracker.record(
                    command="chat", model=session.model,
                    success=False, error=str(e), session_id=session.session_id,
                )
            print(f"\n❌ Ошибка: {e}\n")
            session_storage.save(session)


def _cmd_chat_list(session_storage: SessionStorage) -> None:
    """Выводит список всех сохранённых сессий."""
    sessions = session_storage.list_sessions()
    if not sessions:
        print("Нет сохранённых сессий.")
        return

    print(f"\n{'─' * 65}")
    print(f"  СЕССИИ ({len(sessions)})    ● — активная  ○ — закрытая")
    print(f"{'─' * 65}")
    for s in sessions:
        icon = "●" if s.get("status") == "active" else "○"
        date = s.get("updated_at", "")[:16].replace("T", " ")
        name = s.get("name", "Без имени")[:38]
        msgs = s.get("messages_count", 0)
        model_short = s.get("model", "?").split("/")[-1][:18]
        print(f"  [{s['session_id']}] {icon} {name}")
        print(f"           {date}  |  {msgs} сообщ.  |  {model_short}")
    print(f"{'─' * 65}\n")


def _cmd_chat_delete(session_id: str, session_storage: SessionStorage) -> None:
    """Удаляет сессию по ID."""
    if session_storage.delete(session_id):
        print(f"✅ Сессия {session_id} удалена.")
    else:
        print(f"❌ Сессия '{session_id}' не найдена.")


def _cmd_api_simple(clients: dict, current_model: str, tracker: UsageTracker) -> None:
    """
    Простой режим ручных API-запросов.

    Пользователь выбирает только модель и отправляет произвольные запросы.
    Вся статистика пишется в UsageTracker.

    Команды внутри:
        model <name>  — сменить модель
        models        — список моделей
        back / q      — выйти из режима
    """
    model = current_model
    all_models = get_available_models()

    print(f"\n{'═' * 58}")
    print(f"  API — Простой режим")
    print(f"  Модель: {all_models[model]['name']}  ({model})")
    print(f"{'─' * 58}")
    print("  model <name>  — сменить модель  |  models — список  |  back — выход")
    print(f"{'═' * 58}\n")

    while True:
        try:
            raw = input("api> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        cmd = raw.lower()
        if cmd in ("back", "exit", "q", "quit"):
            break

        if cmd == "models":
            for mid, info in all_models.items():
                marker = "★" if mid == model else " "
                avail = "✓" if info["provider"] in clients else "✗"
                print(f"  {avail}{marker} {info['name']:20} {mid}")
            continue

        if cmd.startswith("model "):
            name = raw[6:].strip()
            if name in all_models:
                info = all_models[name]
                if info["provider"] not in clients:
                    print(f"❌ Провайдер '{info['provider']}' недоступен")
                else:
                    model = name
                    print(f"✓ Модель: {info['name']}")
            else:
                print(f"❌ Модель '{name}' не найдена")
            continue

        client = get_client_for_model(model, clients)
        params = {
            "model": model,
            "messages": [{"role": "user", "content": raw}],
            "max_completion_tokens": 8000,
        }
        if model != "gpt-5-nano":
            params["temperature"] = 0.7

        pt_est = len(raw) // 4
        try:
            with RequestTimer() as t:
                response_text, _, _api_usage = stream_response(client, params)
            ct_est = len(response_text) // 4
            cost = calculate_cost(pt_est, ct_est, model)
            tracker.record(
                command="api_simple",
                model=model,
                prompt_tokens=pt_est,
                completion_tokens=ct_est,
                cost_usd=cost,
                response_time_ms=t.elapsed_ms,
                tokens_estimated=True,
            )
            print(f"  prompt ~{pt_est} | completion ~{ct_est} | ${cost:.6f} | {t.elapsed_ms:.0f}мс\n")
        except Exception as e:
            tracker.record(command="api_simple", model=model, success=False, error=str(e))
            print(f"\n❌ Ошибка: {e}\n")


def _cmd_api_advanced(clients: dict, current_model: str, tracker: UsageTracker) -> None:
    """
    Расширенный режим ручных API-запросов.

    Позволяет вручную настроить все параметры модели перед отправкой запроса.
    Набор параметров зависит от выбранной модели (MODEL_PARAMS_SCHEMA).

    Команды внутри:
        params              — показать текущие параметры
        set <param> <val>   — установить параметр
        reset               — сбросить параметры к значениям по умолчанию
        system <текст>      — задать системный промпт
        system clear        — очистить системный промпт
        model <name>        — сменить модель (параметры сбрасываются)
        models              — список моделей
        back / q            — выйти из режима
    """
    model = current_model
    all_models = get_available_models()
    params = _get_default_params(model)
    system_prompt: str = ""

    print(f"\n{'═' * 62}")
    print(f"  API — Расширенный режим")
    print(f"  Модель: {all_models[model]['name']}  ({model})")
    print(f"{'─' * 62}")
    print("  params | set <param> <val> | reset | system <текст> | model <name>")
    print(f"{'═' * 62}\n")
    _print_params(params, model)

    while True:
        try:
            raw = input("api-adv> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        cmd_lower = raw.lower()

        if cmd_lower in ("back", "exit", "q", "quit"):
            break

        if cmd_lower == "params":
            _print_params(params, model)
            if system_prompt:
                print(f"  system: {system_prompt[:80]}\n")
            continue

        if cmd_lower == "reset":
            params = _get_default_params(model)
            system_prompt = ""
            print("✓ Параметры сброшены к умолчаниям.")
            _print_params(params, model)
            continue

        if cmd_lower == "models":
            for mid, info in all_models.items():
                marker = "★" if mid == model else " "
                avail = "✓" if info["provider"] in clients else "✗"
                print(f"  {avail}{marker} {info['name']:20} {mid}")
            continue

        if cmd_lower.startswith("model "):
            name = raw[6:].strip()
            if name in all_models:
                info = all_models[name]
                if info["provider"] not in clients:
                    print(f"❌ Провайдер '{info['provider']}' недоступен")
                else:
                    model = name
                    params = _get_default_params(model)
                    system_prompt = ""
                    print(f"✓ Модель: {info['name']}  (параметры сброшены)")
                    _print_params(params, model)
            else:
                print(f"❌ Модель '{name}' не найдена")
            continue

        if cmd_lower.startswith("set "):
            parts = raw.split(maxsplit=2)
            if len(parts) < 3:
                print("Использование: set <параметр> <значение>")
                continue
            print(_set_param(params, model, parts[1], parts[2]))
            continue

        if cmd_lower.startswith("system"):
            text = raw[6:].strip()
            if text.lower() == "clear" or text == "":
                system_prompt = ""
                print("✓ Системный промпт очищен.")
            else:
                system_prompt = text
                print(f"✓ Системный промпт: {system_prompt[:60]}")
            continue

        client = get_client_for_model(model, clients)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": raw})

        api_params: dict = {"model": model, "messages": messages}
        for k, v in params.items():
            if k == "thinking":
                if v:
                    api_params.setdefault("extra_body", {})["thinking"] = {
                        "type": "enabled", "clear_thinking": False
                    }
            else:
                api_params[k] = v

        pt_est = sum(len(m["content"]) for m in messages) // 4
        try:
            with RequestTimer() as t:
                response_text, _, _api_usage = stream_response(client, api_params)
            ct_est = len(response_text) // 4
            cost = calculate_cost(pt_est, ct_est, model)
            tracker.record(
                command="api_advanced",
                model=model,
                prompt_tokens=pt_est,
                completion_tokens=ct_est,
                cost_usd=cost,
                response_time_ms=t.elapsed_ms,
                tokens_estimated=True,
            )
            print(f"  prompt ~{pt_est} | completion ~{ct_est} | ${cost:.6f} | {t.elapsed_ms:.0f}мс\n")
        except Exception as e:
            tracker.record(command="api_advanced", model=model, success=False, error=str(e))
            print(f"\n❌ Ошибка: {e}\n")


def _cmd_stats(tracker: UsageTracker, args: list) -> None:
    """
    Статистика использования API.

    stats                        — вся история
    stats today                  — только сегодня
    stats week                   — последние 7 дней
    stats month                  — текущий месяц
    stats YYYY-MM-DD             — конкретный день
    stats YYYY-MM-DD YYYY-MM-DD  — диапазон дат
    stats reset                  — сбросить журнал
    """
    if not args:
        print()
        print(tracker.format_summary())
        print()
        return

    first = args[0].lower()

    if first == "reset":
        confirm = input("Сбросить всю статистику? (yes/no): ").strip().lower()
        if confirm in ("yes", "да", "y"):
            tracker.stats_file.unlink(missing_ok=True)
            tracker._records.clear()
            print("✅ Статистика сброшена.")
        else:
            print("Отменено.")
        return

    second = args[1] if len(args) > 1 else None
    start, end = tracker.parse_period_shorthand(first, second)
    print()
    print(tracker.format_period_report(start, end))
    cost = tracker.get_cost_for_period(start, end)
    tokens = tracker.get_tokens_for_period(start, end)
    print(f"  💰 Расходы за период: ${cost:.6f}  ({tokens['total']:,} токенов)")
    print()


def stream_response(client, params, show_thinking=True):
    """
    Потоковый вывод ответа с поэтапной печатью и размышлениями.

    Поддерживает прерывание рассуждений:
        Ctrl+C (1-й раз)  — пропустить оставшееся рассуждение, дождаться ответа
        Ctrl+C (2-й раз)  — прервать ответ полностью и вернуть накопленное

    Returns:
        (full_response, reasoning_text, api_usage)
        api_usage — dict с prompt/completion/total_tokens от API,
                    или None если модель не вернула usage.
    """
    import signal

    params['stream'] = True

    model_name = params.get('model', '')
    if 'GLM' in model_name.upper():
        if 'extra_body' not in params:
            params['extra_body'] = {}
        params['extra_body']['thinking'] = {
            'type': 'enabled',
            'clear_thinking': False
        }

    params['stream_options'] = {"include_usage": True}

    full_response = ""
    reasoning_text = ""
    in_reasoning = False
    api_usage = None

    _flags = [False, False]  # [skip_reasoning, abort]
    _original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(sig, frame):
        if not _flags[0]:
            _flags[0] = True   # первый Ctrl+C: пропустить рассуждение
        else:
            _flags[1] = True   # второй Ctrl+C: прервать полностью
        signal.signal(signal.SIGINT, _sigint_handler)

    print("\n", flush=True)

    try:
        signal.signal(signal.SIGINT, _sigint_handler)
        stream = client.chat.completions.create(**params)

        for chunk in stream:
            if _flags[1]:
                print("\n\n⛔ Прервано\n", flush=True)
                break

            if not chunk.choices or len(chunk.choices) == 0:
                usage = getattr(chunk, 'usage', None)
                if usage is not None:
                    api_usage = {
                        "prompt_tokens":     getattr(usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(usage, 'completion_tokens', 0),
                        "total_tokens":      getattr(usage, 'total_tokens', 0),
                    }
                continue

            delta = chunk.choices[0].delta

            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_text += delta.reasoning_content

                if _flags[0]:
                    if in_reasoning:
                        in_reasoning = False
                        print("\n\n⏭ [Рассуждение пропущено → жду ответа...]", flush=True)
                    continue

                if not in_reasoning and show_thinking:
                    print("\n💭 [Размышление]  (Ctrl+C — пропустить)", flush=True)
                    in_reasoning = True
                if show_thinking:
                    print(delta.reasoning_content, end="", flush=True)

            if hasattr(delta, 'content') and delta.content:
                if in_reasoning:
                    if show_thinking:
                        print("\n\n📝 [Ответ]", flush=True)
                    in_reasoning = False
                elif _flags[0] and not full_response:
                    print("\n📝 [Ответ]", flush=True)

                content = delta.content
                full_response += content
                print(content, end="", flush=True)

        print("\n", flush=True)
        return full_response, reasoning_text, api_usage

    except Exception as e:
        print(f"\n❌ Ошибка при streaming: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        params['stream'] = False
        params.pop('stream_options', None)
        if 'extra_body' in params:
            del params['extra_body']
        response = client.chat.completions.create(**params)
        usage = getattr(response, 'usage', None)
        fallback_usage = None
        if usage:
            fallback_usage = {
                "prompt_tokens":     getattr(usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(usage, 'completion_tokens', 0),
                "total_tokens":      getattr(usage, 'total_tokens', 0),
            }
        return response.choices[0].message.content, "", fallback_usage

    finally:
        signal.signal(signal.SIGINT, _original_sigint)

def execute_mode(client, mode, user_input, max_retries=3):
    # Режим 6: метапромптинг с двумя этапами
    if mode.get('metadata', {}).get('two_stage'):
        return execute_meta_prompting(client, mode, user_input, max_retries)
    
    # Адаптируем параметры под модель
    model_name = mode['params']['model']
    params = get_model_params(model_name, mode['params'])
    messages = [{"role": "user", "content": user_input}]
    
    if 'system_prompt' in mode:
        messages.insert(0, {"role": "system", "content": mode['system_prompt']})
    
    params['messages'] = messages
    
    # Retry механизм для нестабильных API
    for attempt in range(max_retries):
        try:
            print("🤔 Размышляю...", flush=True)
            
            # Используем streaming для постепенного вывода
            answer_text, reasoning_text, _api_usage = stream_response(client, params.copy())
            
            # Создаём объект ответа для совместимости
            # Для gpt-5-nano не делаем второй запрос (игнорирует max_completion_tokens)
            model_name = mode['params']['model']
            if model_name == "gpt-5-nano":
                # Оцениваем токены на основе длины ответа
                estimated_prompt_tokens = sum(len(m['content']) for m in messages) // 4
                estimated_completion_tokens = (len(answer_text) + len(reasoning_text)) // 4
                
                class MockResponse:
                    def __init__(self, content, usage, reasoning=""):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': content,
                                'reasoning_content': reasoning
                            })()
                        })()]
                        self.usage = usage
                
                mock_usage = type('obj', (object,), {
                    'prompt_tokens': estimated_prompt_tokens,
                    'completion_tokens': estimated_completion_tokens,
                    'total_tokens': estimated_prompt_tokens + estimated_completion_tokens
                })()
                
                return MockResponse(answer_text, mock_usage, reasoning_text)
            
            # Для других моделей делаем запрос для статистики
            params_for_usage = params.copy()
            params_for_usage['max_completion_tokens'] = 1  # Минимальный запрос для статистики
            if 'extra_body' in params_for_usage:
                del params_for_usage['extra_body']  # Убираем thinking для статистики
            usage_response = client.chat.completions.create(**params_for_usage)
            
            # Создаём mock объект с ответом
            class MockResponse:
                def __init__(self, content, usage, reasoning=""):
                    self.choices = [type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': content,
                            'reasoning_content': reasoning
                        })()
                    })()]
                    self.usage = usage
            
            # Примерная оценка токенов (1 токен ≈ 4 символа)
            estimated_tokens = (len(answer_text) + len(reasoning_text)) // 4
            mock_usage = type('obj', (object,), {
                'prompt_tokens': usage_response.usage.prompt_tokens,
                'completion_tokens': estimated_tokens,
                'total_tokens': usage_response.usage.prompt_tokens + estimated_tokens
            })()
            
            return MockResponse(answer_text, mock_usage, reasoning_text)
            
        except Exception as e:
            print("\r", end="", flush=True)
            if attempt < max_retries - 1:
                print(f"⚠️ Попытка {attempt + 1} не удалась, повтор через 2 сек...", flush=True)
                import time
                time.sleep(2)
            else:
                raise

def execute_meta_prompting(client, mode, user_input, max_retries=3):
    model_name = mode['params']['model']
    params = get_model_params(model_name, mode['params'])
    
    # Этап 1: Определение роли и навыков
    meta_prompt = f"""Проанализируй следующий вопрос и определи:
1. Какая роль/профессия лучше всего ответит на этот вопрос?
2. Какая предметная область затрагивается?
3. Какие ключевые навыки и знания нужны для ответа?

Вопрос: {user_input}

Ответь кратко в формате:
РОЛЬ: [роль]
ОБЛАСТЬ: [предметная область]
НАВЫКИ: [список навыков через запятую]"""
    
    params['messages'] = [{"role": "user", "content": meta_prompt}]
    
    # Retry для этапа 1
    for attempt in range(max_retries):
        try:
            print("🤔 Этап 1: Анализирую задачу...", flush=True)
            meta_response = client.chat.completions.create(**params)
            print("\r✓ Этап 1 завершен!        ", flush=True)
            break
        except Exception as e:
            print("\r", end="", flush=True)
            if attempt < max_retries - 1:
                print(f"⚠️ Этап 1: Попытка {attempt + 1} не удалась, повтор через 2 сек...", flush=True)
                import time
                time.sleep(2)
            else:
                raise
    
    meta_analysis = meta_response.choices[0].message.content
    
    # Этап 2: Ответ с учетом определенной роли
    enriched_prompt = f"""Ты - эксперт со следующими характеристиками:

{meta_analysis}

Используя эти знания и навыки, ответь на вопрос максимально качественно и профессионально:

{user_input}"""
    
    params['messages'] = [{"role": "user", "content": enriched_prompt}]
    
    # Retry для этапа 2
    for attempt in range(max_retries):
        try:
            print("🤔 Этап 2: Формирую экспертный ответ...", flush=True)
            final_response = client.chat.completions.create(**params)
            print("\r✓ Этап 2 завершен!              ", flush=True)
            break
        except Exception as e:
            print("\r", end="", flush=True)
            if attempt < max_retries - 1:
                print(f"⚠️ Этап 2: Попытка {attempt + 1} не удалась, повтор через 2 сек...", flush=True)
                import time
                time.sleep(2)
            else:
                raise
    
    # Объединяем usage из обоих запросов
    combined_usage = type('obj', (object,), {
        'prompt_tokens': meta_response.usage.prompt_tokens + final_response.usage.prompt_tokens,
        'completion_tokens': meta_response.usage.completion_tokens + final_response.usage.completion_tokens,
        'total_tokens': meta_response.usage.total_tokens + final_response.usage.total_tokens
    })
    
    # Добавляем метаанализ в начало ответа
    enriched_content = f"[МЕТААНАЛИЗ]\n{meta_analysis}\n\n[ОТВЕТ]\n{final_response.choices[0].message.content}"
    
    # Создаем объект ответа с обогащенным контентом
    final_response.choices[0].message.content = enriched_content
    final_response.usage = combined_usage
    
    return final_response

def compare_formatting_modes(client, user_input, log_file, model_name="zai-org/GLM-4.7-Flash"):
    print("\n" + "=" * 70)
    print("РЕЖИМ СРАВНЕНИЯ: отправка одного запроса с разными параметрами")
    print(f"Используемая модель: {get_available_models()[model_name]['name']}")
    print("=" * 70)
    
    modes = get_available_modes(model_name)
    
    results = []
    
    for mode in modes:
        print(f"\n{'─' * 70}")
        print(f"Режим {mode['id']}: {mode['name']}")
        if 'system_prompt' in mode:
            print(f"Системный промпт: {mode['system_prompt'][:60]}...")
        print(f"{'─' * 70}")
        
        try:
            response = execute_mode(client, mode, user_input)
            assistant_message = response.choices[0].message.content
            usage = response.usage
            
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            word_count = count_words(assistant_message)
            
            cost = calculate_cost(prompt_tokens, completion_tokens, mode['params']['model'])
            
            usage_info = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "word_count": word_count,
                "cost": cost
            }
            
            print(f"\nОтвет: {assistant_message}")
            print("\n[Конец ответа]")
            print(f"\nСтатистика:")
            print(f"  Токены: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
            print(f"  Слов: {word_count}")
            print(f"  Символов: {len(assistant_message)}")
            print(f"  Стоимость: ${cost:.6f}")
            
            log_interaction(log_file, user_input, assistant_message, usage_info, mode['metadata'])
            
            results.append({
                "mode": mode['name'],
                "response": assistant_message,
                "tokens": total_tokens,
                "words": word_count,
                "chars": len(assistant_message),
                "cost": cost
            })
            
        except Exception as e:
            print(f"\nОшибка в режиме '{mode['name']}': {str(e)}")
    
    print("\n" + "=" * 70)
    print("СВОДКА СРАВНЕНИЯ")
    print("=" * 70)
    
    for result in results:
        print(f"\n{result['mode']}:")
        print(f"  Символов: {result['chars']}")
        print(f"  Слов: {result['words']}")
        print(f"  Токенов: {result['tokens']}")
        print(f"  Стоимость: ${result['cost']:.6f}")
    
    if results:
        print("\n" + "─" * 70)
        print("АНАЛИЗ:")
        min_cost = min(r['cost'] for r in results)
        max_cost = max(r['cost'] for r in results)
        min_words = min(r['words'] for r in results)
        max_words = max(r['words'] for r in results)
        print(f"  Разброс стоимости: ${min_cost:.6f} - ${max_cost:.6f}")
        print(f"  Разброс по словам: {min_words} - {max_words}")
        print(f"  Экономия при выборе самого дешевого: ${max_cost - min_cost:.6f}")
    
    return results

def compare_reasoning_approaches(client, user_input, log_file, model_name="zai-org/GLM-4.7-Flash"):
    print("\n" + "=" * 70)
    print("ДЕНЬ 3: СРАВНЕНИЕ СПОСОБОВ РАССУЖДЕНИЯ")
    print(f"Используемая модель: {get_available_models()[model_name]['name']}")
    print(f"Задача: {user_input}")
    print("=" * 70)
    
    results = []
    
    # Способ 1: Прямой ответ (без дополнительных инструкций)
    print(f"\n{'─' * 70}")
    print("СПОСОБ 1: Прямой ответ (без дополнительных инструкций)")
    print(f"{'─' * 70}")
    
    try:
        base_params = {"model": model_name, "messages": [{"role": "user", "content": user_input}], "max_completion_tokens": 120000, "temperature": 0.7}
        params = get_model_params(model_name, base_params)
        print("\n🤔 Размышляю...", flush=True)
        
        # Используем streaming для показа размышлений
        answer1, reasoning1, _u1 = stream_response(client, params.copy())
        
        # Оценка токенов
        estimated_tokens = (len(answer1) + len(reasoning1)) // 4
        usage1 = type('obj', (object,), {
            'total_tokens': estimated_tokens
        })()
        
        print(f"\n\n[Конец ответа]")
        print(f"Ответ: {len(answer1)} символов")
        print("\n[Конец ответа]")
        print(f"\nСтатистика: Токены: {usage1.total_tokens} | Слова: {count_words(answer1)}")
        
        results.append({
            "method": "Прямой ответ",
            "response": answer1,
            "tokens": usage1.total_tokens,
            "words": count_words(answer1),
            "chars": len(answer1)
        })
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
    
    # Способ 2: С инструкцией "решай пошагово"
    print(f"\n{'─' * 70}")
    print("СПОСОБ 2: С инструкцией 'решай пошагово'")
    print(f"{'─' * 70}")
    
    try:
        step_by_step_prompt = f"{user_input}\n\nРешай пошагово."
        base_params = {"model": model_name, "messages": [{"role": "user", "content": step_by_step_prompt}], "max_completion_tokens": 120000, "temperature": 0.7}
        params = get_model_params(model_name, base_params)
        print("\n🤔 Размышляю пошагово...", flush=True)
        
        # Используем streaming для показа размышлений
        answer2, reasoning2, _u2 = stream_response(client, params.copy())
        
        # Оценка токенов
        estimated_tokens = (len(answer2) + len(reasoning2)) // 4
        usage2 = type('obj', (object,), {
            'total_tokens': estimated_tokens
        })()
        
        print(f"\n\n[Конец ответа]")
        print(f"Ответ: {len(answer2)} символов")
        print("\n[Конец ответа]")
        print(f"\nСтатистика: Токены: {usage2.total_tokens} | Слова: {count_words(answer2)}")
        
        results.append({
            "method": "Пошаговое решение",
            "response": answer2,
            "tokens": usage2.total_tokens,
            "words": count_words(answer2),
            "chars": len(answer2)
        })
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
    
    # Способ 3: Метапромпт (сначала создаем промпт, потом решаем)
    print(f"\n{'─' * 70}")
    print("СПОСОБ 3: Метапромпт (сначала создание промпта, затем решение)")
    print(f"{'─' * 70}")
    
    try:
        meta_request = f"Задача: {user_input}\n\nСоставь оптимальный промпт для решения этой задачи. Выведи только промпт, без дополнительных объяснений."
        base_params = {"model": model_name, "messages": [{"role": "user", "content": meta_request}], "max_completion_tokens": 120000, "temperature": 0.7}
        params = get_model_params(model_name, base_params)
        print("\n🤔 Создаю промпт...", flush=True)
        
        # Streaming для создания промпта
        generated_prompt, reasoning_meta, _um = stream_response(client, params.copy())
        
        print(f"\n\nСгенерированный промпт: {generated_prompt}")
        print(f"\n{'·' * 70}")
        
        base_params = {"model": model_name, "messages": [{"role": "user", "content": generated_prompt}], "max_completion_tokens": 120000, "temperature": 0.7}
        params = get_model_params(model_name, base_params)
        print("\n🤔 Решаю по промпту...", flush=True)
        
        # Streaming для решения по промпту
        answer3, reasoning3, _u3 = stream_response(client, params.copy())
        
        # Оценка токенов
        estimated_tokens = (len(generated_prompt) + len(reasoning_meta) + len(answer3) + len(reasoning3)) // 4
        usage3_combined = type('obj', (object,), {
            'total_tokens': estimated_tokens
        })
        
        print(f"\n\n[Конец ответа]")
        print(f"Ответ: {len(answer3)} символов")
        print("\n[Конец ответа]")
        print(f"\nСтатистика: Токены: {usage3_combined.total_tokens} | Слова: {count_words(answer3)}")
        
        results.append({
            "method": "Метапромпт",
            "response": f"[Промпт: {generated_prompt}]\n\n{answer3}",
            "tokens": usage3_combined.total_tokens,
            "words": count_words(answer3),
            "chars": len(answer3)
        })
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
    
    # Способ 4: Группа экспертов
    print(f"\n{'─' * 70}")
    print("СПОСОБ 4: Группа экспертов (аналитик, инженер, критик)")
    print(f"{'─' * 70}")
    
    try:
        experts = [
            ("Аналитик", "Ты - аналитик. Проанализируй задачу и предложи решение с точки зрения анализа данных и логики."),
            ("Инженер", "Ты - инженер. Реши задачу с технической и практической точки зрения."),
            ("Критик", "Ты - критик. Оцени задачу критически, найди возможные проблемы и предложи решение.")
        ]
        
        expert_answers = []
        total_tokens_experts = 0
        
        for expert_name, expert_role in experts:
            print(f"\n[{expert_name}]", flush=True)
            messages = [
                {"role": "system", "content": expert_role},
                {"role": "user", "content": user_input}
            ]
            base_params = {"model": model_name, "messages": messages, "max_completion_tokens": 120000, "temperature": 0.7}
            params = get_model_params(model_name, base_params)
            print(f"🤔 {expert_name} размышляет...", flush=True)
            
            # Используем streaming для показа размышлений эксперта
            expert_answer, expert_reasoning, _ue = stream_response(client, params.copy())
            
            # Оценка токенов
            estimated_tokens = (len(expert_answer) + len(expert_reasoning)) // 4
            total_tokens_experts += estimated_tokens
            
            print(f"\n(Токены: ~{estimated_tokens})", flush=True)
            expert_answers.append(f"[{expert_name}]: {expert_answer}")
        
        combined_answer = "\n\n".join(expert_answers)
        
        print("\n[Конец ответа]")
        print(f"\nСтатистика: Токены: {total_tokens_experts} | Слова: {count_words(combined_answer)}")
        
        results.append({
            "method": "Группа экспертов",
            "response": combined_answer,
            "tokens": total_tokens_experts,
            "words": count_words(combined_answer),
            "chars": len(combined_answer)
        })
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
    
    # Сводка сравнения
    print("\n" + "=" * 70)
    print("СВОДКА СРАВНЕНИЯ СПОСОБОВ РАССУЖДЕНИЯ")
    print("=" * 70)
    
    for result in results:
        print(f"\n{result['method']}:")
        print(f"  Символов: {result['chars']}")
        print(f"  Слов: {result['words']}")
        print(f"  Токенов: {result['tokens']}")
    
    if results:
        print("\n" + "─" * 70)
        print("АНАЛИЗ:")
        min_tokens = min(r['tokens'] for r in results)
        max_tokens = max(r['tokens'] for r in results)
        min_words = min(r['words'] for r in results)
        max_words = max(r['words'] for r in results)
        print(f"  Разброс по токенам: {min_tokens} - {max_tokens}")
        print(f"  Разброс по словам: {min_words} - {max_words}")
        print(f"  Самый краткий: {min(results, key=lambda x: x['words'])['method']}")
        print(f"  Самый подробный: {max(results, key=lambda x: x['words'])['method']}")
    
    return results

def compare_temperatures(client, user_input, log_file, model_name="zai-org/GLM-4.7-Flash"):
    """Сравнение влияния температуры на ответы модели"""
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ ВЛИЯНИЯ ТЕМПЕРАТУРЫ НА ОТВЕТЫ")
    print(f"Используемая модель: {get_available_models()[model_name]['name']}")
    print(f"Задача: {user_input}")
    print("=" * 70)
    
    # Проверка поддержки температуры
    if model_name == "gpt-5-nano":
        print("\n⚠️ ВНИМАНИЕ: Модель gpt-5-nano НЕ поддерживает параметр temperature!")
        print("Переключитесь на GPT-5.4, GLM-4.7-Flash или GLM-4.7 для сравнения температур.")
        print("\nИспользуйте команду: model gpt-5.4")
        return []
    
    # Для OpenAI моделей используем диапазон 0-1 с 4 точками
    # Для GLM моделей используем расширенный диапазон 0-1.2 с 3 точками
    models = get_available_models()
    provider = models[model_name]["provider"]
    
    if provider == "openai":
        temperatures = [0, 0.5, 0.75, 1.0]
        print("\n📊 Диапазон температуры для OpenAI: 0-1 (4 точки измерения)")
    else:
        temperatures = [0, 0.5, 0.75, 1.0]
        print("\n📊 Диапазон температуры: 0-1 (4 точки измерения)")
    
    results = []
    
    for temp in temperatures:
        print(f"\n{'─' * 70}")
        print(f"TEMPERATURE = {temp}")
        print(f"{'─' * 70}")
        
        # Описание характеристик температуры
        if temp == 0:
            print("Характеристики: Детерминированность, точность, повторяемость")
        elif temp == 0.5:
            print("Характеристики: Умеренная вариативность, надежность")
        elif temp == 0.75:
            print("Характеристики: Сбалансированная креативность и точность")
        elif temp == 1.0:
            print("Характеристики: Максимальная креативность в безопасном диапазоне")
        
        try:
            base_params = {
                "model": model_name,
                "messages": [{"role": "user", "content": user_input}],
                "max_completion_tokens": 120000,
                "temperature": temp
            }
            params = get_model_params(model_name, base_params)
            
            print("\n🤔 Генерирую ответ...", flush=True)
            answer, reasoning, _ut = stream_response(client, params.copy())
            
            # Оценка токенов
            estimated_tokens = (len(answer) + len(reasoning)) // 4
            word_count = count_words(answer)
            
            print(f"\n\n[Конец ответа]")
            print(f"\nСтатистика:")
            print(f"  Символов: {len(answer)}")
            print(f"  Слов: {word_count}")
            print(f"  Токенов (примерно): {estimated_tokens}")
            
            # Логирование
            usage_info = {
                "prompt_tokens": len(user_input) // 4,
                "completion_tokens": estimated_tokens,
                "total_tokens": len(user_input) // 4 + estimated_tokens,
                "word_count": word_count,
                "cost": calculate_cost(len(user_input) // 4, estimated_tokens, model_name)
            }
            
            metadata = {
                "experiment": "temperature_comparison",
                "temperature": temp
            }
            
            log_interaction(log_file, user_input, answer, usage_info, metadata)
            
            results.append({
                "temperature": temp,
                "response": answer,
                "chars": len(answer),
                "words": word_count,
                "tokens": estimated_tokens
            })
            
        except Exception as e:
            print(f"\n❌ Ошибка при temperature={temp}: {str(e)}")
    
    # Сводка и анализ
    print("\n" + "=" * 70)
    print("СВОДКА СРАВНЕНИЯ ТЕМПЕРАТУР")
    print("=" * 70)
    
    for result in results:
        print(f"\nTemperature = {result['temperature']}:")
        print(f"  Символов: {result['chars']}")
        print(f"  Слов: {result['words']}")
        print(f"  Токенов: {result['tokens']}")
    
    if results:
        print("\n" + "─" * 70)
        print("АНАЛИЗ И ВЫВОДЫ")
        print("─" * 70)
        
        # Статистика по метрикам
        chars_data = [r['chars'] for r in results]
        words_data = [r['words'] for r in results]
        
        print(f"\n📊 Разброс метрик:")
        print(f"  Символы: {min(chars_data)} - {max(chars_data)} (разница: {max(chars_data) - min(chars_data)})")
        print(f"  Слова: {min(words_data)} - {max(words_data)} (разница: {max(words_data) - min(words_data)})")
        
        # Выводы и рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ:")
        print(f"\n🎯 Temperature = 0 (Детерминированная):")
        print(f"   ✓ Задачи требующие точности и повторяемости")
        print(f"   ✓ Технические объяснения, документация")
        print(f"   ✓ Извлечение фактов, классификация")
        print(f"   ✓ Когда нужен один правильный ответ")
        
        print(f"\n📐 Temperature = 0.5 (Умеренная):")
        print(f"   ✓ Задачи требующие точности с небольшой вариативностью")
        print(f"   ✓ Деловая переписка, официальные документы")
        print(f"   ✓ Ответы на FAQ, база знаний")
        print(f"   ✓ Когда нужна стабильность, но не абсолютная")
        
        print(f"\n⚖️ Temperature = 0.75 (Сбалансированная):")
        print(f"   ✓ Универсальный вариант для большинства задач")
        print(f"   ✓ Диалоги, консультации, вопросы-ответы")
        print(f"   ✓ Объяснения с примерами и аналогиями")
        print(f"   ✓ Когда нужен баланс точности и естественности")
        
        print(f"\n🎨 Temperature = 1.0 (Креативная):")
        print(f"   ✓ Творческие задачи (рассказы, идеи, концепции)")
        print(f"   ✓ Brainstorming, генерация альтернатив")
        print(f"   ✓ Маркетинговые тексты, контент-маркетинг")
        print(f"   ✓ Когда нужно разнообразие и оригинальность")
        
        print(f"\n⚠️ ВАЖНО:")
        print(f"   • Диапазон 0-1 обеспечивает стабильное качество")
        print(f"   • Низкая температура (0-0.3) для критичных задач")
        print(f"   • Средняя температура (0.5-0.75) для общих задач")
        print(f"   • Высокая температура (0.8-1.0) для креативных задач")
    
    return results

def interactive_mode_selection(client, user_input, log_file, mode_id, model_name="zai-org/GLM-4.7-Flash"):
    modes = get_available_modes(model_name)
    selected_mode = None
    
    for mode in modes:
        if mode['id'] == mode_id:
            selected_mode = mode
            break
    
    if not selected_mode:
        print(f"Ошибка: Режим {mode_id} не найден")
        return
    
    print(f"\nИспользуется режим: {selected_mode['name']}")
    if 'system_prompt' in selected_mode:
        print(f"Системный промпт: {selected_mode['system_prompt']}")
    
    try:
        print(f"\nОтправка запроса к API...")
        print(f"Модель: {selected_mode['params']['model']}")
        response = execute_mode(client, selected_mode, user_input)
        print(f"Ответ получен!")
        assistant_message = response.choices[0].message.content
        usage = response.usage
        
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        word_count = count_words(assistant_message)
        
        cost = calculate_cost(prompt_tokens, completion_tokens, selected_mode['params']['model'])
        
        usage_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "word_count": word_count,
            "cost": cost,
            "model": selected_mode['params']['model']
        }
        
        log_interaction(log_file, user_input, assistant_message, usage_info, selected_mode['metadata'])
        
        print(f"\nAssistant: {assistant_message}")
        print(f"\n[Режим: {selected_mode['name']} | Токены: {total_tokens} | Слова: {word_count} | Стоимость: ${cost:.6f}]")
        
        return usage_info
        
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении запроса:")
        print(f"Тип ошибки: {type(e).__name__}")
        print(f"Сообщение: {str(e)}")
        import traceback
        print(f"\nПолный traceback:")
        traceback.print_exc()
        return None

def _cmd_generate(
    topic: str,
    post_type: Optional[str],
    client,
    model: str,
    storage: PostStorage,
) -> None:
    """Генерирует пост через pipeline и сохраняет."""
    pipeline = NewsPipeline(client=client, model=model, verbose=True)
    try:
        result = pipeline.run(topic=topic, post_type=post_type)
    except Exception as e:
        print(f"\n❌ Ошибка pipeline: {e}")
        import traceback; traceback.print_exc()
        return

    result["model"] = model
    post_id = storage.save(result)

    print(f"\n✅ Пост сохранён  ID: {post_id}")
    print(f"   Файл: posts/{post_id}/post.md")

    seo = result.get("seo", {})
    best_headline = (seo.get("headline_variants") or [{}])[0].get("text", "")
    if best_headline:
        print(f"   Заголовок: {best_headline}")

    meta = seo.get("meta_description", "")
    if meta:
        print(f"   Meta: {meta[:80]}...")


def _cmd_agent(
    topic: str,
    client,
    model: str,
    storage: PostStorage,
    tracker: Optional[UsageTracker] = None,
) -> None:
    """Запускает автономный ReAct-агент."""
    agent = AgentLoop(client=client, model=model, storage=storage, verbose=True, tracker=tracker)
    try:
        with RequestTimer() as t:
            result = agent.run(task=topic)
    except Exception as e:
        if tracker is not None:
            tracker.record(command="agent", model=model, success=False, error=str(e))
        print(f"\n❌ Ошибка агента: {e}")
        import traceback; traceback.print_exc()
        return

    post_id = result.get("saved_post_id")
    iterations = result.get("iterations", "?")
    tu = result.get("token_usage", {})
    tokens = tu.get("total_tokens", "?")

    print(f"\n✅ Агент завершил работу")
    print(f"   Итераций: {iterations}  |  Токенов: {tokens}  |  {t.elapsed_ms:.0f}мс")
    if post_id:
        print(f"   Сохранён пост ID: {post_id}  →  posts/{post_id}/post.md")
    print(f"\n{result['final_post']}")


def _cmd_batch(filepath: str, client, model: str, storage: PostStorage) -> None:
    """Пакетная генерация из файла (одна тема = одна строка)."""
    path = Path(filepath)
    if not path.exists():
        print(f"❌ Файл не найден: {filepath}")
        return

    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"📋 Найдено тем: {len(lines)}")

    pipeline = NewsPipeline(client=client, model=model, verbose=False)
    success, failed = 0, 0

    for i, topic in enumerate(lines, 1):
        print(f"\n[{i}/{len(lines)}] {topic[:70]}")
        try:
            result = pipeline.run(topic=topic)
            result["model"] = model
            post_id = storage.save(result)
            print(f"  ✓ ID: {post_id}  слов: {len(result['post'].split())}")
            success += 1
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            failed += 1

    print(f"\n📊 Итого: {success} успешно, {failed} с ошибками")


def _cmd_history(n: int, storage: PostStorage) -> None:
    """Показывает последние n постов."""
    posts = storage.list_posts(n=n)
    if not posts:
        print("История пуста.")
        return
    print(f"\n{'─' * 65}")
    print(f"  ПОСЛЕДНИЕ ПОСТЫ ({len(posts)})")
    print(f"{'─' * 65}")
    for p in posts:
        date = p["created_at"][:10]
        tags = ", ".join(p.get("tags", [])[:3]) or "—"
        print(f"  [{p['id']}]  {date}  [{p['post_type']:8}]  {p['title'][:45]}")
        print(f"           слов: {p.get('word_count', '?')}  теги: {tags}")
    print(f"{'─' * 65}")


def _cmd_export(post_id: str, fmt: str, storage: PostStorage) -> None:
    """Экспортирует пост в указанном формате."""
    content = storage.export_post(post_id, fmt=fmt)
    if content is None:
        print(f"❌ Пост с ID '{post_id}' не найден")
        return
    out_file = Path("posts") / post_id / f"post.{fmt}"
    out_file.write_text(content, encoding="utf-8")
    print(f"✅ Экспортировано: {out_file}")
    print(f"\n--- Предпросмотр ({fmt}) ---")
    print(content[:600])
    if len(content) > 600:
        print(f"\n... ({len(content)} символов, полный файл: {out_file})")


def _cmd_template(args: list) -> None:
    """Управление шаблонами типов постов."""
    sub = args[0] if args else "list"

    if sub == "list":
        print(f"\n{'─' * 50}")
        print("  ТИПЫ ПОСТОВ")
        print(f"{'─' * 50}")
        for key, guide in POST_TYPE_GUIDES.items():
            wmin, wmax = guide["word_count"]
            print(f"  {key:12} — {guide['description']} ({wmin}-{wmax} слов, тон: {guide['tone']})")
        print(f"{'─' * 50}")

    elif sub == "show" and len(args) >= 2:
        key = args[1]
        if key not in POST_TYPE_GUIDES:
            print(f"❌ Тип '{key}' не найден. Доступны: {', '.join(POST_TYPE_GUIDES)}")
            return
        guide = POST_TYPE_GUIDES[key]
        print(f"\n[{key}] {guide['description']}")
        print(f"  Объём:     {guide['word_count'][0]}-{guide['word_count'][1]} слов")
        print(f"  Тон:       {guide['tone']}")
        print(f"  Структура: {' → '.join(guide['structure'])}")
    else:
        print("Использование: template list | template show <тип>")


def _print_help() -> None:
    print("""
╔═══════════════════════════════════════════════════════════════╗
║         NEWS AGENT + CHAT CLI  —  Команды                    ║
╠═══════════════════════════════════════════════════════════════╣
║  ── ГЕНЕРАЦИЯ ПОСТОВ ────────────────────────────────────────║
║  generate <тема>              Создать пост (pipeline)        ║
║  generate -t <тип> <тема>     С указанием типа               ║
║    типы: breaking, analysis, digest, social, press           ║
║  agent <тема>                 Автономный ReAct-агент         ║
║  batch <файл>                 Пакетная генерация из файла    ║
║  history [n]                  Последние n постов (def 10)    ║
║  export <id> <формат>         Экспорт поста                  ║
║    форматы: md, html, telegram, json, plain                  ║
║  template list/show <тип>     Типы постов                    ║
║                                                               ║
║  ── ДИАЛОГОВЫЙ ЧАТ ─────────────────────────────────────────║
║  chat                         Начать новую сессию            ║
║  chat new [имя]               Именованная сессия             ║
║  chat list                    Список сессий                  ║
║  chat load <id>               Загрузить и продолжить         ║
║  chat resume <id>             Продолжить незаконченную       ║
║  chat delete <id>             Удалить сессию                 ║
║  chat info [id]               Инфо + использование контекста ║
║  (внутри чата) info           Контекст текущей сессии        ║
║  (внутри чата) tokens         Таблица роста токенов          ║
║  (внутри чата) compress       Статус компрессии              ║
║  (внутри чата) compress on    Включить компрессию истории    ║
║  (внутри чата) compress off   Выключить компрессию           ║
║  (внутри чата) compress compare Сравнить токены с/без сжатия ║
║  (внутри чата) close/q        Закрыть сессию                 ║
║                                                               ║
║  ── РУЧНЫЕ API-ЗАПРОСЫ ──────────────────────────────────────║
║  api                          Простой режим (выбор модели)   ║
║  api advanced                 Расширенный режим:             ║
║    (внутри) params            Текущие параметры              ║
║    (внутри) set <p> <v>       Установить параметр            ║
║    (внутри) reset             Сбросить к умолчаниям          ║
║    (внутри) system <текст>    Задать системный промпт        ║
║    (внутри) model <name>      Сменить модель                 ║
║    (внутри) back              Выйти из режима                ║
║                                                               ║
║  ── СТАТИСТИКА ─────────────────────────────────────────────║
║  stats                        Накопленная статистика         ║
║  stats reset                  Сбросить статистику            ║
║                                                               ║
║  ── ОБЩЕЕ ──────────────────────────────────────────────────║
║  models                       Список моделей                 ║
║  model <name>                 Переключить модель             ║
║  help / ?                     Эта справка                    ║
║  quit / exit / q              Выход                          ║
╚═══════════════════════════════════════════════════════════════╝""")


def main():
    cloud_api_key = os.getenv("CLOUD_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not cloud_api_key and not openai_api_key:
        print("❌ Ни CLOUD_API_KEY, ни OPENAI_API_KEY не найдены в .env")
        return

    clients = {}
    if cloud_api_key:
        clients["cloud_ru"] = OpenAI(
            api_key=cloud_api_key,
            base_url="https://foundation-models.api.cloud.ru/v1",
            timeout=120.0,
        )
    if openai_api_key:
        clients["openai"] = OpenAI(api_key=openai_api_key, timeout=120.0)

    storage = PostStorage(base_dir="posts")
    session_storage = SessionStorage("sessions")
    tracker = UsageTracker(logs_dir="logs")
    current_model = "zai-org/GLM-4.7-Flash" if "cloud_ru" in clients else "gpt-5-nano"

    s = tracker.get_summary()
    print("╔══════════════════════════════════════╗")
    print("║     NEWS AGENT + CHAT  v3.0          ║")
    print("║  AI-агент, чат, ручные API-запросы   ║")
    print("╚══════════════════════════════════════╝")
    print(f"  Модель:      {get_available_models()[current_model]['name']}")
    print(f"  Посты:       posts/  |  Сессии:  sessions/")
    print(f"  Всего запросов: {s['total_requests']}  |  Токенов: {s['total_tokens']:,}  |  ${s['total_cost_usd']:.4f}")
    print("  'help' — справка  |  'api' — запросы  |  'stats' — статистика")
    print()

    session_posts = 0

    while True:
        try:
            raw = input("news-agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            print(f"\n📊 Сессия: создано постов — {session_posts}")
            print("До свидания!")
            break

        elif cmd in ("help", "?"):
            _print_help()

        elif cmd == "models":
            models = get_available_models()
            print(f"\n{'─' * 55}")
            for mid, info in models.items():
                marker = "★" if mid == current_model else " "
                avail = "✓" if info["provider"] in clients else "✗"
                print(f" {avail}{marker} {info['name']:20} {mid}")
            print(f"{'─' * 55}")

        elif cmd == "model" and len(parts) >= 2:
            name = " ".join(parts[1:])
            if name in get_available_models():
                info = get_available_models()[name]
                if info["provider"] not in clients:
                    print(f"❌ Провайдер '{info['provider']}' недоступен (нет ключа API)")
                else:
                    current_model = name
                    print(f"✓ Модель: {info['name']}")
            else:
                print(f"❌ Модель '{name}' не найдена. Используйте 'models'")

        elif cmd == "generate":
            post_type = None
            topic_parts = parts[1:]
            if len(topic_parts) >= 3 and topic_parts[0] == "-t":
                post_type = topic_parts[1]
                topic_parts = topic_parts[2:]
            topic = " ".join(topic_parts)
            if not topic:
                topic = input("Тема поста: ").strip()
            if not topic:
                print("❌ Тема не может быть пустой")
                continue
            client = get_client_for_model(current_model, clients)
            _cmd_generate(topic, post_type, client, current_model, storage)
            session_posts += 1

        elif cmd == "agent":
            topic = " ".join(parts[1:])
            if not topic:
                topic = input("Задача для агента: ").strip()
            if not topic:
                print("❌ Задача не может быть пустой")
                continue
            client = get_client_for_model(current_model, clients)
            _cmd_agent(topic, client, current_model, storage, tracker)
            session_posts += 1

        elif cmd == "batch":
            if len(parts) < 2:
                print("Использование: batch <файл>")
                continue
            client = get_client_for_model(current_model, clients)
            _cmd_batch(parts[1], client, current_model, storage)

        elif cmd == "history":
            n = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 10
            _cmd_history(n, storage)

        elif cmd == "export":
            if len(parts) < 3:
                print("Использование: export <id> <формат>")
                continue
            _cmd_export(parts[1], parts[2], storage)

        elif cmd == "template":
            _cmd_template(parts[1:])

        elif cmd == "chat":
            sub = parts[1].lower() if len(parts) > 1 else ""
            sub_args = parts[2:] if len(parts) > 2 else []

            if sub == "list":
                _cmd_chat_list(session_storage)

            elif sub in ("load", "resume"):
                if not sub_args:
                    print("Использование: chat load <id>")
                else:
                    sid = sub_args[0]
                    sess = session_storage.load(sid)
                    if sess is None:
                        print(f"❌ Сессия '{sid}' не найдена.")
                    else:
                        if sess.status == "closed":
                            print(f"ℹ️  Сессия '{sess.name}' была закрыта, открываем заново...")
                            sess.status = "active"
                            sess.updated_at = datetime.now().isoformat()
                        chat_client = get_client_for_model(sess.model, clients)
                        _run_chat_loop(chat_client, sess, session_storage, tracker)

            elif sub == "delete":
                if not sub_args:
                    print("Использование: chat delete <id>")
                else:
                    _cmd_chat_delete(sub_args[0], session_storage)

            elif sub == "info":
                if sub_args:
                    sess = session_storage.load(sub_args[0])
                    if sess:
                        _print_session_info(sess)
                    else:
                        print(f"❌ Сессия '{sub_args[0]}' не найдена.")
                else:
                    active = session_storage.find_active()
                    if active:
                        print(f"\nАктивных сессий: {len(active)}")
                        for e in active:
                            print(f"  [{e['session_id']}] {e['name']}")
                    else:
                        print("Нет активных сессий.")

            else:
                if sub == "new":
                    name = " ".join(sub_args).strip()
                elif sub and sub not in ("list", "load", "resume", "delete", "info", "new"):
                    name = " ".join(parts[1:]).strip()
                else:
                    name = ""
                sess = ConversationSession(
                    name=name or f"Чат {datetime.now().strftime('%d.%m %H:%M')}",
                    model=current_model,
                )
                session_storage.save(sess)
                chat_client = get_client_for_model(current_model, clients)
                _run_chat_loop(chat_client, sess, session_storage, tracker)

        elif cmd == "api":
            sub = parts[1].lower() if len(parts) > 1 else ""
            if sub == "advanced":
                _cmd_api_advanced(clients, current_model, tracker)
            else:
                _cmd_api_simple(clients, current_model, tracker)

        elif cmd == "stats":
            _cmd_stats(tracker, parts[1:])

        else:
            print(f"❓ Неизвестная команда: '{cmd}'. Введите 'help'")


if __name__ == "__main__":
    main()
