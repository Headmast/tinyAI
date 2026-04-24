#!/usr/bin/env python3
"""
Бенчмарк оптимизации локальной LLM (qwen3:8b) для RAG Chat Agent.

Сравнивает конфигурации:
  - Разные temperature (0.0, 0.1, 0.3, 0.5, 0.7)
  - Разные max_tokens (800, 1200, 2000, 3000)
  - Разные top_p (0.7, 0.8, 0.9, 1.0)
  - Квантованные модели (qwen3:8b, qwen3:8b-q4_K_M, qwen3:8b-q8_0)
  - Оригинальный vs оптимизированный промпт-шаблон
  - С /nothink и без

Метрики: время ответа, tokens/sec, длина, RAM, качество (рубрика).

Использование:
    python benchmarks/benchmark_rag_local.py                    # Полный бенчмарк
    python benchmarks/benchmark_rag_local.py --phase params     # Только параметры
    python benchmarks/benchmark_rag_local.py --phase quant      # Только квантование
    python benchmarks/benchmark_rag_local.py --phase prompts    # Только промпты
    python benchmarks/benchmark_rag_local.py --phase combined   # Лучшая комбинация
    python benchmarks/benchmark_rag_local.py --quick            # Быстрый (2 вопроса)
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from core.config import get_llm_client, OLLAMA_BASE_URL

# ── Тестовые вопросы ──────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    {
        "id": "factual_1",
        "question": "Какие компоненты входят в архитектуру RAG-системы TinyAI?",
        "type": "factual",
        "expected_keywords": ["FAISS", "embedder", "reranker", "chunker", "search"],
        "description": "Фактический поиск по архитектуре",
    },
    {
        "id": "factual_2",
        "question": "Как работает MCP Agent в TinyAI и какие инструменты он использует?",
        "type": "factual",
        "expected_keywords": ["MCP", "tool", "agent", "JSON-RPC"],
        "description": "Поиск по MCP-компонентам",
    },
    {
        "id": "multihop_1",
        "question": "Чем отличается fast mode от standard mode в RAG-поиске и когда стоит использовать каждый?",
        "type": "multi_hop",
        "expected_keywords": ["MathReranker", "LLM", "rewrite", "fast", "speed"],
        "description": "Multi-hop: сравнение двух режимов",
    },
    {
        "id": "detail_1",
        "question": "Какие параметры temperature рекомендуются для разных задач в TinyAI?",
        "type": "detail",
        "expected_keywords": ["temperature", "0.0", "0.7", "1.2"],
        "description": "Детальный поиск по параметрам",
    },
    {
        "id": "synthesis_1",
        "question": "Опиши полный цикл обработки запроса в journalist FSM agent от получения темы до публикации.",
        "type": "synthesis",
        "expected_keywords": ["planning", "research", "drafting", "editing", "validation"],
        "description": "Синтез — полный pipeline",
    },
]

QUICK_QUESTIONS = TEST_QUESTIONS[:2]


# ── Оптимизированный промпт для локальной модели ─────────────────────────────

OPTIMIZED_SYSTEM_PROMPT = """\
Ты — ассистент проекта TinyAI. Отвечай ТОЛЬКО на основе контекста ниже.

Правила:
- Используй ТОЛЬКО информацию из контекста. Нет данных — скажи прямо.
- Отвечай на языке вопроса.
- Формат: сначала ответ, потом цитаты, потом источники.

Формат ответа:
1. Основной ответ (2-4 абзаца, конкретно и по делу)
2. Цитаты (2-3 штуки):
   > «цитата» — [N] документ
3. Источники (после ───):
   [N] Файл — Раздел

{context}
"""

ORIGINAL_SYSTEM_PROMPT = """\
Ты — интеллектуальный ассистент, который помогает пользователям разобраться в сложных темах.
Ты отвечаешь на вопросы опираясь на предоставленный контекст из базы знаний, но говоришь живым, понятным языком.

### Правила:
1. Используй ТОЛЬКО информацию из контекста ниже.
2. Если информации недостаточно — скажи об этом прямо и честно.
3. Отвечай на языке вопроса (русский или английский).
4. Отвечай как знающий эксперт, но простым, понятным языком — не сухо и не как справочник.

### Формат ответа:
1. **Основной ответ** — дай полный, связный текст, объясни тему как человеку. Используй абзацы, списки, примеры где уместно.
2. **Цитаты из документов** — приведи 2-3 наиболее уместных цитаты прямо из текста контекста, чтобы подтвердить ответ. Цитаты оборачивай в знаки кавычек «» и указывай откуда они:
   > «цитата» — _документ_
3. **Источники** — в конце после разделителя "───" перечисли все документы, на которые опираешься, в формате:
   [N] Имя файла — Раздел

{context}

Отвечай на вопрос пользователя, придерживаясь формата выше. Начни сразу с объяснения, без вступлений вроде "конечно" или "вот".
"""


# ── Утилиты ───────────────────────────────────────────────────────────────────

def get_memory_usage_mb() -> float:
    """Текущее потребление RAM процессом (RSS) в МБ."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def estimate_tokens(text: str) -> int:
    """Оценка числа токенов (≈3 символа на токен для русского текста)."""
    return max(1, len(text) // 3)


def score_keywords(response: str, expected: List[str]) -> float:
    """Доля ожидаемых ключевых слов, найденных в ответе (0.0–1.0)."""
    if not expected:
        return 1.0
    lower = response.lower()
    found = sum(1 for kw in expected if kw.lower() in lower)
    return found / len(expected)


def has_citations(response: str) -> bool:
    """Есть ли в ответе цитаты в формате > «...»."""
    return bool("«" in response and "»" in response)


def has_sources(response: str) -> bool:
    """Есть ли блок источников [N]."""
    import re
    return bool(re.search(r'\[\d+\]', response))


def build_context_from_rag(
    question: str,
    index_dir: Path,
    top_k: int = 5,
    top_k_before: int = 8,
    strategy: str = "structure",
) -> str:
    """Выполнить RAG-поиск и вернуть текстовый контекст для промпта."""
    try:
        from rag.search import search as rag_search
        from rag.reranker import MathReranker

        reranker = MathReranker(cosine_weight=0.6, bm25_weight=0.4, verbose=False)

        results = rag_search(
            query=question,
            top_k=top_k,
            strategy=strategy,
            index_dir=index_dir,
            top_k_before=top_k_before,
            top_k_after=top_k,
            similarity_threshold=0.30,
            reranker=reranker,
        )

        if not results:
            return "(Контекст не найден)"

        parts = []
        for i, r in enumerate(results, start=1):
            source = r.chunk.metadata.get("source", "unknown")
            section = r.chunk.metadata.get("section", "")
            header = f"[{i}] Источник: {source}"
            if section:
                header += f" | Раздел: {section}"
            parts.append(f"{header}\n{r.chunk.text}")

        return "\n\n---\n\n".join(parts)

    except Exception as e:
        return f"(Ошибка RAG: {e})"


def call_ollama(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    max_tokens: int = 1200,
    top_p: float = 0.9,
    use_nothink: bool = True,
) -> Dict[str, Any]:
    """
    Вызов Ollama-модели с замером всех метрик.

    Returns:
        dict с ключами: response, total_time, ttft, tokens_est,
        tokens_per_sec, prompt_tokens_est, response_length, word_count
    """
    if use_nothink and "qwen3" in model.lower():
        user_message = user_message + " /nothink"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }

    full_response = ""
    start_time = time.monotonic()
    first_token_time = None

    try:
        stream = client.chat.completions.create(**params)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                if first_token_time is None:
                    first_token_time = time.monotonic()
                full_response += delta.content

        end_time = time.monotonic()
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_time

        completion_tokens = estimate_tokens(full_response)
        prompt_tokens = estimate_tokens(system_prompt + user_message)

        return {
            "response": full_response,
            "total_time": round(total_time, 3),
            "ttft": round(ttft, 3),
            "completion_tokens_est": completion_tokens,
            "prompt_tokens_est": prompt_tokens,
            "tokens_per_sec": round(completion_tokens / total_time, 2) if total_time > 0 else 0,
            "response_length": len(full_response),
            "word_count": len(full_response.split()),
            "error": None,
        }

    except Exception as e:
        return {
            "response": "",
            "total_time": round(time.monotonic() - start_time, 3),
            "ttft": 0,
            "completion_tokens_est": 0,
            "prompt_tokens_est": 0,
            "tokens_per_sec": 0,
            "response_length": 0,
            "word_count": 0,
            "error": str(e),
        }


# ── Конфигурации для бенчмарка ────────────────────────────────────────────────

def get_param_configs() -> List[Dict[str, Any]]:
    """Phase 2: grid search по temperature, max_tokens, top_p."""
    configs = []

    # Базовый конфиг (текущие значения)
    configs.append({
        "name": "baseline",
        "label": "Baseline (temp=0.4, max=1200, top_p=0.9)",
        "model": "qwen3:8b",
        "temperature": 0.4,
        "max_tokens": 1200,
        "top_p": 0.9,
        "prompt": "original",
        "nothink": True,
    })

    # Temperature sweep
    for temp in [0.0, 0.1, 0.3, 0.5, 0.7]:
        if temp == 0.4:
            continue  # уже в baseline
        configs.append({
            "name": f"temp_{temp}",
            "label": f"Temperature={temp}",
            "model": "qwen3:8b",
            "temperature": temp,
            "max_tokens": 1200,
            "top_p": 0.9,
            "prompt": "original",
            "nothink": True,
        })

    # Max tokens sweep
    for mt in [800, 2000, 3000]:
        configs.append({
            "name": f"max_tok_{mt}",
            "label": f"Max tokens={mt}",
            "model": "qwen3:8b",
            "temperature": 0.3,
            "max_tokens": mt,
            "top_p": 0.9,
            "prompt": "original",
            "nothink": True,
        })

    # Top-p sweep
    for tp in [0.7, 0.8, 1.0]:
        configs.append({
            "name": f"top_p_{tp}",
            "label": f"Top-p={tp}",
            "model": "qwen3:8b",
            "temperature": 0.3,
            "max_tokens": 1200,
            "top_p": tp,
            "prompt": "original",
            "nothink": True,
        })

    # /nothink comparison
    configs.append({
        "name": "no_nothink",
        "label": "Без /nothink (thinking mode ON)",
        "model": "qwen3:8b",
        "temperature": 0.3,
        "max_tokens": 1200,
        "top_p": 0.9,
        "prompt": "original",
        "nothink": False,
    })

    return configs


def get_quant_configs() -> List[Dict[str, Any]]:
    """Phase 3: сравнение квантованных моделей."""
    return [
        {
            "name": "qwen3_8b_full",
            "label": "qwen3:8b (default/fp16)",
            "model": "qwen3:8b",
            "temperature": 0.3,
            "max_tokens": 1200,
            "top_p": 0.9,
            "prompt": "original",
            "nothink": True,
        },
        {
            "name": "qwen3_8b_q8",
            "label": "qwen3:8b-q8_0 (8-bit quant)",
            "model": "qwen3:8b-q8_0",
            "temperature": 0.3,
            "max_tokens": 1200,
            "top_p": 0.9,
            "prompt": "original",
            "nothink": True,
        },
        {
            "name": "qwen3_8b_q4",
            "label": "qwen3:8b-q4_K_M (4-bit quant)",
            "model": "qwen3:8b-q4_K_M",
            "temperature": 0.3,
            "max_tokens": 1200,
            "top_p": 0.9,
            "prompt": "original",
            "nothink": True,
        },
    ]


def get_prompt_configs() -> List[Dict[str, Any]]:
    """Phase 4: сравнение промптов."""
    return [
        {
            "name": "prompt_original",
            "label": "Оригинальный промпт (GLM-style)",
            "model": "qwen3:8b",
            "temperature": 0.3,
            "max_tokens": 1200,
            "top_p": 0.9,
            "prompt": "original",
            "nothink": True,
        },
        {
            "name": "prompt_optimized",
            "label": "Оптимизированный промпт (local-style)",
            "model": "qwen3:8b",
            "temperature": 0.3,
            "max_tokens": 1200,
            "top_p": 0.9,
            "prompt": "optimized",
            "nothink": True,
        },
        {
            "name": "prompt_optimized_no_nothink",
            "label": "Оптим. промпт + thinking mode",
            "model": "qwen3:8b",
            "temperature": 0.3,
            "max_tokens": 1200,
            "top_p": 0.9,
            "prompt": "optimized",
            "nothink": False,
        },
    ]


def get_combined_configs() -> List[Dict[str, Any]]:
    """Phase 5: лучшая конфигурация на разных квантизациях."""
    configs = []
    for model_tag, label in [
        ("qwen3:8b", "Full (fp16)"),
        ("qwen3:8b-q8_0", "Q8_0"),
        ("qwen3:8b-q4_K_M", "Q4_K_M"),
    ]:
        configs.append({
            "name": f"best_{model_tag.replace(':', '_').replace('-', '_')}",
            "label": f"Best config + {label}",
            "model": model_tag,
            "temperature": 0.1,
            "max_tokens": 1200,
            "top_p": 0.9,
            "prompt": "optimized",
            "nothink": True,
        })
    return configs


# ── Движок бенчмарка ──────────────────────────────────────────────────────────

def run_benchmark(
    configs: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    index_dir: Path,
    phase_name: str = "benchmark",
) -> List[Dict[str, Any]]:
    """
    Запустить бенчмарк: для каждой конфигурации × каждый вопрос.

    Returns:
        Список результатов с полной детализацией.
    """
    client = get_llm_client("ollama")
    all_results = []

    print(f"\n{'=' * 70}")
    print(f"  БЕНЧМАРК: {phase_name}")
    print(f"  Конфигураций: {len(configs)} | Вопросов: {len(questions)}")
    print(f"{'=' * 70}")

    # Предварительно получаем RAG-контекст для каждого вопроса (один раз)
    print("\n⏳ Подготовка RAG-контекста...")
    contexts = {}
    for q in questions:
        ctx = build_context_from_rag(q["question"], index_dir)
        contexts[q["id"]] = ctx
        print(f"  ✓ {q['id']}: {len(ctx)} chars")

    for ci, config in enumerate(configs, start=1):
        print(f"\n{'─' * 70}")
        print(f"  [{ci}/{len(configs)}] {config['label']}")
        print(f"  model={config['model']}  temp={config['temperature']}  "
              f"max_tok={config['max_tokens']}  top_p={config['top_p']}  "
              f"nothink={config['nothink']}  prompt={config['prompt']}")
        print(f"{'─' * 70}")

        config_results = {
            "config": config,
            "questions": [],
            "aggregated": {},
        }

        total_time = 0
        total_ttft = 0
        total_tokens = 0
        total_words = 0
        total_keyword_score = 0
        total_citations = 0
        total_sources = 0
        errors = 0

        for qi, q in enumerate(questions, start=1):
            prompt_template = (
                OPTIMIZED_SYSTEM_PROMPT if config["prompt"] == "optimized"
                else ORIGINAL_SYSTEM_PROMPT
            )
            system_prompt = prompt_template.format(context=contexts[q["id"]])

            ram_before = get_memory_usage_mb()

            print(f"  Q{qi}: {q['description']}...", end=" ", flush=True)

            result = call_ollama(
                client=client,
                model=config["model"],
                system_prompt=system_prompt,
                user_message=q["question"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                top_p=config["top_p"],
                use_nothink=config["nothink"],
            )

            ram_after = get_memory_usage_mb()

            # Оценка качества
            kw_score = score_keywords(result["response"], q["expected_keywords"])
            cit = has_citations(result["response"])
            src = has_sources(result["response"])

            q_result = {
                "question_id": q["id"],
                "question_type": q["type"],
                **result,
                "keyword_score": round(kw_score, 2),
                "has_citations": cit,
                "has_sources": src,
                "ram_before_mb": round(ram_before, 1),
                "ram_after_mb": round(ram_after, 1),
            }

            config_results["questions"].append(q_result)

            if result["error"]:
                errors += 1
                print(f"❌ {result['error']}")
            else:
                total_time += result["total_time"]
                total_ttft += result["ttft"]
                total_tokens += result["completion_tokens_est"]
                total_words += result["word_count"]
                total_keyword_score += kw_score
                total_citations += int(cit)
                total_sources += int(src)
                print(
                    f"✓ {result['total_time']:.1f}s  "
                    f"{result['tokens_per_sec']:.0f} tok/s  "
                    f"{result['word_count']}w  "
                    f"kw={kw_score:.0%}  "
                    f"cit={'✓' if cit else '✗'}  "
                    f"src={'✓' if src else '✗'}"
                )

        # Агрегированные метрики
        n = len(questions) - errors
        if n > 0:
            config_results["aggregated"] = {
                "avg_time": round(total_time / n, 3),
                "avg_ttft": round(total_ttft / n, 3),
                "avg_tokens_per_sec": round(total_tokens / total_time, 2) if total_time > 0 else 0,
                "avg_word_count": round(total_words / n, 1),
                "avg_keyword_score": round(total_keyword_score / n, 2),
                "citation_rate": round(total_citations / n, 2),
                "source_rate": round(total_sources / n, 2),
                "errors": errors,
                "total_questions": len(questions),
            }
        else:
            config_results["aggregated"] = {"errors": errors, "total_questions": len(questions)}

        all_results.append(config_results)

        # Пауза между конфигурациями (дать модели остыть)
        time.sleep(1)

    return all_results


# ── Отчёт ─────────────────────────────────────────────────────────────────────

def create_json_report(results: List[Dict], phase: str, output_dir: Path) -> Path:
    """Сохранить результаты в JSON."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"rag_local_{phase}_{ts}.json"

    report = {
        "benchmark": "RAG Local LLM Optimization",
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📄 JSON: {filename}")
    return filename


def create_markdown_report(results: List[Dict], phase: str, output_dir: Path) -> Path:
    """Создать Markdown-отчёт с таблицами."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"rag_local_{phase}_{ts}.md"

    lines = [
        f"# Бенчмарк RAG Local LLM — {phase}",
        f"\n**Дата:** {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        f"\n**Модель базовая:** qwen3:8b (Ollama)",
        f"\n**Фаза:** {phase}",
        "",
    ]

    # Таблица агрегированных метрик
    lines.append("## Сводная таблица\n")
    lines.append(
        "| Конфигурация | Ср. время (с) | TTFT (с) | Tok/sec | Слов | "
        "Keywords | Цитаты | Источники | Ошибки |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|"
    )

    for r in results:
        cfg = r["config"]
        agg = r.get("aggregated", {})
        if "avg_time" not in agg:
            lines.append(f"| {cfg['label']} | — | — | — | — | — | — | — | {agg.get('errors', '?')} |")
            continue
        lines.append(
            f"| {cfg['label']} "
            f"| {agg['avg_time']:.2f} "
            f"| {agg['avg_ttft']:.2f} "
            f"| {agg['avg_tokens_per_sec']:.1f} "
            f"| {agg['avg_word_count']:.0f} "
            f"| {agg['avg_keyword_score']:.0%} "
            f"| {agg['citation_rate']:.0%} "
            f"| {agg['source_rate']:.0%} "
            f"| {agg['errors']} |"
        )

    # Детали по каждому конфигу
    lines.append("\n## Детализация по конфигурациям\n")

    for r in results:
        cfg = r["config"]
        lines.append(f"### {cfg['label']}\n")
        lines.append(f"- **Модель:** `{cfg['model']}`")
        lines.append(f"- **Temperature:** {cfg['temperature']}")
        lines.append(f"- **Max tokens:** {cfg['max_tokens']}")
        lines.append(f"- **Top-p:** {cfg['top_p']}")
        lines.append(f"- **Промпт:** {cfg['prompt']}")
        lines.append(f"- **nothink:** {cfg['nothink']}")
        lines.append("")

        for qr in r["questions"]:
            status = "❌" if qr.get("error") else "✓"
            lines.append(f"**{qr['question_id']}** ({qr['question_type']}) — {status}")
            if qr.get("error"):
                lines.append(f"  Ошибка: {qr['error']}")
            else:
                lines.append(
                    f"  Время: {qr['total_time']:.2f}s | "
                    f"TTFT: {qr['ttft']:.2f}s | "
                    f"{qr['tokens_per_sec']:.0f} tok/s | "
                    f"{qr['word_count']} слов | "
                    f"KW: {qr['keyword_score']:.0%} | "
                    f"Цит: {'✓' if qr['has_citations'] else '✗'} | "
                    f"Ист: {'✓' if qr['has_sources'] else '✗'}"
                )
                # Первые 300 символов ответа
                preview = qr["response"][:300].replace("\n", " ")
                lines.append(f"\n  > {preview}...\n")

        lines.append("")

    # Выводы
    lines.append("## Выводы\n")

    # Найти лучшую конфигурацию по комбинированному скору
    best = None
    best_score = -1
    for r in results:
        agg = r.get("aggregated", {})
        if "avg_keyword_score" not in agg:
            continue
        # Скор = keyword_score * 0.4 + citation_rate * 0.2 + source_rate * 0.2 + speed_score * 0.2
        speed_score = min(1.0, agg["avg_tokens_per_sec"] / 50)  # 50 tok/s = max
        combined = (
            agg["avg_keyword_score"] * 0.4
            + agg["citation_rate"] * 0.2
            + agg["source_rate"] * 0.2
            + speed_score * 0.2
        )
        if combined > best_score:
            best_score = combined
            best = r

    if best:
        cfg = best["config"]
        agg = best["aggregated"]
        lines.append(f"**Лучшая конфигурация:** {cfg['label']}")
        lines.append(f"- Комбинированный скор: {best_score:.2f}")
        lines.append(f"- Среднее время: {agg['avg_time']:.2f}s")
        lines.append(f"- Keywords: {agg['avg_keyword_score']:.0%}")
        lines.append(f"- Цитаты: {agg['citation_rate']:.0%}")
        lines.append(f"- Источники: {agg['source_rate']:.0%}")
        lines.append(f"- Скорость: {agg['avg_tokens_per_sec']:.1f} tok/s")
    else:
        lines.append("Недостаточно данных для выводов.")

    lines.append("")

    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📋 Markdown: {filename}")
    return filename


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG Local LLM optimization")
    parser.add_argument(
        "--phase",
        choices=["params", "quant", "prompts", "combined", "all"],
        default="all",
        help="Какую фазу запустить",
    )
    parser.add_argument("--quick", action="store_true", help="Быстрый режим (2 вопроса)")
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Папка для результатов",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    index_dir = project_root / "rag_data"
    output_dir = project_root / args.output_dir

    questions = QUICK_QUESTIONS if args.quick else TEST_QUESTIONS

    print("=" * 70)
    print("  RAG LOCAL LLM OPTIMIZATION BENCHMARK")
    print(f"  Фаза: {args.phase} | Вопросов: {len(questions)} | Quick: {args.quick}")
    print("=" * 70)

    # Проверяем доступность Ollama
    try:
        client = get_llm_client("ollama")
        test_resp = client.models.list()
        available_models = [m.id for m in test_resp.data] if hasattr(test_resp, 'data') else []
        print(f"\n✓ Ollama доступен. Модели: {', '.join(available_models[:5])}")
    except Exception as e:
        print(f"\n❌ Ollama недоступен: {e}")
        print("Запустите: ./run_ollama.sh")
        sys.exit(1)

    all_phase_results = {}

    phases_to_run = []
    if args.phase == "all":
        phases_to_run = ["params", "quant", "prompts", "combined"]
    else:
        phases_to_run = [args.phase]

    for phase in phases_to_run:
        print(f"\n\n{'#' * 70}")
        print(f"  ФАЗА: {phase.upper()}")
        print(f"{'#' * 70}")

        if phase == "params":
            configs = get_param_configs()
        elif phase == "quant":
            configs = get_quant_configs()
            # Проверяем доступность квантованных моделей
            for cfg in configs:
                model = cfg["model"]
                if available_models and model not in available_models:
                    print(f"  ⚠️ Модель {model} не найдена. Пропускаем или скачайте: ollama pull {model}")
            configs = [c for c in configs if not available_models or c["model"] in available_models]
        elif phase == "prompts":
            configs = get_prompt_configs()
        elif phase == "combined":
            configs = get_combined_configs()
            configs = [c for c in configs if not available_models or c["model"] in available_models]
        else:
            continue

        if not configs:
            print(f"  ⚠️ Нет доступных конфигураций для фазы {phase}")
            continue

        results = run_benchmark(configs, questions, index_dir, phase_name=phase)
        all_phase_results[phase] = results

        create_json_report(results, phase, output_dir)
        create_markdown_report(results, phase, output_dir)

    # Итоговый отчёт
    if len(all_phase_results) > 1:
        # Объединяем все результаты в один отчёт
        combined = []
        for phase_name, phase_results in all_phase_results.items():
            combined.extend(phase_results)
        create_json_report(combined, "full", output_dir)
        create_markdown_report(combined, "full", output_dir)

    print(f"\n\n{'=' * 70}")
    print("  БЕНЧМАРК ЗАВЕРШЁН")
    print(f"  Результаты: {output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
