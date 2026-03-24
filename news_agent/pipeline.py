"""
NewsPipeline — многошаговый конвейер генерации новостных постов.

Этапы: Planner → Researcher → Writer → Editor → SEO
Каждый этап использует отдельную роль с уникальным системным промптом.
"""

import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from news_agent.roles import ROLES, POST_TYPE_GUIDES, get_role


class PipelineError(Exception):
    pass


class NewsPipeline:
    """
    Оркестрирует 5-этапный процесс создания новостного поста.
    Каждый этап — отдельный LLM-вызов с уникальной ролью.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "zai-org/GLM-4.7-Flash",
        verbose: bool = True,
        max_retries: int = 3,
    ):
        self.client = client
        self.model = model
        self.verbose = verbose
        self.max_retries = max_retries
        self._token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def run(
        self,
        topic: str,
        post_type: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Запускает полный pipeline и возвращает финальный результат.

        Возвращает dict:
          - post: финальный текст поста
          - title: заголовок
          - plan: результат планирования
          - research: результат исследования
          - seo: SEO-метаданные
          - quality_score: оценка редактора
          - pipeline_stats: статистика по токенам/времени
        """
        start_time = time.time()
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        self._print_header(topic, post_type)

        step_times: Dict[str, float] = {}

        t0 = time.time()
        plan = self._step_plan(topic, post_type, extra_context)
        step_times["plan"] = round(time.time() - t0, 2)

        t0 = time.time()
        research = self._step_research(topic, plan)
        step_times["research"] = round(time.time() - t0, 2)

        t0 = time.time()
        draft = self._step_write(topic, plan, research)
        step_times["write"] = round(time.time() - t0, 2)

        t0 = time.time()
        edited = self._step_edit(draft, plan)
        step_times["edit"] = round(time.time() - t0, 2)

        t0 = time.time()
        seo = self._step_seo(edited["edited_post"], plan)
        step_times["seo"] = round(time.time() - t0, 2)

        total_time = round(time.time() - start_time, 2)

        best_title = seo.get("headline_variants", [{}])[0].get("text", "") or plan.get("angle", topic)

        if self.verbose:
            self._print_footer(edited, seo, total_time)

        return {
            "post": edited["edited_post"],
            "title": best_title,
            "plan": plan,
            "research": research,
            "seo": seo,
            "quality_score": edited.get("quality_score", 7),
            "quality_breakdown": edited.get("quality_breakdown", {}),
            "tags": seo.get("tags", []),
            "meta_description": seo.get("meta_description", ""),
            "pipeline_stats": {
                "step_times": step_times,
                "total_time": total_time,
                "token_usage": self._token_usage.copy(),
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
            },
        }

    def _step_plan(
        self,
        topic: str,
        post_type: Optional[str],
        extra_context: Optional[str],
    ) -> Dict[str, Any]:
        self._print_step(1, "ПЛАНИРОВАНИЕ", "Анализирую тему и составляю план")
        role = get_role("planner")

        type_hint = ""
        if post_type and post_type in POST_TYPE_GUIDES:
            guide = POST_TYPE_GUIDES[post_type]
            type_hint = (
                f"\nТребуемый тип поста: {post_type} ({guide['description']}). "
                f"Целевой объём: {guide['word_count'][0]}-{guide['word_count'][1]} слов."
            )

        context_hint = f"\nДополнительный контекст: {extra_context}" if extra_context else ""

        user_msg = f"Тема новостного поста: {topic}{type_hint}{context_hint}"
        result = self._call_llm_json(role, user_msg, step_name="plan")

        if self.verbose:
            print(f"  ✓ Тип: {result.get('post_type', '?')} | Угол: {result.get('angle', '?')[:60]}")
        return result

    def _step_research(self, topic: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        self._print_step(2, "ИССЛЕДОВАНИЕ", "Собираю факты и контекст")
        role = get_role("researcher")
        plan_summary = json.dumps(plan, ensure_ascii=False)
        user_msg = (
            f"Тема: {topic}\n\n"
            f"План поста:\n{plan_summary}\n\n"
            "Подготовь фактологическую базу для написания этого поста."
        )
        result = self._call_llm_json(role, user_msg, step_name="research")
        if self.verbose:
            facts_count = len(result.get("key_facts", []))
            print(f"  ✓ Фактов: {facts_count} | Уверенность: {result.get('research_confidence', '?')}")
        return result

    def _step_write(
        self, topic: str, plan: Dict[str, Any], research: Dict[str, Any]
    ) -> str:
        self._print_step(3, "НАПИСАНИЕ", "Создаю черновик поста")
        role = get_role("writer")
        user_msg = (
            f"Тема: {topic}\n\n"
            f"ПЛАН:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
            f"ФАКТЫ И ИССЛЕДОВАНИЕ:\n{json.dumps(research, ensure_ascii=False)}\n\n"
            "Напиши полноценный новостной пост."
        )
        draft, _ = self._call_llm_stream(role, user_msg, step_name="write")
        if self.verbose:
            print(f"  ✓ Черновик: {len(draft.split())} слов, {len(draft)} символов")
        return draft

    def _step_edit(self, draft: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        self._print_step(4, "РЕДАКТУРА", "Проверяю и улучшаю текст")
        role = get_role("editor")
        user_msg = (
            f"ПЛАН ПОСТА:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
            f"ЧЕРНОВИК ДЛЯ РЕДАКТУРЫ:\n{draft}"
        )
        result = self._call_llm_json(role, user_msg, step_name="edit")
        if self.verbose:
            score = result.get("quality_score", "?")
            changes = len(result.get("changes_made", []))
            print(f"  ✓ Оценка качества: {score}/10 | Правок: {changes}")
        return result

    def _step_seo(self, post: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        self._print_step(5, "SEO & МЕТАДАННЫЕ", "Оптимизирую заголовки и теги")
        role = get_role("seo")
        user_msg = (
            f"ГОТОВЫЙ ПОСТ:\n{post}\n\n"
            f"ПЛАН (для контекста):\n{json.dumps(plan, ensure_ascii=False)}"
        )
        result = self._call_llm_json(role, user_msg, step_name="seo")
        if self.verbose:
            headline_count = len(result.get("headline_variants", []))
            print(f"  ✓ Вариантов заголовков: {headline_count} | Тегов: {len(result.get('tags', []))}")
        return result

    def _call_llm_json(
        self,
        role,
        user_message: str,
        step_name: str = "",
    ) -> Dict[str, Any]:
        """Вызывает LLM и парсит JSON-ответ."""
        text, _ = self._call_llm_stream(role, user_message, step_name=step_name, show_stream=False)
        return self._parse_json_response(text, step_name)

    def _call_llm_stream(
        self,
        role,
        user_message: str,
        step_name: str = "",
        show_stream: bool = True,
    ) -> Tuple[str, str]:
        """Вызывает LLM со стримингом и возвращает (content, reasoning)."""
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": role.system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_completion_tokens": 8000,
            "temperature": role.temperature,
            "stream": True,
        }

        if "GLM" in self.model.upper():
            params["extra_body"] = {"thinking": {"type": "enabled", "clear_thinking": False}}

        for attempt in range(self.max_retries):
            try:
                full_content = ""
                reasoning = ""
                in_reasoning = False

                stream = self.client.chat.completions.create(**params)
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        if not in_reasoning and self.verbose and show_stream:
                            print("\n  💭 ", end="", flush=True)
                            in_reasoning = True
                        reasoning += delta.reasoning_content
                        if self.verbose and show_stream:
                            print("·", end="", flush=True)

                    if hasattr(delta, "content") and delta.content:
                        if in_reasoning and self.verbose and show_stream:
                            print()
                            in_reasoning = False
                        full_content += delta.content
                        if self.verbose and show_stream:
                            print(delta.content, end="", flush=True)

                if self.verbose and show_stream:
                    print()

                self._update_token_estimate(full_content, reasoning, user_message)
                return full_content, reasoning

            except Exception as e:
                if attempt < self.max_retries - 1:
                    if self.verbose:
                        print(f"\n  ⚠️  Попытка {attempt + 1} не удалась: {e}. Повтор через 2с...")
                    time.sleep(2)
                else:
                    raise PipelineError(f"Шаг '{step_name}' не выполнен после {self.max_retries} попыток: {e}")

        return "", ""

    def _parse_json_response(self, text: str, step_name: str = "") -> Dict[str, Any]:
        """Извлекает JSON из ответа LLM, игнорируя markdown-обёртки."""
        text = text.strip()

        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        if not text.startswith("{"):
            brace_start = text.find("{")
            brace_end = text.rfind("}") + 1
            if brace_start != -1 and brace_end > brace_start:
                text = text[brace_start:brace_end]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if self.verbose:
                print(f"\n  ⚠️  Не удалось распарсить JSON на шаге '{step_name}'. Возвращаю raw-текст.")
            return {"raw_response": text, "parse_error": True}

    def _update_token_estimate(self, content: str, reasoning: str, prompt: str) -> None:
        estimated_prompt = len(prompt) // 4
        estimated_completion = (len(content) + len(reasoning)) // 4
        self._token_usage["prompt_tokens"] += estimated_prompt
        self._token_usage["completion_tokens"] += estimated_completion
        self._token_usage["total_tokens"] += estimated_prompt + estimated_completion

    def _print_header(self, topic: str, post_type: Optional[str]) -> None:
        if not self.verbose:
            return
        print("\n" + "═" * 65)
        print("  NEWS AGENT PIPELINE")
        print("═" * 65)
        print(f"  Тема:    {topic}")
        print(f"  Тип:     {post_type or 'авто'}")
        print(f"  Модель:  {self.model}")
        print("═" * 65)

    def _print_step(self, num: int, name: str, desc: str) -> None:
        if not self.verbose:
            return
        print(f"\n[{num}/5] {name}")
        print(f"  → {desc}...", flush=True)

    def _print_footer(
        self,
        edited: Dict[str, Any],
        seo: Dict[str, Any],
        total_time: float,
    ) -> None:
        print("\n" + "═" * 65)
        print("  PIPELINE ЗАВЕРШЁН")
        print("═" * 65)
        post_text = edited.get("edited_post", "")
        print(f"  Слов:        {len(post_text.split())}")
        print(f"  Символов:    {len(post_text)}")
        print(f"  Качество:    {edited.get('quality_score', '?')}/10")
        print(f"  Токены ~:    {self._token_usage['total_tokens']}")
        print(f"  Время:       {total_time}с")
        variants = seo.get("headline_variants", [])
        if variants:
            print(f"\n  Заголовки:")
            for v in variants:
                print(f"    [{v.get('type', '?')}] {v.get('text', '')}")
        print("═" * 65)
