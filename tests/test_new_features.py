"""
Тесты для новых фич (День 20+):
  1. PipelineTemplates — шаблоны пайплайнов
  2. MCPRegistry.health_check — проверка доступности серверов
  3. IntentRouter — маршрутизация запросов к агентам
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# 1. PipelineTemplates
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineTemplates:
    """Тесты шаблонов пайплайнов."""

    def test_list_templates(self):
        from pipeline_templates import list_templates
        templates = list_templates()
        assert len(templates) >= 4
        names = {t["name"] for t in templates}
        assert "social_media_post" in names
        assert "daily_digest" in names
        assert "research_report" in names
        assert "content_publish" in names

    def test_template_has_required_fields(self):
        from pipeline_templates import list_templates
        for t in list_templates():
            assert "name" in t
            assert "description" in t
            assert "tags" in t
            assert "default_params" in t
            assert len(t["description"]) > 0

    def test_get_template(self):
        from pipeline_templates import get_template
        t = get_template("social_media_post")
        assert t is not None
        assert t.name == "social_media_post"
        assert callable(t.build)

    def test_get_template_not_found(self):
        from pipeline_templates import get_template
        assert get_template("nonexistent") is None

    def test_build_social_media_post(self):
        from pipeline_templates import build_steps
        steps = build_steps("social_media_post", topic="AI нейросети")
        assert len(steps) == 4
        assert steps[0].tool == "search"
        assert steps[0].args["query"] == "AI нейросети"
        assert steps[1].tool == "summarize"
        assert steps[1].args["text"] == "$prev.result"
        assert steps[2].tool == "format_content"
        assert steps[2].args["platform"] == "telegram"
        assert steps[3].tool == "save_to_file"

    def test_build_social_media_custom_platform(self):
        from pipeline_templates import build_steps
        steps = build_steps("social_media_post", topic="Test", platform="website")
        assert steps[2].args["platform"] == "website"

    def test_build_daily_digest(self):
        from pipeline_templates import build_steps
        steps = build_steps("daily_digest", topic="AI news")
        assert len(steps) == 3
        assert steps[0].tool == "search"
        assert steps[1].tool == "summarize"
        assert steps[1].args["style"] == "digest"
        assert steps[2].tool == "save_to_file"

    def test_build_daily_digest_no_topic(self):
        """daily_digest может работать без topic."""
        from pipeline_templates import build_steps
        steps = build_steps("daily_digest")
        assert len(steps) == 3
        assert steps[0].args["query"] == "итоги дня"

    def test_build_research_report(self):
        from pipeline_templates import build_steps
        steps = build_steps("research_report", topic="Machine Learning")
        assert len(steps) == 5
        assert steps[0].tool == "search"
        assert steps[1].tool == "summarize"
        assert steps[1].args["style"] == "brief"
        assert steps[2].tool == "format_content"
        assert steps[2].args["platform"] == "website"
        assert steps[3].tool == "save_to_file"
        assert steps[4].tool == "save_to_db"
        # Проверяем подстановки
        assert steps[4].args["content"] == "$steps.2.result"
        assert steps[4].args["summary"] == "$steps.1.result"

    def test_build_content_publish(self):
        from pipeline_templates import build_steps
        steps = build_steps("content_publish", topic="Neural nets")
        assert len(steps) == 5
        assert steps[3].tool == "save_to_file"
        assert steps[4].tool == "save_to_db"

    def test_build_missing_topic_raises(self):
        from pipeline_templates import build_steps
        with pytest.raises(ValueError, match="topic"):
            build_steps("social_media_post")  # no topic

    def test_build_unknown_template_raises(self):
        from pipeline_templates import build_steps
        with pytest.raises(ValueError, match="не найден"):
            build_steps("nonexistent_template", topic="test")

    def test_steps_have_names(self):
        """Все шаги шаблонов должны иметь имена для логирования."""
        from pipeline_templates import build_steps
        steps = build_steps("content_publish", topic="Test")
        for step in steps:
            assert step.name is not None, f"Step {step.tool} has no name"

    def test_custom_filename(self):
        from pipeline_templates import build_steps
        steps = build_steps("social_media_post", topic="X", filename="custom.md")
        assert steps[3].args["filename"] == "custom.md"

    def test_custom_style(self):
        from pipeline_templates import build_steps
        steps = build_steps("social_media_post", topic="X", style="digest")
        assert steps[1].args["style"] == "digest"


# ══════════════════════════════════════════════════════════════════════════════
# 2. MCPRegistry.health_check
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthCheck:
    """Тесты проверки доступности серверов."""

    def test_health_check_healthy_server(self):
        from mcp_registry import MCPRegistry

        reg = MCPRegistry()
        reg.register("s1", str(PROJECT_ROOT / "mcp_server.py"))

        mock_conn = MagicMock()
        mock_conn.list_tools.return_value = [
            {"name": "t1"}, {"name": "t2"},
        ]
        reg._connections["s1"] = mock_conn

        results = reg.health_check()
        assert len(results) == 1
        assert results[0]["name"] == "s1"
        assert results[0]["status"] == "healthy"
        assert results[0]["tool_count"] == 2
        assert results[0]["latency_ms"] >= 0
        assert results[0]["error"] is None

    def test_health_check_unhealthy_server(self):
        from mcp_registry import MCPRegistry

        reg = MCPRegistry()
        reg.register("broken", str(PROJECT_ROOT / "mcp_server.py"))

        mock_conn = MagicMock()
        mock_conn.list_tools.side_effect = RuntimeError("connection lost")
        reg._connections["broken"] = mock_conn

        results = reg.health_check()
        assert results[0]["status"] == "unhealthy"
        assert "connection lost" in results[0]["error"]

    def test_health_check_multiple_servers(self):
        from mcp_registry import MCPRegistry

        reg = MCPRegistry()
        reg.register("s1", str(PROJECT_ROOT / "mcp_server.py"))
        reg.register("s2", str(PROJECT_ROOT / "mcp_scheduler_server.py"))

        healthy_conn = MagicMock()
        healthy_conn.list_tools.return_value = [{"name": "t1"}]

        broken_conn = MagicMock()
        broken_conn.list_tools.side_effect = RuntimeError("fail")

        reg._connections["s1"] = healthy_conn
        reg._connections["s2"] = broken_conn

        results = reg.health_check()
        assert len(results) == 2
        statuses = {r["name"]: r["status"] for r in results}
        assert statuses["s1"] == "healthy"
        assert statuses["s2"] == "unhealthy"

    @pytest.mark.integration
    def test_health_check_real_servers(self):
        from mcp_registry import MCPRegistry

        with MCPRegistry() as reg:
            reg.register("logs", str(PROJECT_ROOT / "mcp_server.py"))
            reg.register("scheduler", str(PROJECT_ROOT / "mcp_scheduler_server.py"))

            results = reg.health_check()
            assert len(results) == 2
            for r in results:
                assert r["status"] == "healthy"
                assert r["tool_count"] > 0
                assert r["latency_ms"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. IntentRouter
# ══════════════════════════════════════════════════════════════════════════════

class TestIntentRouter:
    """Тесты маршрутизатора намерений."""

    def test_classify_generate_content(self):
        from intent_router import IntentRouter, Intent
        router = IntentRouter(client=MagicMock(), verbose=False)
        assert router.classify("Напиши статью про AI") == Intent.GENERATE_CONTENT
        assert router.classify("Сгенерируй пост для телеграма") == Intent.GENERATE_CONTENT

    def test_classify_schedule_task(self):
        from intent_router import IntentRouter, Intent
        router = IntentRouter(client=MagicMock(), verbose=False)
        assert router.classify("Добавь задачу в планировщик") == Intent.SCHEDULE_TASK
        assert router.classify("Создай напоминание на 5 минут") == Intent.SCHEDULE_TASK

    def test_classify_search_logs(self):
        from intent_router import IntentRouter, Intent
        router = IntentRouter(client=MagicMock(), verbose=False)
        assert router.classify("Покажи логи за сегодня") == Intent.SEARCH_LOGS
        assert router.classify("Поищи в истории слово transformer") == Intent.SEARCH_LOGS

    def test_classify_memory(self):
        from intent_router import IntentRouter, Intent
        router = IntentRouter(client=MagicMock(), verbose=False)
        assert router.classify("Что ты помнишь обо мне?") == Intent.MEMORY
        assert router.classify("Запомни: я предпочитаю краткие ответы") == Intent.MEMORY

    def test_classify_journalism(self):
        from intent_router import IntentRouter, Intent
        router = IntentRouter(client=MagicMock(), verbose=False)
        assert router.classify("Проверь текст на ошибки, как журналист") == Intent.JOURNALISM

    def test_classify_pipeline(self):
        from intent_router import IntentRouter, Intent
        router = IntentRouter(client=MagicMock(), verbose=False)
        assert router.classify("Запусти pipeline для дайджеста") == Intent.PIPELINE
        assert router.classify("Используй шаблон пайплайн") == Intent.PIPELINE

    def test_classify_general_fallback_to_llm(self):
        """Если keywords не сработали, используется LLM."""
        from intent_router import IntentRouter, Intent

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "general"
        mock_client.chat.completions.create.return_value = mock_response

        router = IntentRouter(client=mock_client, verbose=False)
        result = router.classify("Какая погода сегодня?")
        assert result == Intent.GENERAL

    def test_classify_llm_returns_unknown(self):
        """Если LLM вернул невалидный intent — fallback на GENERAL."""
        from intent_router import IntentRouter, Intent

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "unknown_type_123"
        mock_client.chat.completions.create.return_value = mock_response

        router = IntentRouter(client=mock_client, verbose=False)
        result = router.classify("xyzzy blorf")
        assert result == Intent.GENERAL

    def test_handle_with_registered_handler(self):
        from intent_router import IntentRouter, Intent

        router = IntentRouter(client=MagicMock(), verbose=False)
        router.register_handler(
            Intent.MEMORY,
            lambda text: f"Я помню: {text}",
        )

        result = router.handle("Что ты помнишь?")
        assert result.intent == Intent.MEMORY
        assert "помню" in result.response
        assert result.agent_used == "MemoryAgent"

    def test_handle_without_handler_uses_llm(self):
        """Если обработчик не зарегистрирован — прямой LLM-ответ."""
        from intent_router import IntentRouter, Intent

        mock_client = MagicMock()
        # LLM classify response
        classify_resp = MagicMock()
        classify_resp.choices = [MagicMock()]
        classify_resp.choices[0].message.content = "general"
        # LLM direct response
        direct_resp = MagicMock()
        direct_resp.choices = [MagicMock()]
        direct_resp.choices[0].message.content = "42"
        mock_client.chat.completions.create.side_effect = [classify_resp, direct_resp]

        router = IntentRouter(client=mock_client, verbose=False)
        result = router.handle("Сколько будет 6x7?")
        assert result.agent_used == "DirectLLM"
        assert result.response == "42"

    def test_handle_exception_in_handler(self):
        from intent_router import IntentRouter, Intent

        def failing_handler(text):
            raise RuntimeError("agent crashed")

        router = IntentRouter(client=MagicMock(), verbose=False)
        router.register_handler(Intent.MEMORY, failing_handler)

        result = router.handle("Запомни что-нибудь")
        assert "Ошибка" in result.response

    def test_route_history(self):
        from intent_router import IntentRouter, Intent

        router = IntentRouter(client=MagicMock(), verbose=False)
        router.register_handler(Intent.MEMORY, lambda t: "ok")
        router.register_handler(Intent.SEARCH_LOGS, lambda t: "found")

        router.handle("Что помнишь?")
        router.handle("Покажи логи")

        history = router.route_history
        assert len(history) == 2
        assert history[0].intent == Intent.MEMORY
        assert history[1].intent == Intent.SEARCH_LOGS

    def test_get_stats(self):
        from intent_router import IntentRouter, Intent

        router = IntentRouter(client=MagicMock(), verbose=False)
        router.register_handler(Intent.MEMORY, lambda t: "ok")

        router.handle("Помнишь?")
        router.handle("Запомни это")

        stats = router.get_stats()
        assert stats["total_requests"] == 2
        assert stats["by_intent"]["memory"] == 2

    def test_multiple_intents_best_match(self):
        """При нескольких совпадениях — выбирается intent с наибольшим score."""
        from intent_router import _keyword_classify, Intent
        # "Поищи в логах и запомни результат" — has both search_logs and memory keywords
        result = _keyword_classify("Поищи в логах и запомни результат")
        assert result in (Intent.SEARCH_LOGS, Intent.MEMORY)
