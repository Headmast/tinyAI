"""Тесты для intent_router.py — классификация намерений."""

import pytest
from unittest.mock import MagicMock, patch
from intent_router import IntentRouter, Intent, RouteResult, _keyword_classify


class TestKeywordClassify:
    """Тесты быстрой keyword-классификации."""

    def test_generate_content(self):
        assert _keyword_classify("Напиши статью про ИИ") == Intent.GENERATE_CONTENT

    def test_schedule_task(self):
        assert _keyword_classify("Добавь задачу на бэкап") == Intent.SCHEDULE_TASK

    def test_search_logs(self):
        assert _keyword_classify("Покажи мне логи разговоров") == Intent.SEARCH_LOGS

    def test_memory(self):
        assert _keyword_classify("Что ты помнишь обо мне?") == Intent.MEMORY

    def test_journalism(self):
        assert _keyword_classify("Проверь текст на ошибки") == Intent.JOURNALISM

    def test_pipeline(self):
        assert _keyword_classify("Запусти pipeline дайджест") == Intent.PIPELINE

    def test_general_returns_none(self):
        assert _keyword_classify("Привет, как дела?") is None

    def test_empty_string(self):
        assert _keyword_classify("") is None

    def test_case_insensitive(self):
        assert _keyword_classify("НАПИШИ ПОСТ") == Intent.GENERATE_CONTENT


class TestIntentRouter:
    """Тесты маршрутизатора IntentRouter."""

    def _make_router(self, mock_client=None):
        client = mock_client or MagicMock()
        return IntentRouter(client=client, verbose=False)

    def test_classify_keywords(self):
        router = self._make_router()
        assert router.classify("Напиши статью") == Intent.GENERATE_CONTENT

    def test_classify_llm_fallback(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="general"))]
        )
        router = self._make_router(mock_client)
        intent = router.classify("Какая сегодня погода?")
        assert intent == Intent.GENERAL

    def test_register_handler_and_handle(self):
        router = self._make_router()
        handler = MagicMock(return_value="Статья готова!")
        router.register_handler(Intent.GENERATE_CONTENT, handler)
        result = router.handle("Напиши статью про Python")
        assert result.intent == Intent.GENERATE_CONTENT
        assert result.response == "Статья готова!"
        handler.assert_called_once()

    def test_handle_without_handler_uses_llm(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Привет!"))]
        )
        router = self._make_router(mock_client)
        result = router.handle("Расскажи о Python")
        assert isinstance(result, RouteResult)

    def test_route_log_grows(self):
        router = self._make_router()
        handler = MagicMock(return_value="ok")
        router.register_handler(Intent.MEMORY, handler)
        router.handle("Запомни что я люблю Python")
        assert len(router._route_log) == 1


class TestIntent:
    """Тесты enum Intent."""

    def test_all_intents_exist(self):
        expected = {
            "generate_content", "schedule_task", "search_logs",
            "memory", "journalism", "pipeline", "general",
        }
        assert {i.value for i in Intent} == expected

    def test_intent_from_string(self):
        assert Intent("general") == Intent.GENERAL
        assert Intent("memory") == Intent.MEMORY
