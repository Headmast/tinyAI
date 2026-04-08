"""
Тесты для news_agent: roles, tools, storage, formatter, pipeline (мок), agent (мок).
Все тесты работают без реального API — используют моки.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from news_agent.roles import ROLES, get_role, POST_TYPE_GUIDES, Role
from news_agent.tools import TOOL_DEFINITIONS, ToolDispatcher
from news_agent.storage import PostStorage
from news_agent.formatter import OutputFormatter
from news_agent.pipeline import NewsPipeline, PipelineError
from news_agent.agent import AgentLoop


class TestRoles(unittest.TestCase):

    def test_all_roles_exist(self):
        for name in ("planner", "researcher", "writer", "editor", "seo", "autonomous"):
            self.assertIn(name, ROLES)

    def test_role_has_required_fields(self):
        for name, role in ROLES.items():
            self.assertIsInstance(role, Role)
            self.assertTrue(role.name)
            self.assertTrue(role.system_prompt)
            self.assertIsInstance(role.temperature, float)
            self.assertGreaterEqual(role.temperature, 0.0)
            self.assertLessEqual(role.temperature, 2.0)

    def test_get_role_returns_correct(self):
        role = get_role("writer")
        self.assertEqual(role.name, "News Writer")

    def test_get_role_fallback(self):
        role = get_role("nonexistent_role")
        self.assertEqual(role, ROLES["writer"])

    def test_post_type_guides_complete(self):
        required_types = {"breaking", "analysis", "digest", "social", "press"}
        self.assertEqual(set(POST_TYPE_GUIDES.keys()), required_types)

    def test_post_type_guide_fields(self):
        for key, guide in POST_TYPE_GUIDES.items():
            self.assertIn("description", guide)
            self.assertIn("word_count", guide)
            self.assertIn("tone", guide)
            self.assertIn("structure", guide)
            self.assertIsInstance(guide["word_count"], tuple)
            self.assertEqual(len(guide["word_count"]), 2)


class TestToolDefinitions(unittest.TestCase):

    def test_tool_definitions_not_empty(self):
        self.assertGreater(len(TOOL_DEFINITIONS), 0)

    def test_tool_definitions_valid_structure(self):
        for tool in TOOL_DEFINITIONS:
            self.assertEqual(tool["type"], "function")
            fn = tool["function"]
            self.assertIn("name", fn)
            self.assertIn("description", fn)
            self.assertIn("parameters", fn)
            self.assertEqual(fn["parameters"]["type"], "object")

    def test_required_tools_present(self):
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        for expected in ("save_post", "load_post_history", "analyze_topic", "format_post"):
            self.assertIn(expected, names)


class TestToolDispatcher(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = PostStorage(base_dir=self.tmp)
        self.dispatcher = ToolDispatcher(storage=self.storage)

    def test_dispatch_unknown_tool(self):
        result = self.dispatcher.dispatch("nonexistent_tool", {})
        data = json.loads(result)
        self.assertIn("error", data)

    def test_analyze_topic_short(self):
        result = json.loads(self.dispatcher.dispatch("analyze_topic", {"topic": "ИИ и рынок"}))
        self.assertEqual(result["suggested_type"], "social")

    def test_analyze_topic_with_numbers(self):
        result = json.loads(
            self.dispatcher.dispatch("analyze_topic", {"topic": "ЦБ поднял ставку до 21% в 2025 году"})
        )
        self.assertTrue(result["has_numbers"])

    def test_format_post_plain(self):
        result = self.dispatcher.dispatch(
            "format_post", {"content": "Тестовый текст", "target_format": "plain", "title": "Заголовок"}
        )
        self.assertIn("Тестовый текст", result)

    def test_save_and_load_post(self):
        save_result = json.loads(self.dispatcher.dispatch(
            "save_post",
            {
                "title": "Тест",
                "content": "Тестовый контент",
                "post_type": "breaking",
                "tags": ["тест"],
            },
        ))
        self.assertTrue(save_result["success"])
        post_id = save_result["post_id"]

        load_result = json.loads(
            self.dispatcher.dispatch("get_post_by_id", {"post_id": post_id})
        )
        self.assertEqual(load_result["title"], "Тест")

    def test_load_history_empty(self):
        result = json.loads(self.dispatcher.dispatch("load_post_history", {}))
        self.assertEqual(result["count"], 0)

    def test_check_duplicate_no_match(self):
        result = json.loads(
            self.dispatcher.dispatch("check_duplicate", {"topic": "космический корабль Mars"})
        )
        self.assertFalse(result["is_duplicate"])


class TestPostStorage(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = PostStorage(base_dir=self.tmp)

    def test_save_returns_id(self):
        post_id = self.storage.save({
            "title": "Test", "content": "Content", "post_type": "breaking"
        })
        self.assertIsInstance(post_id, str)
        self.assertEqual(len(post_id), 8)

    def test_load_saved_post(self):
        post_id = self.storage.save({
            "title": "Hello", "content": "World", "post_type": "analysis"
        })
        post = self.storage.load(post_id)
        self.assertIsNotNone(post)
        self.assertEqual(post["title"], "Hello")
        self.assertEqual(post["content"], "World")
        self.assertEqual(post["post_type"], "analysis")

    def test_load_nonexistent(self):
        self.assertIsNone(self.storage.load("00000000"))

    def test_list_posts_order(self):
        ids = []
        for i in range(3):
            ids.append(self.storage.save({"title": f"Post {i}", "content": "x", "post_type": "social"}))
        posts = self.storage.list_posts(n=10)
        self.assertEqual(len(posts), 3)
        self.assertEqual(posts[0]["id"], ids[-1])

    def test_list_posts_filter_by_type(self):
        self.storage.save({"title": "B", "content": "x", "post_type": "breaking"})
        self.storage.save({"title": "A", "content": "x", "post_type": "analysis"})
        breaking = self.storage.list_posts(n=10, post_type="breaking")
        self.assertEqual(len(breaking), 1)
        self.assertEqual(breaking[0]["title"], "B")

    def test_markdown_file_created(self):
        post_id = self.storage.save({"title": "MD Test", "content": "Content", "post_type": "digest"})
        md_file = Path(self.tmp) / post_id / "post.md"
        self.assertTrue(md_file.exists())
        content = md_file.read_text(encoding="utf-8")
        self.assertIn("# MD Test", content)

    def test_delete_post(self):
        post_id = self.storage.save({"title": "Del", "content": "x", "post_type": "social"})
        self.assertTrue(self.storage.delete(post_id))
        self.assertIsNone(self.storage.load(post_id))
        posts = self.storage.list_posts()
        self.assertFalse(any(p["id"] == post_id for p in posts))

    def test_search_similar(self):
        self.storage.save({"title": "Ставка ЦБ повышена", "content": "x", "post_type": "breaking"})
        results = self.storage.search_similar("ЦБ ставка 2025", threshold=0.3)
        self.assertGreater(len(results), 0)

    def test_word_count_stored(self):
        post_id = self.storage.save({"title": "T", "content": "один два три", "post_type": "social"})
        post = self.storage.load(post_id)
        self.assertEqual(post["word_count"], 3)


class TestOutputFormatter(unittest.TestCase):

    def setUp(self):
        self.fmt = OutputFormatter()
        self.content = "Центробанк повысил ставку.\n\nЭто важное решение."
        self.title = "ЦБ принял решение"
        self.tags = ["ЦБ", "ставка"]

    def test_plain_format(self):
        result = self.fmt.to_plain(self.content, title=self.title, tags=self.tags)
        self.assertIn(self.title.upper(), result)
        self.assertIn("ЦБ", result)

    def test_markdown_format(self):
        result = self.fmt.to_markdown(self.content, title=self.title, tags=self.tags)
        self.assertIn(f"# {self.title}", result)
        self.assertIn(self.content, result)

    def test_html_format(self):
        result = self.fmt.to_html(self.content, title=self.title, tags=self.tags)
        self.assertIn("<html", result)
        self.assertIn(self.title, result)
        self.assertIn("<p>", result)

    def test_telegram_format(self):
        result = self.fmt.to_telegram(self.content, title=self.title, tags=self.tags)
        self.assertIn(f"**{self.title}**", result)
        self.assertIn("#ЦБ", result)

    def test_json_format(self):
        result = self.fmt.to_json(self.content, title=self.title, tags=self.tags)
        data = json.loads(result)
        self.assertEqual(data["title"], self.title)
        self.assertEqual(data["tags"], self.tags)
        self.assertGreater(data["word_count"], 0)

    def test_convert_dispatcher(self):
        for fmt in ("markdown", "html", "telegram", "json", "plain"):
            result = self.fmt.convert(self.content, fmt=fmt, title=self.title)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_convert_unknown_format_fallback(self):
        result = self.fmt.convert(self.content, fmt="unknown_format", title=self.title)
        self.assertIn(self.content.strip(), result)


class TestNewsPipelineMocked(unittest.TestCase):
    """Тестирует Pipeline с мокнутым LLM — без реальных API-вызовов."""

    def _make_mock_stream(self, content: str):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        chunk.choices[0].delta.reasoning_content = None
        return [chunk]

    def _make_plan_json(self):
        return json.dumps({
            "post_type": "breaking",
            "angle": "тестовый угол",
            "tone": "neutral",
            "target_audience": "тест",
            "word_count_target": 200,
            "key_points": ["факт 1", "факт 2"],
            "structure": ["Заголовок", "Лид", "Тело"],
            "notes_for_writer": ""
        })

    def _make_research_json(self):
        return json.dumps({
            "background": "Тестовый контекст",
            "key_facts": ["факт 1", "факт 2"],
            "statistics": [],
            "key_players": [],
            "timeline": [],
            "expert_perspectives": [],
            "controversies": [],
            "research_confidence": "medium",
            "notes": ""
        })

    def _make_edit_json(self):
        return json.dumps({
            "edited_post": "Отредактированный пост. Второй абзац.",
            "changes_made": ["правка 1"],
            "quality_score": 8,
            "quality_breakdown": {"clarity": 8, "accuracy": 8, "engagement": 7, "structure": 9},
            "editorial_notes": "хорошо"
        })

    def _make_seo_json(self):
        return json.dumps({
            "headline_variants": [
                {"type": "seo", "text": "SEO заголовок"},
                {"type": "social", "text": "Социальный заголовок"}
            ],
            "meta_description": "Тестовое мета-описание",
            "tags": ["тест", "новость"],
            "hashtags": ["#тест"],
            "keywords": ["тест"],
            "suggested_channels": ["Telegram"],
            "best_posting_time": "утро"
        })

    def setUp(self):
        self.mock_client = MagicMock()
        self.tmp = tempfile.mkdtemp()

    def _setup_stream_side_effects(self, responses):
        """Настраивает mock для последовательных streaming-ответов."""
        call_count = [0]
        def side_effect(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(responses):
                return iter(self._make_mock_stream(responses[idx]))
            return iter(self._make_mock_stream("{}"))
        self.mock_client.chat.completions.create.side_effect = side_effect

    def test_pipeline_run_full(self):
        responses = [
            self._make_plan_json(),
            self._make_research_json(),
            "Тестовый черновик поста. Подробности следуют.",
            self._make_edit_json(),
            self._make_seo_json(),
        ]
        self._setup_stream_side_effects(responses)

        pipeline = NewsPipeline(client=self.mock_client, model="test-model", verbose=False)
        result = pipeline.run(topic="Тестовая тема")

        self.assertIn("post", result)
        self.assertIn("title", result)
        self.assertIn("seo", result)
        self.assertIn("plan", result)
        self.assertIn("pipeline_stats", result)
        self.assertGreater(result["quality_score"], 0)

    def test_pipeline_saves_token_stats(self):
        responses = [
            self._make_plan_json(),
            self._make_research_json(),
            "Текст поста",
            self._make_edit_json(),
            self._make_seo_json(),
        ]
        self._setup_stream_side_effects(responses)

        pipeline = NewsPipeline(client=self.mock_client, model="test-model", verbose=False)
        result = pipeline.run(topic="Тема")
        stats = result["pipeline_stats"]

        self.assertIn("total_time", stats)
        self.assertIn("token_usage", stats)
        self.assertIn("step_times", stats)
        self.assertEqual(set(stats["step_times"].keys()), {"plan", "research", "write", "edit", "seo"})

    def test_pipeline_parse_json_fallback(self):
        """Если LLM вернул не-JSON — pipeline не падает, возвращает raw_response."""
        responses = [
            "просто текст без JSON",
            self._make_research_json(),
            "Черновик",
            self._make_edit_json(),
            self._make_seo_json(),
        ]
        self._setup_stream_side_effects(responses)

        pipeline = NewsPipeline(client=self.mock_client, model="test-model", verbose=False)
        result = pipeline.run(topic="Тема")

        plan = result.get("plan", {})
        self.assertIn("raw_response", plan)


class TestAgentLoopMocked(unittest.TestCase):
    """Тестирует AgentLoop с мокнутым LLM."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.tmp = tempfile.mkdtemp()
        self.storage = PostStorage(base_dir=self.tmp)

    def _make_response(self, content: str, tool_calls=None):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        response.choices[0].message.tool_calls = tool_calls or []
        response.usage = MagicMock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        response.usage.total_tokens = 150
        return response

    def test_agent_finishes_with_marker(self):
        self.mock_client.chat.completions.create.return_value = self._make_response(
            "FINAL_POST: Готовый тестовый пост о технологиях."
        )
        agent = AgentLoop(
            client=self.mock_client,
            model="test-model",
            storage=self.storage,
            verbose=False,
        )
        result = agent.run("Тема: технологии")

        self.assertIn("технологиях", result["final_post"])
        self.assertEqual(result["iterations"], 1)

    def test_agent_accumulates_token_usage(self):
        self.mock_client.chat.completions.create.return_value = self._make_response(
            "FINAL_POST: Финальный пост."
        )
        agent = AgentLoop(
            client=self.mock_client,
            model="test-model",
            storage=self.storage,
            verbose=False,
        )
        result = agent.run("Тема")
        self.assertGreater(result["token_usage"]["total_tokens"], 0)

    def test_agent_respects_max_iterations(self):
        self.mock_client.chat.completions.create.return_value = self._make_response(
            "Думаю..."
        )
        agent = AgentLoop(
            client=self.mock_client,
            model="test-model",
            storage=self.storage,
            verbose=False,
            max_iterations=3,
        )
        result = agent.run("Тема")
        self.assertEqual(result["iterations"], 3)


class TestIntegrationStorageAndFormatter(unittest.TestCase):
    """Интеграционные тесты: storage → formatter → экспорт."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = PostStorage(base_dir=self.tmp)

    def test_export_markdown(self):
        post_id = self.storage.save({
            "title": "Экспорт тест",
            "content": "Содержание поста",
            "post_type": "breaking",
            "tags": ["тест"],
        })
        result = self.storage.export_post(post_id, fmt="markdown")
        self.assertIsNotNone(result)
        self.assertIn("# Экспорт тест", result)

    def test_export_html(self):
        post_id = self.storage.save({
            "title": "HTML пост",
            "content": "Первый абзац.\n\nВторой абзац.",
            "post_type": "analysis",
        })
        result = self.storage.export_post(post_id, fmt="html")
        self.assertIn("<!DOCTYPE html>", result)
        self.assertIn("HTML пост", result)

    def test_export_json(self):
        post_id = self.storage.save({
            "title": "JSON пост",
            "content": "Контент",
            "post_type": "social",
            "tags": ["json", "тест"],
        })
        result = self.storage.export_post(post_id, fmt="json")
        data = json.loads(result)
        self.assertEqual(data["title"], "JSON пост")
        self.assertIn("json", data["tags"])

    def test_export_nonexistent(self):
        result = self.storage.export_post("nonexistent_id", fmt="markdown")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
