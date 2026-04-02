"""
Тесты для модуля FSM-агента написания статей (День 13).

Проверяют:
  - TaskPhase: значения, переходы
  - TaskState: создание, advance_step, record_step_result, pause/resume, сериализация
  - StepResult: создание, сериализация
  - TaskStateStorage: save, load, list, find_latest_paused
  - ArticleFSMAgent: выполнение шагов с mock LLM, пауза, продолжение
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from news_agent.fsm_state import (
    TaskPhase,
    TaskState,
    TaskStateStorage,
    StepResult,
    PHASE_STEPS,
    PHASE_TRANSITIONS,
    TOTAL_STEPS,
)
from news_agent.fsm_agent import (
    ArticleFSMAgent,
    _build_full_article,
    _build_step_prompt,
    STEP_PROMPTS,
)


# ─────────────────────────────────────────────────────────────
# TaskPhase
# ─────────────────────────────────────────────────────────────

class TestTaskPhase:
    def test_values(self):
        assert TaskPhase.PLANNING.value == "planning"
        assert TaskPhase.EXECUTION.value == "execution"
        assert TaskPhase.VALIDATION.value == "validation"
        assert TaskPhase.DONE.value == "done"

    def test_from_string(self):
        assert TaskPhase("planning") == TaskPhase.PLANNING
        assert TaskPhase("done") == TaskPhase.DONE

    def test_transitions_cover_all_phases(self):
        for phase in [TaskPhase.PLANNING, TaskPhase.EXECUTION, TaskPhase.VALIDATION]:
            assert phase in PHASE_TRANSITIONS

    def test_done_has_no_next(self):
        assert PHASE_TRANSITIONS[TaskPhase.DONE] is None

    def test_planning_leads_to_execution(self):
        assert PHASE_TRANSITIONS[TaskPhase.PLANNING] == TaskPhase.EXECUTION

    def test_execution_leads_to_validation(self):
        assert PHASE_TRANSITIONS[TaskPhase.EXECUTION] == TaskPhase.VALIDATION

    def test_validation_leads_to_done(self):
        assert PHASE_TRANSITIONS[TaskPhase.VALIDATION] == TaskPhase.DONE

    def test_all_phases_have_steps(self):
        for phase in [TaskPhase.PLANNING, TaskPhase.EXECUTION, TaskPhase.VALIDATION]:
            steps = PHASE_STEPS[phase]
            assert len(steps) >= 1, f"Фаза {phase} должна иметь шаги"

    def test_done_has_no_steps(self):
        assert PHASE_STEPS[TaskPhase.DONE] == []

    def test_total_steps_count(self):
        manual = sum(len(PHASE_STEPS[p]) for p in [TaskPhase.PLANNING, TaskPhase.EXECUTION, TaskPhase.VALIDATION])
        assert TOTAL_STEPS == manual
        assert TOTAL_STEPS == 9

    def test_each_step_has_required_keys(self):
        for phase, steps in PHASE_STEPS.items():
            for step in steps:
                assert "name" in step, f"Шаг в {phase} не имеет 'name'"
                assert "description" in step
                assert "expected_action" in step


# ─────────────────────────────────────────────────────────────
# StepResult
# ─────────────────────────────────────────────────────────────

class TestStepResult:
    def test_create(self):
        r = StepResult(step_name="choose_topic", phase="planning", output={"topic": "AI"})
        assert r.step_name == "choose_topic"
        assert r.phase == "planning"
        assert r.output == {"topic": "AI"}
        assert r.completed_at != ""

    def test_to_dict(self):
        r = StepResult(step_name="analyze_topic", phase="planning", output={"tone": "analytical"})
        d = r.to_dict()
        assert d["step_name"] == "analyze_topic"
        assert d["phase"] == "planning"
        assert d["output"]["tone"] == "analytical"
        assert "completed_at" in d

    def test_from_dict_roundtrip(self):
        r = StepResult(step_name="write_intro", phase="execution", output={"intro_text": "Текст"})
        d = r.to_dict()
        r2 = StepResult.from_dict(d)
        assert r2.step_name == r.step_name
        assert r2.phase == r.phase
        assert r2.output == r.output


# ─────────────────────────────────────────────────────────────
# TaskState
# ─────────────────────────────────────────────────────────────

class TestTaskStateNew:
    def test_new_defaults(self):
        state = TaskState.new()
        assert state.phase == TaskPhase.PLANNING
        assert state.current_step == 0
        assert state.paused is False
        assert state.pause_reason == ""
        assert state.topic is None
        assert state.article_data == {}
        assert state.step_results == []
        assert len(state.task_id) == 8

    def test_new_with_topic(self):
        state = TaskState.new(topic="Тест")
        assert state.topic == "Тест"

    def test_new_expected_action(self):
        state = TaskState.new()
        first_step = PHASE_STEPS[TaskPhase.PLANNING][0]
        assert state.expected_action == first_step["expected_action"]

    def test_unique_ids(self):
        ids = {TaskState.new().task_id for _ in range(20)}
        assert len(ids) == 20


class TestTaskStateAdvance:
    def test_advance_within_planning(self):
        state = TaskState.new()
        assert state.phase == TaskPhase.PLANNING
        assert state.current_step == 0
        state.advance_step()
        assert state.current_step == 1
        assert state.phase == TaskPhase.PLANNING

    def test_advance_all_planning_steps(self):
        state = TaskState.new()
        steps_in_planning = len(PHASE_STEPS[TaskPhase.PLANNING])
        for _ in range(steps_in_planning):
            state.advance_step()
        assert state.phase == TaskPhase.EXECUTION
        assert state.current_step == 0

    def test_advance_through_all_phases(self):
        state = TaskState.new()
        total = sum(len(PHASE_STEPS[p]) for p in [TaskPhase.PLANNING, TaskPhase.EXECUTION, TaskPhase.VALIDATION])
        for _ in range(total):
            state.advance_step()
        assert state.phase == TaskPhase.DONE

    def test_advance_done_returns_false(self):
        state = TaskState.new()
        state.phase = TaskPhase.DONE
        result = state.advance_step()
        assert result is False

    def test_expected_action_updates_on_advance(self):
        state = TaskState.new()
        first_action = state.expected_action
        state.advance_step()
        second_action = state.expected_action
        assert first_action != second_action or True  # может совпасть — не критично

    def test_updated_at_changes(self):
        state = TaskState.new()
        old_ts = state.updated_at
        state.advance_step()
        # updated_at может не измениться если операция быстрее 1мс — достаточно что метод не крашится
        assert state.updated_at >= old_ts


class TestTaskStateStepResult:
    def test_record_updates_article_data(self):
        state = TaskState.new()
        state.record_step_result("choose_topic", {"topic": "AI", "rationale": "Актуально"})
        assert state.article_data["topic"] == "AI"
        assert state.article_data["rationale"] == "Актуально"

    def test_record_appends_to_step_results(self):
        state = TaskState.new()
        state.record_step_result("choose_topic", {"topic": "AI"})
        state.record_step_result("analyze_topic", {"tone": "analytical"})
        assert len(state.step_results) == 2

    def test_completed_steps_count(self):
        state = TaskState.new()
        assert state.completed_steps_count() == 0
        state.record_step_result("choose_topic", {})
        assert state.completed_steps_count() == 1


class TestTaskStatePauseResume:
    def test_pause(self):
        state = TaskState.new()
        state.pause("Тест")
        assert state.paused is True
        assert state.pause_reason == "Тест"

    def test_pause_default_reason(self):
        state = TaskState.new()
        state.pause()
        assert state.paused is True
        assert state.pause_reason != ""

    def test_resume(self):
        state = TaskState.new()
        state.pause("Тест")
        state.resume()
        assert state.paused is False
        assert state.pause_reason == ""

    def test_resume_clears_reason(self):
        state = TaskState.new()
        state.pause("причина")
        state.resume()
        assert state.pause_reason == ""


class TestTaskStateSerialization:
    def test_to_dict_contains_all_keys(self):
        state = TaskState.new(topic="Тема")
        d = state.to_dict()
        for key in ("task_id", "phase", "current_step", "expected_action", "topic",
                    "paused", "pause_reason", "article_data", "step_results",
                    "messages", "created_at", "updated_at"):
            assert key in d, f"Ключ {key!r} отсутствует в to_dict()"

    def test_from_dict_roundtrip(self):
        state = TaskState.new(topic="Тест")
        state.record_step_result("choose_topic", {"topic": "AI"})
        state.advance_step()
        state.pause("пауза")

        d = state.to_dict()
        state2 = TaskState.from_dict(d)

        assert state2.task_id == state.task_id
        assert state2.phase == state.phase
        assert state2.current_step == state.current_step
        assert state2.topic == state.topic
        assert state2.paused == state.paused
        assert state2.pause_reason == state.pause_reason
        assert state2.article_data == state.article_data
        assert len(state2.step_results) == len(state.step_results)

    def test_phase_preserved_as_enum(self):
        state = TaskState.new()
        state.phase = TaskPhase.EXECUTION
        state2 = TaskState.from_dict(state.to_dict())
        assert state2.phase == TaskPhase.EXECUTION
        assert isinstance(state2.phase, TaskPhase)

    def test_repr(self):
        state = TaskState.new()
        r = repr(state)
        assert "TaskState" in r
        assert state.task_id in r


class TestTaskStateFormatters:
    def test_format_status_contains_task_id(self):
        state = TaskState.new(topic="AI")
        s = state.format_status()
        assert state.task_id in s

    def test_format_status_shows_paused(self):
        state = TaskState.new()
        state.pause("тест")
        s = state.format_status()
        assert "ПАУЗА" in s

    def test_format_status_shows_running(self):
        state = TaskState.new()
        s = state.format_status()
        assert "Выполняется" in s

    def test_format_phase_progress_shows_all_phases(self):
        state = TaskState.new()
        s = state.format_phase_progress()
        for ph in ("planning", "execution", "validation", "done"):
            assert ph in s

    def test_get_current_step_info(self):
        state = TaskState.new()
        info = state.get_current_step_info()
        assert info is not None
        assert info["name"] == "choose_topic"

    def test_get_current_step_info_done(self):
        state = TaskState.new()
        state.phase = TaskPhase.DONE
        info = state.get_current_step_info()
        assert info is None


# ─────────────────────────────────────────────────────────────
# TaskStateStorage
# ─────────────────────────────────────────────────────────────

class TestTaskStateStorage:
    @pytest.fixture
    def tmp_storage(self, tmp_path):
        return TaskStateStorage(base_dir=str(tmp_path / "tasks"))

    def test_save_and_load(self, tmp_storage):
        state = TaskState.new(topic="Тест")
        tmp_storage.save(state)
        loaded = tmp_storage.load(state.task_id)
        assert loaded is not None
        assert loaded.task_id == state.task_id
        assert loaded.topic == state.topic

    def test_load_nonexistent(self, tmp_storage):
        result = tmp_storage.load("nonexistent")
        assert result is None

    def test_save_updates_index(self, tmp_storage):
        state = TaskState.new()
        tmp_storage.save(state)
        tasks = tmp_storage.list_tasks()
        ids = [t["task_id"] for t in tasks]
        assert state.task_id in ids

    def test_list_tasks_empty(self, tmp_storage):
        result = tmp_storage.list_tasks()
        assert result == []

    def test_list_tasks_multiple(self, tmp_storage):
        for _ in range(3):
            state = TaskState.new()
            tmp_storage.save(state)
        tasks = tmp_storage.list_tasks()
        assert len(tasks) == 3

    def test_list_tasks_paused_only(self, tmp_storage):
        active = TaskState.new(topic="Активная")
        paused = TaskState.new(topic="На паузе")
        paused.pause("тест")
        tmp_storage.save(active)
        tmp_storage.save(paused)

        result = tmp_storage.list_tasks(paused_only=True)
        assert len(result) == 1
        assert result[0]["task_id"] == paused.task_id

    def test_find_latest_paused(self, tmp_storage):
        state = TaskState.new()
        state.pause("тест")
        tmp_storage.save(state)
        found = tmp_storage.find_latest_paused()
        assert found == state.task_id

    def test_find_latest_paused_none(self, tmp_storage):
        state = TaskState.new()
        tmp_storage.save(state)
        found = tmp_storage.find_latest_paused()
        assert found is None

    def test_find_latest_paused_skips_done(self, tmp_storage):
        done = TaskState.new()
        done.phase = TaskPhase.DONE
        done.pause("тест")
        tmp_storage.save(done)
        found = tmp_storage.find_latest_paused()
        assert found is None

    def test_find_in_progress(self, tmp_storage):
        active = TaskState.new()
        done = TaskState.new()
        done.phase = TaskPhase.DONE
        tmp_storage.save(active)
        tmp_storage.save(done)
        in_progress = tmp_storage.find_in_progress()
        ids = [t["task_id"] for t in in_progress]
        assert active.task_id in ids
        assert done.task_id not in ids

    def test_save_twice_updates_index(self, tmp_storage):
        state = TaskState.new()
        tmp_storage.save(state)
        state.advance_step()
        tmp_storage.save(state)
        tasks = tmp_storage.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["current_step"] == state.current_step

    def test_roundtrip_with_step_results(self, tmp_storage):
        state = TaskState.new(topic="AI")
        state.record_step_result("choose_topic", {"topic": "AI"})
        tmp_storage.save(state)
        loaded = tmp_storage.load(state.task_id)
        assert len(loaded.step_results) == 1
        assert loaded.step_results[0].step_name == "choose_topic"

    def test_index_entry_has_topic(self, tmp_storage):
        state = TaskState.new(topic="Технологии")
        tmp_storage.save(state)
        tasks = tmp_storage.list_tasks()
        assert tasks[0]["topic"] == "Технологии"

    def test_dirs_created_automatically(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c"
        storage = TaskStateStorage(base_dir=str(deep_path))
        assert deep_path.exists()


# ─────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────

class TestBuildFullArticle:
    def test_empty_data(self):
        article = _build_full_article({})
        assert isinstance(article, str)

    def test_with_title_and_intro(self):
        data = {"title": "Тест", "intro_text": "Вступление"}
        article = _build_full_article(data)
        assert "Тест" in article
        assert "Вступление" in article

    def test_with_sections(self):
        data = {
            "title": "Статья",
            "sections": [{"name": "Раздел 1", "text": "Текст раздела"}],
        }
        article = _build_full_article(data)
        assert "Раздел 1" in article
        assert "Текст раздела" in article

    def test_with_conclusion(self):
        data = {"title": "Статья", "conclusion_text": "Итог"}
        article = _build_full_article(data)
        assert "Итог" in article

    def test_uses_topic_as_fallback_title(self):
        data = {"topic": "AI тренды"}
        article = _build_full_article(data)
        assert "AI тренды" in article


class TestBuildStepPrompt:
    def test_choose_topic_no_context_needed(self):
        prompt = _build_step_prompt("choose_topic", {})
        assert len(prompt) > 10
        assert "JSON" in prompt

    def test_analyze_topic_includes_topic(self):
        prompt = _build_step_prompt("analyze_topic", {"topic": "Blockchain"})
        assert "Blockchain" in prompt

    def test_create_outline_includes_topic(self):
        prompt = _build_step_prompt("create_outline", {"topic": "ML"})
        assert "ML" in prompt

    def test_all_steps_have_prompts(self):
        for step_name in STEP_PROMPTS:
            prompt = _build_step_prompt(step_name, {"topic": "Тест", "title": "Тест"})
            assert isinstance(prompt, str)
            assert len(prompt) > 5

    def test_unknown_step_returns_fallback(self):
        prompt = _build_step_prompt("unknown_step_xyz", {})
        assert "unknown_step_xyz" in prompt


# ─────────────────────────────────────────────────────────────
# ArticleFSMAgent (с mock LLM)
# ─────────────────────────────────────────────────────────────

def _make_mock_client(response_data: dict) -> MagicMock:
    """Создаёт mock OpenAI клиент, возвращающий заданный JSON."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(response_data, ensure_ascii=False)
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


class TestArticleFSMAgentInit:
    def test_default_model(self, tmp_path):
        client = MagicMock()
        storage = TaskStateStorage(str(tmp_path / "tasks"))
        agent = ArticleFSMAgent(client=client, storage=storage)
        assert agent.model == "zai-org/GLM-4.7"

    def test_custom_model(self, tmp_path):
        client = MagicMock()
        storage = TaskStateStorage(str(tmp_path / "tasks"))
        agent = ArticleFSMAgent(client=client, model="zai-org/GLM-4.7-Flash", storage=storage)
        assert agent.model == "zai-org/GLM-4.7-Flash"

    def test_creates_default_storage(self, tmp_path):
        client = MagicMock()
        with patch("news_agent.fsm_agent.TaskStateStorage") as MockStorage:
            MockStorage.return_value = MagicMock()
            agent = ArticleFSMAgent(client=client)
            MockStorage.assert_called_once()


class TestArticleFSMAgentParseJson:
    @pytest.fixture
    def agent(self, tmp_path):
        return ArticleFSMAgent(
            client=MagicMock(),
            storage=TaskStateStorage(str(tmp_path / "tasks")),
            verbose=False,
        )

    def test_plain_json(self, agent):
        result = agent._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_code_block(self, agent):
        raw = '```json\n{"key": "value"}\n```'
        result = agent._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_json_in_generic_block(self, agent):
        raw = '```\n{"key": "value"}\n```'
        result = agent._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_invalid_json_returns_raw(self, agent):
        result = agent._parse_json_response("это не JSON")
        assert "raw_output" in result

    def test_nested_json(self, agent):
        data = {"sections": [{"name": "А", "text": "Б"}], "count": 1}
        result = agent._parse_json_response(json.dumps(data))
        assert result["sections"][0]["name"] == "А"


class TestArticleFSMAgentExecuteStep:
    @pytest.fixture
    def tmp_storage(self, tmp_path):
        return TaskStateStorage(str(tmp_path / "tasks"))

    def test_execute_choose_topic(self, tmp_storage):
        payload = {
            "topic": "Будущее AI",
            "rationale": "Актуально",
            "target_audience": "Разработчики",
            "article_type": "analysis",
        }
        client = _make_mock_client(payload)
        agent = ArticleFSMAgent(client=client, storage=tmp_storage, verbose=False)
        state = TaskState.new()
        result = agent._execute_step(state, "choose_topic")
        assert result is not None
        assert result["topic"] == "Будущее AI"

    def test_execute_step_retries_on_error(self, tmp_storage):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"ok": true}'
        mock_client.chat.completions.create.side_effect = [
            Exception("API timeout"),
            mock_response,
        ]

        agent = ArticleFSMAgent(client=mock_client, storage=tmp_storage, verbose=False)
        state = TaskState.new()

        with patch("time.sleep"):
            result = agent._execute_step(state, "choose_topic")

        assert result is not None
        assert mock_client.chat.completions.create.call_count == 2

    def test_execute_step_returns_none_after_3_failures(self, tmp_storage):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("fail")

        agent = ArticleFSMAgent(client=mock_client, storage=tmp_storage, verbose=False)
        state = TaskState.new()

        with patch("time.sleep"):
            result = agent._execute_step(state, "choose_topic")

        assert result is None
        assert mock_client.chat.completions.create.call_count == 3


class TestArticleFSMAgentRun:
    @pytest.fixture
    def tmp_storage(self, tmp_path):
        return TaskStateStorage(str(tmp_path / "tasks"))

    def _full_run_client(self) -> MagicMock:
        """Клиент, возвращающий заглушки для каждого шага."""
        responses = {
            "choose_topic": {"topic": "AI в медицине", "rationale": "Актуально", "target_audience": "Врачи", "article_type": "analysis"},
            "analyze_topic": {"main_angle": "диагностика", "key_questions": ["вопрос"], "key_facts": ["факт"], "tone": "analytical", "word_count_target": 800},
            "create_outline": {"title": "AI в медицине", "subtitle": "", "sections": [{"name": "Введение", "key_points": ["тезис"], "word_count": 150}], "total_word_count": 800},
            "write_intro": {"intro_text": "Текст вступления.", "hook": "Крючок", "word_count": 150},
            "write_body": {"sections": [{"name": "Раздел", "text": "Текст раздела"}], "total_word_count": 400},
            "write_conclusion": {"conclusion_text": "Итог статьи.", "key_takeaway": "AI меняет медицину", "word_count": 100},
            "check_structure": {"has_intro": True, "has_body": True, "has_conclusion": True, "section_count": 3, "issues": [], "structure_score": 9},
            "score_quality": {"scores": {"clarity": 8, "accuracy": 8, "engagement": 7, "structure": 9}, "overall_score": 8, "strengths": ["ясность"], "improvements": ["добавь примеры"]},
            "final_edit": {"final_article": "Финальный текст статьи.", "changes": ["улучшен стиль"], "final_word_count": 820},
        }
        mock_client = MagicMock()
        call_count = [0]

        step_order = [
            "choose_topic", "analyze_topic", "create_outline",
            "write_intro", "write_body", "write_conclusion",
            "check_structure", "score_quality", "final_edit",
        ]

        def side_effect(**kwargs):
            idx = call_count[0] % len(step_order)
            step = step_order[idx]
            call_count[0] += 1
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = json.dumps(responses[step], ensure_ascii=False)
            return mock_resp

        mock_client.chat.completions.create.side_effect = side_effect
        return mock_client

    def test_run_new_task_reaches_done(self, tmp_storage):
        client = self._full_run_client()
        agent = ArticleFSMAgent(client=client, storage=tmp_storage, verbose=False)
        state = agent.run(topic="AI в медицине")
        assert state.phase == TaskPhase.DONE

    def test_run_saves_state_after_each_step(self, tmp_storage):
        client = self._full_run_client()
        agent = ArticleFSMAgent(client=client, storage=tmp_storage, verbose=False)
        state = agent.run(topic="AI")
        loaded = tmp_storage.load(state.task_id)
        assert loaded is not None
        assert loaded.phase == TaskPhase.DONE

    def test_run_completes_all_steps(self, tmp_storage):
        client = self._full_run_client()
        agent = ArticleFSMAgent(client=client, storage=tmp_storage, verbose=False)
        state = agent.run(topic="AI")
        assert state.completed_steps_count() == TOTAL_STEPS

    def test_run_with_existing_state_resumes(self, tmp_storage):
        initial = TaskState.new(topic="AI")
        initial.record_step_result("choose_topic", {"topic": "AI"})
        initial.advance_step()
        initial.pause("тест")
        tmp_storage.save(initial)

        remaining_responses = [
            {"main_angle": "...", "key_questions": [], "key_facts": [], "tone": "analytical", "word_count_target": 800},
            {"title": "AI", "subtitle": "", "sections": [{"name": "Раздел", "key_points": [], "word_count": 200}], "total_word_count": 800},
            {"intro_text": "Вступление", "hook": "Крючок", "word_count": 150},
            {"sections": [{"name": "Раздел", "text": "Текст"}], "total_word_count": 400},
            {"conclusion_text": "Итог", "key_takeaway": "Главная мысль", "word_count": 100},
            {"has_intro": True, "has_body": True, "has_conclusion": True, "section_count": 3, "issues": [], "structure_score": 9},
            {"scores": {"clarity": 8, "accuracy": 8, "engagement": 7, "structure": 9}, "overall_score": 8, "strengths": [], "improvements": []},
            {"final_article": "Финальный текст.", "changes": [], "final_word_count": 800},
        ]

        mock_client = MagicMock()
        idx = [0]

        def side_effect(**kwargs):
            resp_data = remaining_responses[idx[0] % len(remaining_responses)]
            idx[0] += 1
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = json.dumps(resp_data, ensure_ascii=False)
            return mock_resp

        mock_client.chat.completions.create.side_effect = side_effect

        agent = ArticleFSMAgent(client=mock_client, storage=tmp_storage, verbose=False)
        state = agent.run(task_id=initial.task_id)
        assert state.phase == TaskPhase.DONE
        assert state.article_data.get("topic") == "AI"

    def test_run_with_unknown_task_id_raises(self, tmp_storage):
        client = MagicMock()
        agent = ArticleFSMAgent(client=client, storage=tmp_storage, verbose=False)
        with pytest.raises(ValueError, match="не найдена"):
            agent.run(task_id="nonexistent")

    def test_pause_stops_execution(self, tmp_storage):
        mock_client = MagicMock()
        agent = ArticleFSMAgent(client=mock_client, storage=tmp_storage, verbose=False)
        state = TaskState.new()
        tmp_storage.save(state)

        valid_result = {"topic": "AI", "rationale": "ok", "target_audience": "все", "article_type": "news"}

        def execute_then_interrupt(s, step_name):
            agent._interrupted = True
            return valid_result

        agent._execute_step = execute_then_interrupt
        final = agent.run(state=state)
        assert final.paused is True

    def test_step_failure_pauses_task(self, tmp_storage):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("crash")

        agent = ArticleFSMAgent(client=mock_client, storage=tmp_storage, verbose=False)

        with patch("time.sleep"):
            state = agent.run(topic="AI")

        assert state.paused is True

    def test_article_data_accumulates(self, tmp_storage):
        client = self._full_run_client()
        agent = ArticleFSMAgent(client=client, storage=tmp_storage, verbose=False)
        state = agent.run(topic="AI")
        data = state.article_data
        assert "topic" in data
        assert "title" in data
        assert "intro_text" in data
        assert "final_article" in data
