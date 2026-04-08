"""
Тесты для JournalistFSMAgent.

Проверяются:
  - начальные состояния всех типов контента
  - допустимые переходы
  - недопустимые переходы (перепрыгивание шагов)
  - попытки выйти из терминального состояния
  - попытки перейти из paused без resume()
  - пауза и возобновление задачи
  - сохранение результатов после каждого шага
  - try_transition() — проверка без выполнения
  - все 5 типов контента охватывают верные шаги и переходы
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from journalist_agent.workflow import (
    ContentType,
    JournalistTask,
    JournalistTaskStorage,
    STATE_PAUSED,
    STATE_PUBLISHED,
    TransitionError,
    WORKFLOWS,
)
from journalist_agent.step_prompts import _build_context, build_step_prompt
from journalist_agent.fsm_agent import JournalistFSMAgent


# ─────────────────────────────────────────────────────────────
# Вспомогательные фикстуры
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_storage(tmp_path):
    return JournalistTaskStorage(base_dir=str(tmp_path))


@pytest.fixture
def mock_client():
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps({
        "title": "Тестовый заголовок",
        "sections": [{"name": "Введение", "key_points": ["тезис"], "word_count": 100}],
        "target_audience": "разработчики",
        "tone": "analytical",
        "word_count_target": 800,
    })
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def agent(mock_client, tmp_storage):
    return JournalistFSMAgent(
        client=mock_client,
        storage=tmp_storage,
        verbose=False,
    )


def make_task(content_type: ContentType, topic: str = "тест") -> JournalistTask:
    return JournalistTask.new(content_type, topic=topic)


# ─────────────────────────────────────────────────────────────
# 1. WorkflowConfig — структура переходов
# ─────────────────────────────────────────────────────────────

class TestWorkflowConfig:

    @pytest.mark.parametrize("ct", list(ContentType))
    def test_initial_state_is_first_step(self, ct):
        wf = WORKFLOWS[ct]
        assert wf.initial_state == wf.steps[0].state

    @pytest.mark.parametrize("ct", list(ContentType))
    def test_all_steps_have_unique_states(self, ct):
        wf = WORKFLOWS[ct]
        states = [s.state for s in wf.steps]
        assert len(states) == len(set(states)), "Дублирующиеся состояния в workflow"

    @pytest.mark.parametrize("ct", list(ContentType))
    def test_last_step_is_published(self, ct):
        wf = WORKFLOWS[ct]
        assert wf.steps[-1].state == STATE_PUBLISHED

    @pytest.mark.parametrize("ct", list(ContentType))
    def test_sequential_transitions_allowed(self, ct):
        """Каждый шаг (кроме published) имеет разрешённый следующий шаг."""
        wf = WORKFLOWS[ct]
        for step in wf.steps[:-1]:  # все кроме published
            nexts = wf.allowed_next(step.state)
            assert len(nexts) > 0, f"У шага {step.state!r} нет разрешённых переходов"

    @pytest.mark.parametrize("ct", list(ContentType))
    def test_published_has_no_transitions(self, ct):
        wf = WORKFLOWS[ct]
        assert wf.allowed_next(STATE_PUBLISHED) == []
        assert not wf.can_transition(STATE_PUBLISHED, "anything")

    @pytest.mark.parametrize("ct", list(ContentType))
    def test_paused_has_no_transitions(self, ct):
        wf = WORKFLOWS[ct]
        assert wf.allowed_next(STATE_PAUSED) == []
        assert not wf.can_transition(STATE_PAUSED, wf.initial_state)

    def test_article_full_chain(self):
        wf = WORKFLOWS[ContentType.ARTICLE]
        chain = ["planning", "research", "drafting", "editing", "validation", STATE_PUBLISHED]
        for i in range(len(chain) - 1):
            assert wf.can_transition(chain[i], chain[i + 1]), \
                f"Переход {chain[i]}→{chain[i+1]} должен быть разрешён"

    def test_article_skip_not_allowed(self):
        wf = WORKFLOWS[ContentType.ARTICLE]
        # нельзя перепрыгнуть с planning прямо на drafting (минуя research)
        assert not wf.can_transition("planning", "drafting")
        # нельзя перепрыгнуть с research на editing (минуя drafting)
        assert not wf.can_transition("research", "editing")
        # нельзя сразу в published из editing (минуя validation)
        assert not wf.can_transition("editing", STATE_PUBLISHED)

    def test_note_full_chain(self):
        wf = WORKFLOWS[ContentType.NOTE]
        assert wf.can_transition("idea_capture", "drafting")
        assert wf.can_transition("drafting", STATE_PUBLISHED)

    def test_news_research_chain(self):
        wf = WORKFLOWS[ContentType.NEWS_RESEARCH]
        chain = ["topic_selection", "source_gathering", "analysis", "drafting", "fact_check", STATE_PUBLISHED]
        for i in range(len(chain) - 1):
            assert wf.can_transition(chain[i], chain[i + 1])

    def test_review_chain(self):
        wf = WORKFLOWS[ContentType.REVIEW]
        chain = ["subject_definition", "criteria_setting", "examination", "drafting", "validation", STATE_PUBLISHED]
        for i in range(len(chain) - 1):
            assert wf.can_transition(chain[i], chain[i + 1])

    def test_blog_analysis_chain(self):
        wf = WORKFLOWS[ContentType.BLOG_ANALYSIS]
        chain = [
            "blog_selection", "metrics_collection", "content_analysis",
            "insights_extraction", "drafting", "validation", STATE_PUBLISHED
        ]
        for i in range(len(chain) - 1):
            assert wf.can_transition(chain[i], chain[i + 1])

    def test_pause_allowed_from_any_active_state(self):
        wf = WORKFLOWS[ContentType.ARTICLE]
        for step in wf.steps:
            if step.state == STATE_PUBLISHED:
                assert not wf.can_transition(step.state, STATE_PAUSED)
            else:
                assert wf.can_transition(step.state, STATE_PAUSED)

    def test_auto_next_returns_first_allowed(self):
        wf = WORKFLOWS[ContentType.ARTICLE]
        assert wf.auto_next("planning") == "research"
        assert wf.auto_next("research") == "drafting"
        assert wf.auto_next(STATE_PUBLISHED) is None

    def test_step_index_correct(self):
        wf = WORKFLOWS[ContentType.ARTICLE]
        assert wf.step_index("planning") == 0
        assert wf.step_index("research") == 1
        assert wf.step_index(STATE_PUBLISHED) == 5
        assert wf.step_index("nonexistent") == -1


# ─────────────────────────────────────────────────────────────
# 2. TransitionError
# ─────────────────────────────────────────────────────────────

class TestTransitionError:

    def test_error_contains_from_and_to(self):
        err = TransitionError("planning", "drafting", ["research"])
        assert "planning" in str(err)
        assert "drafting" in str(err)

    def test_error_contains_allowed(self):
        err = TransitionError("planning", "drafting", ["research"])
        assert "research" in str(err)

    def test_error_terminal(self):
        err = TransitionError("published", "planning", [])
        assert "терминальное" in str(err) or "(нет" in str(err)

    def test_error_has_content_type(self):
        err = TransitionError("planning", "drafting", ["research"], ContentType.ARTICLE)
        assert "Статья" in str(err)


# ─────────────────────────────────────────────────────────────
# 3. JournalistTask
# ─────────────────────────────────────────────────────────────

class TestJournalistTask:

    def test_new_task_starts_at_initial_state(self):
        task = make_task(ContentType.ARTICLE)
        assert task.state == "planning"
        assert not task.is_paused
        assert not task.is_done

    def test_new_note_task_starts_at_idea_capture(self):
        task = make_task(ContentType.NOTE)
        assert task.state == "idea_capture"

    def test_task_id_generated(self):
        t1 = make_task(ContentType.ARTICLE)
        t2 = make_task(ContentType.ARTICLE)
        assert t1.task_id != t2.task_id
        assert len(t1.task_id) == 10

    def test_serialization_roundtrip(self):
        task = make_task(ContentType.REVIEW, topic="ChatGPT ревью")
        task.step_results = {"subject_name": "ChatGPT", "score": 9}
        d = task.to_dict()
        restored = JournalistTask.from_dict(d)
        assert restored.task_id == task.task_id
        assert restored.content_type == task.content_type
        assert restored.topic == task.topic
        assert restored.step_results == task.step_results

    def test_active_state_when_paused(self):
        task = make_task(ContentType.ARTICLE)
        task.state_before_pause = "research"
        task.state = STATE_PAUSED
        assert task.active_state == "research"

    def test_active_state_when_not_paused(self):
        task = make_task(ContentType.ARTICLE)
        task.state = "drafting"
        assert task.active_state == "drafting"

    def test_format_status_contains_content_type(self):
        task = make_task(ContentType.ARTICLE, topic="AI тренды")
        status = task.format_status()
        assert "Статья" in status
        assert "AI тренды" in status

    def test_format_progress_marks_completed(self):
        task = make_task(ContentType.ARTICLE)
        task.state = "drafting"
        progress = task.format_progress()
        assert "✓" in progress  # planning и research уже пройдены

    def test_current_step_for_initial_state(self):
        task = make_task(ContentType.ARTICLE)
        step = task.current_step
        assert step is not None
        assert step.state == "planning"


# ─────────────────────────────────────────────────────────────
# 4. JournalistTaskStorage
# ─────────────────────────────────────────────────────────────

class TestJournalistTaskStorage:

    def test_save_and_load(self, tmp_storage):
        task = make_task(ContentType.NOTE, topic="Заметка о Python")
        tmp_storage.save(task)
        loaded = tmp_storage.load(task.task_id)
        assert loaded is not None
        assert loaded.task_id == task.task_id
        assert loaded.topic == "Заметка о Python"

    def test_load_nonexistent_returns_none(self, tmp_storage):
        assert tmp_storage.load("nonexistent_id") is None

    def test_save_is_atomic(self, tmp_storage):
        """После сохранения нет .tmp файлов."""
        task = make_task(ContentType.NOTE)
        tmp_storage.save(task)
        tmp_files = list(tmp_storage.base_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_list_tasks(self, tmp_storage):
        for ct in [ContentType.NOTE, ContentType.REVIEW, ContentType.ARTICLE]:
            tmp_storage.save(make_task(ct, topic=f"Тест {ct.value}"))
        items = tmp_storage.list_tasks()
        assert len(items) == 3

    def test_list_tasks_limit(self, tmp_storage):
        for i in range(5):
            tmp_storage.save(make_task(ContentType.NOTE, topic=f"Заметка {i}"))
        items = tmp_storage.list_tasks(n=3)
        assert len(items) == 3

    def test_update_saved_correctly(self, tmp_storage):
        task = make_task(ContentType.ARTICLE)
        tmp_storage.save(task)
        task.state = "research"
        task.step_results = {"title": "Статья о тестировании"}
        tmp_storage.save(task)
        loaded = tmp_storage.load(task.task_id)
        assert loaded.state == "research"
        assert loaded.step_results["title"] == "Статья о тестировании"


# ─────────────────────────────────────────────────────────────
# 5. JournalistFSMAgent — переходы
# ─────────────────────────────────────────────────────────────

class TestFSMAgentTransitions:

    def test_start_creates_task_in_initial_state(self, agent):
        task = agent.start(ContentType.NOTE, topic="Заметка")
        assert task.state == "idea_capture"
        assert task.topic == "Заметка"

    def test_advance_moves_to_next_state(self, agent, mock_client):
        """Один advance() должен выполнить шаг и перейти к следующему."""
        mock_client.chat.completions.create.return_value.choices[0].message.content = json.dumps({
            "core_idea": "Тестовая идея",
            "key_points": ["пункт 1"],
            "tone": "analytical",
        })
        task = agent.start(ContentType.NOTE, topic="Тест идея")
        task = agent.advance(task)
        assert task.state == "drafting"
        assert "core_idea" in task.step_results

    def test_advance_explicit_valid_transition(self, agent, mock_client):
        mock_client.chat.completions.create.return_value.choices[0].message.content = json.dumps({
            "core_idea": "идея",
            "key_points": [],
            "tone": "analytical",
        })
        task = agent.start(ContentType.NOTE, topic="тема")
        task = agent.advance(task, to_state="drafting")
        assert task.state == "drafting"

    def test_advance_invalid_transition_raises(self, agent):
        """Попытка перепрыгнуть шаг вызывает TransitionError."""
        task = agent.start(ContentType.ARTICLE, topic="AI")
        assert task.state == "planning"
        with pytest.raises(TransitionError) as exc_info:
            agent.advance(task, to_state="drafting")  # пропускает research
        assert "planning" in str(exc_info.value)
        assert "drafting" in str(exc_info.value)

    def test_advance_skip_to_published_raises(self, agent):
        """Нельзя перейти в published, пропустив валидацию."""
        task = agent.start(ContentType.NOTE, topic="заметка")
        with pytest.raises(TransitionError):
            agent.advance(task, to_state=STATE_PUBLISHED)

    def test_advance_on_done_task_raises(self, agent):
        """Нельзя advance() завершённую задачу."""
        task = agent.start(ContentType.NOTE, topic="тема")
        task.state = STATE_PUBLISHED
        agent.storage.save(task)
        with pytest.raises(ValueError, match="завершена"):
            agent.advance(task)

    def test_advance_on_paused_task_raises(self, agent):
        """Нельзя advance() задачу на паузе."""
        task = agent.start(ContentType.NOTE, topic="тема")
        task = agent.pause(task, "тест")
        with pytest.raises(ValueError, match="паузе"):
            agent.advance(task)

    def test_skip_multiple_steps_article(self, agent):
        """Попытка перепрыгнуть несколько шагов сразу."""
        task = agent.start(ContentType.ARTICLE, topic="AI безопасность")
        assert task.state == "planning"
        with pytest.raises(TransitionError):
            agent.advance(task, to_state="editing")   # пропуск research + drafting
        with pytest.raises(TransitionError):
            agent.advance(task, to_state="validation")  # пропуск 3 шагов
        with pytest.raises(TransitionError):
            agent.advance(task, to_state=STATE_PUBLISHED)  # пропуск всех шагов

    def test_skip_news_research_analysis(self, agent):
        """В NEWS_RESEARCH нельзя идти от source_gathering прямо к drafting."""
        task = agent.start(ContentType.NEWS_RESEARCH, topic="NFT")
        task.state = "source_gathering"
        agent.storage.save(task)
        with pytest.raises(TransitionError):
            agent.advance(task, to_state="drafting")  # пропуск analysis

    def test_skip_review_criteria(self, agent):
        """В REVIEW нельзя пропустить criteria_setting."""
        task = agent.start(ContentType.REVIEW, topic="книга")
        assert task.state == "subject_definition"
        with pytest.raises(TransitionError):
            agent.advance(task, to_state="examination")  # пропуск criteria_setting

    def test_blog_analysis_cannot_skip_insights(self, agent):
        """В BLOG_ANALYSIS нельзя прыгнуть от content_analysis к drafting."""
        task = agent.start(ContentType.BLOG_ANALYSIS, topic="блог")
        task.state = "content_analysis"
        agent.storage.save(task)
        with pytest.raises(TransitionError):
            agent.advance(task, to_state="drafting")  # пропуск insights_extraction


# ─────────────────────────────────────────────────────────────
# 6. Пауза и возобновление
# ─────────────────────────────────────────────────────────────

class TestPauseResume:

    def test_pause_saves_current_state(self, agent):
        task = agent.start(ContentType.ARTICLE, topic="Тест")
        task.state = "research"
        agent.storage.save(task)
        task = agent.pause(task, "обед")
        assert task.state == STATE_PAUSED
        assert task.state_before_pause == "research"
        assert task.pause_reason == "обед"

    def test_resume_restores_state(self, agent):
        task = agent.start(ContentType.ARTICLE, topic="Тест")
        task.state = "research"
        agent.storage.save(task)
        task = agent.pause(task)
        assert task.is_paused
        task = agent.resume(task)
        assert task.state == "research"
        assert not task.is_paused
        assert task.state_before_pause is None

    def test_resume_non_paused_raises(self, agent):
        task = agent.start(ContentType.NOTE, topic="заметка")
        with pytest.raises(ValueError, match="не на паузе"):
            agent.resume(task)

    def test_pause_published_raises(self, agent):
        task = agent.start(ContentType.NOTE, topic="заметка")
        task.state = STATE_PUBLISHED
        agent.storage.save(task)
        with pytest.raises(ValueError, match="завершённую"):
            agent.pause(task)

    def test_pause_already_paused_is_noop(self, agent):
        task = agent.start(ContentType.NOTE, topic="заметка")
        task = agent.pause(task, "первый раз")
        task_again = agent.pause(task, "второй раз")
        # должно оставаться первой причиной
        assert task_again.pause_reason == "первый раз"

    def test_advance_after_resume_works(self, agent, mock_client):
        mock_client.chat.completions.create.return_value.choices[0].message.content = json.dumps({
            "core_idea": "идея",
            "key_points": ["п1"],
            "tone": "analytical",
        })
        task = agent.start(ContentType.NOTE, topic="заметка")
        task = agent.pause(task, "тест")
        task = agent.resume(task)
        task = agent.advance(task)
        assert task.state == "drafting"

    def test_storage_persists_pause_state(self, agent, tmp_storage):
        task = agent.start(ContentType.ARTICLE, topic="тест")
        task.state = "editing"
        agent.storage.save(task)
        task = agent.pause(task, "конец дня")
        loaded = tmp_storage.load(task.task_id)
        assert loaded.state == STATE_PAUSED
        assert loaded.state_before_pause == "editing"
        assert loaded.pause_reason == "конец дня"

    def test_resume_from_initial_when_no_before_pause(self, agent):
        """Если state_before_pause не задан — восстанавливаем начальное состояние."""
        task = agent.start(ContentType.NOTE, topic="тест")
        task.state = STATE_PAUSED
        task.state_before_pause = None
        agent.storage.save(task)
        task = agent.resume(task)
        assert task.state == "idea_capture"


# ─────────────────────────────────────────────────────────────
# 7. try_transition() — проверка без выполнения
# ─────────────────────────────────────────────────────────────

class TestTryTransition:

    def test_valid_transition_returns_true(self, agent):
        task = agent.start(ContentType.ARTICLE, topic="тест")
        ok, reason = agent.try_transition(task, "research")
        assert ok is True
        assert "разрешён" in reason

    def test_invalid_transition_returns_false(self, agent):
        task = agent.start(ContentType.ARTICLE, topic="тест")
        ok, reason = agent.try_transition(task, "drafting")  # пропуск research
        assert ok is False
        assert "запрещён" in reason

    def test_paused_task_returns_false(self, agent):
        task = agent.start(ContentType.ARTICLE, topic="тест")
        task = agent.pause(task)
        ok, reason = agent.try_transition(task, "research")
        assert ok is False
        assert "паузе" in reason.lower()

    def test_published_task_returns_false(self, agent):
        task = agent.start(ContentType.NOTE, topic="тест")
        task.state = STATE_PUBLISHED
        ok, reason = agent.try_transition(task, "idea_capture")
        assert ok is False
        assert "завершена" in reason.lower()

    def test_skip_info_mentioned(self, agent):
        task = agent.start(ContentType.ARTICLE, topic="тест")
        ok, reason = agent.try_transition(task, "drafting")
        assert ok is False
        # Должно объяснить что research пропускается
        assert "research" in reason.lower() or "Пропуск" in reason or "пропуска" in reason.lower()


# ─────────────────────────────────────────────────────────────
# 8. Step prompts
# ─────────────────────────────────────────────────────────────

class TestStepPrompts:

    def test_build_step_prompt_returns_string(self):
        task = make_task(ContentType.ARTICLE, topic="Python 2025")
        prompt = build_step_prompt(task, "planning")
        assert isinstance(prompt, str)
        assert len(prompt) > 10

    def test_topic_substituted_in_prompt(self):
        task = make_task(ContentType.ARTICLE, topic="Квантовые компьютеры")
        prompt = build_step_prompt(task, "planning")
        assert "Квантовые компьютеры" in prompt

    def test_context_includes_topic(self):
        task = make_task(ContentType.REVIEW, topic="Книга про Rust")
        ctx = _build_context(task)
        assert ctx["topic"] == "Книга про Rust"

    def test_context_uses_step_results(self):
        task = make_task(ContentType.ARTICLE, topic="тест")
        task.step_results = {"title": "Моя Статья", "tone": "educational"}
        ctx = _build_context(task)
        assert ctx["title"] == "Моя Статья"
        assert ctx["tone"] == "educational"

    def test_all_content_types_have_prompts_for_all_steps(self):
        """Для каждого шага каждого типа есть шаблон промпта."""
        from journalist_agent.step_prompts import STEP_PROMPTS
        for ct in ContentType:
            wf = WORKFLOWS[ct]
            for step in wf.steps:
                if step.state == STATE_PUBLISHED:
                    continue  # published не требует LLM-промпта
                templates = STEP_PROMPTS.get(ct.value, {})
                assert step.state in templates, \
                    f"Нет промпта для {ct.value}.{step.state}"

    def test_missing_placeholder_does_not_crash(self):
        """Промпт с недостающим ключом не бросает исключение."""
        task = make_task(ContentType.ARTICLE, topic="тест")
        # Намеренно пустые step_results
        prompt = build_step_prompt(task, "research")
        assert isinstance(prompt, str)


# ─────────────────────────────────────────────────────────────
# 9. Накопление результатов шагов
# ─────────────────────────────────────────────────────────────

class TestStepResultsAccumulation:

    def test_results_accumulated_across_steps(self, agent, mock_client):
        """step_results пополняются при каждом advance()."""
        mock_client.chat.completions.create.return_value.choices[0].message.content = json.dumps({
            "core_idea": "идея первого шага",
            "key_points": ["п1", "п2"],
            "tone": "analytical",
        })
        task = agent.start(ContentType.NOTE, topic="накопление")
        task = agent.advance(task)  # idea_capture → drafting
        assert "core_idea" in task.step_results

    def test_draft_text_persisted(self, agent, mock_client):
        """Текстовый результат шага сохраняется в draft_text."""
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            "Это текст заметки о тестировании."
        )
        task = make_task(ContentType.NOTE, topic="тест")
        task.state = "drafting"
        task.step_results = {"core_idea": "идея", "key_points": ["п1"], "tone": "analytical"}
        agent.storage.save(task)
        task = agent.advance(task)
        assert "draft_text" in task.step_results or "word_count" in task.step_results


# ─────────────────────────────────────────────────────────────
# 10. ContentType.from_str
# ─────────────────────────────────────────────────────────────

class TestContentTypeFromStr:

    def test_valid_strings(self):
        assert ContentType.from_str("article") == ContentType.ARTICLE
        assert ContentType.from_str("NOTE") == ContentType.NOTE
        assert ContentType.from_str("Blog_Analysis") == ContentType.BLOG_ANALYSIS

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Неизвестный тип"):
            ContentType.from_str("poem")

    def test_labels_defined(self):
        for ct in ContentType:
            label = ct.label()
            assert isinstance(label, str)
            assert len(label) > 0
