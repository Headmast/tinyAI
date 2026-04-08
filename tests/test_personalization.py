"""
Тесты для модуля персонализации (День 12).

Проверяют:
  - UserProfile: создание, сериализация, build_prompt_section
  - ProfileManager: CRUD, встроенные профили, активный профиль
  - PersonalizedAgent: интеграция профиля в system prompt, смена профиля
  - MemoryManager: инъекция секции профиля в build_system_prompt
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memory_agent.memory import MemoryManager
from memory_agent.profile import (
    UserProfile,
    ProfileManager,
    BUILTIN_PROFILES,
)
from memory_agent.personalized_agent import (
    PersonalizedAgent,
    PERSONALIZED_SYSTEM_PROMPT,
)
from memory_agent.agent import MEMORY_BLOCK_PATTERN


# ─────────────────────────────────────────────────────────────
# UserProfile
# ─────────────────────────────────────────────────────────────

class TestUserProfile:
    """Тесты для UserProfile dataclass."""

    def test_create_minimal(self):
        """Минимальный профиль — только обязательные поля."""
        p = UserProfile(name="Тест", role="Тестер", description="Описание")
        assert p.name == "Тест"
        assert p.role == "Тестер"
        assert p.description == "Описание"
        assert p.constraints == []
        assert p.expertise_areas == []

    def test_create_full(self):
        """Полный профиль со всеми полями."""
        p = UserProfile(
            name="Аналитик",
            role="Системный аналитик",
            description="Опытный",
            response_style="Формальный",
            tone="деловой",
            language_level="профессиональный",
            preferred_format="списки",
            structure_preferences=["резюме", "выводы"],
            constraints=["без жаргона"],
            expertise_areas=["UML", "BPMN"],
            custom_preferences={"глубина": "максимальная"},
        )
        assert len(p.structure_preferences) == 2
        assert len(p.constraints) == 1
        assert "UML" in p.expertise_areas
        assert p.custom_preferences["глубина"] == "максимальная"

    def test_build_prompt_section_contains_fields(self):
        """build_prompt_section включает все заданные поля."""
        p = UserProfile(
            name="Тест",
            role="QA",
            description="Тестировщик",
            response_style="краткий",
            tone="нейтральный",
            preferred_format="списки",
            constraints=["без эмодзи", "формальный стиль"],
            expertise_areas=["тестирование", "автоматизация"],
        )
        section = p.build_prompt_section()

        assert "Тест" in section
        assert "QA" in section
        assert "краткий" in section
        assert "нейтральный" in section
        assert "без эмодзи" in section
        assert "тестирование" in section

    def test_build_prompt_section_empty_fields(self):
        """Пустые необязательные поля не попадают в prompt."""
        p = UserProfile(name="Тест", role="Dev", description="")
        section = p.build_prompt_section()
        assert "Стиль ответов" not in section
        assert "Тон общения" not in section
        assert "Ограничения" not in section

    def test_to_dict_from_dict_roundtrip(self):
        """Сериализация → десериализация сохраняет данные."""
        original = UserProfile(
            name="Roundtrip",
            role="Tester",
            description="Test",
            response_style="formal",
            constraints=["c1", "c2"],
            expertise_areas=["e1"],
            custom_preferences={"k": "v"},
        )
        data = original.to_dict()
        restored = UserProfile.from_dict(data)

        assert restored.name == original.name
        assert restored.role == original.role
        assert restored.constraints == original.constraints
        assert restored.expertise_areas == original.expertise_areas
        assert restored.custom_preferences == original.custom_preferences

    def test_update(self):
        """update() изменяет указанные поля."""
        p = UserProfile(name="Old", role="Dev", description="")
        p.update(name="New", tone="casual")
        assert p.name == "New"
        assert p.tone == "casual"

    def test_from_dict_ignores_extra_keys(self):
        """from_dict игнорирует неизвестные ключи."""
        data = {"name": "X", "role": "Y", "description": "Z", "unknown_field": 42}
        p = UserProfile.from_dict(data)
        assert p.name == "X"
        assert not hasattr(p, "unknown_field") or p.__dict__.get("unknown_field") is None


# ─────────────────────────────────────────────────────────────
# BUILTIN_PROFILES
# ─────────────────────────────────────────────────────────────

class TestBuiltinProfiles:
    """Тесты встроенных профилей."""

    def test_system_analyst_exists(self):
        assert "system_analyst" in BUILTIN_PROFILES
        p = BUILTIN_PROFILES["system_analyst"]
        assert p.name == "Системный аналитик"
        assert len(p.constraints) > 0
        assert len(p.expertise_areas) > 0

    def test_journalist_exists(self):
        assert "journalist" in BUILTIN_PROFILES
        p = BUILTIN_PROFILES["journalist"]
        assert p.name == "Журналист"
        assert len(p.constraints) > 0
        assert len(p.expertise_areas) > 0

    def test_profiles_have_different_styles(self):
        """Два профиля имеют разные стили ответов."""
        analyst = BUILTIN_PROFILES["system_analyst"]
        journalist = BUILTIN_PROFILES["journalist"]
        assert analyst.response_style != journalist.response_style
        assert analyst.tone != journalist.tone
        assert analyst.preferred_format != journalist.preferred_format

    def test_profiles_prompt_sections_differ(self):
        """Секции для system prompt существенно различаются."""
        a_section = BUILTIN_PROFILES["system_analyst"].build_prompt_section()
        j_section = BUILTIN_PROFILES["journalist"].build_prompt_section()
        assert a_section != j_section
        assert len(a_section) > 100
        assert len(j_section) > 100


# ─────────────────────────────────────────────────────────────
# ProfileManager
# ─────────────────────────────────────────────────────────────

class TestProfileManager:
    """Тесты ProfileManager."""

    def test_get_builtin_profile(self):
        """Встроенные профили доступны без файла."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as f:
            pm = ProfileManager(storage_path=f.name)
            p = pm.get_profile("system_analyst")
            assert p is not None
            assert p.name == "Системный аналитик"

    def test_add_and_get_custom_profile(self):
        """Добавление и чтение пользовательского профиля."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            pm = ProfileManager(storage_path=path)
            custom = UserProfile(name="Custom", role="Custom Role", description="Test")
            pm.add_profile("my_profile", custom)

            retrieved = pm.get_profile("my_profile")
            assert retrieved is not None
            assert retrieved.name == "Custom"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_set_active_profile(self):
        """Установка и получение активного профиля."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            pm = ProfileManager(storage_path=path)
            assert pm.set_active("system_analyst") is True
            active = pm.get_active_profile()
            assert active is not None
            assert active.name == "Системный аналитик"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_set_active_nonexistent(self):
        """Попытка установить несуществующий профиль."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as f:
            pm = ProfileManager(storage_path=f.name)
            assert pm.set_active("nonexistent") is False

    def test_remove_profile(self):
        """Удаление пользовательского профиля."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            pm = ProfileManager(storage_path=path)
            custom = UserProfile(name="ToRemove", role="R", description="")
            pm.add_profile("removable", custom)
            assert pm.remove_profile("removable") is True
            assert pm.get_profile("removable") is None
        finally:
            Path(path).unlink(missing_ok=True)

    def test_remove_builtin_returns_false(self):
        """Встроенные профили нельзя удалить через remove_profile."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as f:
            pm = ProfileManager(storage_path=f.name)
            assert pm.remove_profile("system_analyst") is False

    def test_list_profiles(self):
        """list_profiles возвращает все профили."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as f:
            pm = ProfileManager(storage_path=f.name)
            profiles = pm.list_profiles()
            assert "system_analyst" in profiles
            assert "journalist" in profiles
            assert profiles["system_analyst"]["type"] == "builtin"

    def test_persistence(self):
        """Профили сохраняются и загружаются из файла."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            pm1 = ProfileManager(storage_path=path)
            custom = UserProfile(name="Persistent", role="P", description="")
            pm1.add_profile("persistent", custom)
            pm1.set_active("persistent")

            pm2 = ProfileManager(storage_path=path)
            assert pm2.get_profile("persistent") is not None
            assert pm2.active_profile_name == "persistent"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_clear_active(self):
        """clear_active снимает активный профиль."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            pm = ProfileManager(storage_path=path)
            pm.set_active("journalist")
            pm.clear_active()
            assert pm.get_active_profile() is None
        finally:
            Path(path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────
# MemoryManager с профилем
# ─────────────────────────────────────────────────────────────

class TestMemoryManagerProfile:
    """Тесты интеграции профиля в MemoryManager."""

    def test_profile_section_in_system_prompt(self):
        """Секция профиля появляется в build_system_prompt."""
        mm = MemoryManager(long_term_path="/tmp/test_mm_profile.json")
        mm.set_profile_section("Имя: Тест\nРоль: QA")

        prompt = mm.build_system_prompt("Базовый промпт.")
        assert "ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ" in prompt
        assert "Имя: Тест" in prompt
        assert "Базовый промпт." in prompt

    def test_no_profile_section_when_empty(self):
        """Без профиля секция не добавляется."""
        mm = MemoryManager(long_term_path="/tmp/test_mm_noprof.json")
        prompt = mm.build_system_prompt("Base.")
        assert "ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ" not in prompt

    def test_clear_profile_section(self):
        """clear_profile_section убирает секцию."""
        mm = MemoryManager(long_term_path="/tmp/test_mm_clear.json")
        mm.set_profile_section("Данные")
        mm.clear_profile_section()
        prompt = mm.build_system_prompt("Base.")
        assert "ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ" not in prompt


# ─────────────────────────────────────────────────────────────
# PersonalizedAgent
# ─────────────────────────────────────────────────────────────

def _make_mock_client(response_text: str = "Ответ ассистента."):
    """Создаёт mock OpenAI клиент."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = response_text
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    client.chat.completions.create.return_value = mock_response
    return client


class TestPersonalizedAgent:
    """Тесты PersonalizedAgent."""

    def test_create_with_profile(self):
        """Агент создаётся с профилем."""
        client = _make_mock_client()
        profile = BUILTIN_PROFILES["system_analyst"]
        agent = PersonalizedAgent(
            client=client,
            profile=profile,
            verbose=False,
        )
        assert agent.profile is not None
        assert agent.profile.name == "Системный аналитик"

    def test_create_without_profile(self):
        """Агент создаётся без профиля."""
        client = _make_mock_client()
        agent = PersonalizedAgent(client=client, verbose=False)
        assert agent.profile is None

    def test_chat_sends_profile_in_system_prompt(self):
        """При chat() профиль попадает в system prompt."""
        client = _make_mock_client()
        profile = BUILTIN_PROFILES["journalist"]
        mm = MemoryManager(long_term_path="/tmp/test_pa_prompt.json")

        agent = PersonalizedAgent(
            client=client,
            profile=profile,
            memory_manager=mm,
            verbose=False,
        )
        agent.chat("Привет")

        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_msg = messages[0]["content"]

        assert "ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ" in system_msg
        assert "Журналист" in system_msg

    def test_chat_returns_visible_text(self):
        """chat() возвращает видимый текст без блока memory."""
        response_with_memory = (
            "Вот ответ.\n\n```memory\n"
            '{"working": {"facts": ["факт1"]}}\n'
            "```"
        )
        client = _make_mock_client(response_with_memory)
        agent = PersonalizedAgent(client=client, verbose=False)
        result = agent.chat("Вопрос")
        assert "Вот ответ." in result
        assert "```memory" not in result

    def test_memory_block_applied(self):
        """Блок memory из ответа применяется к рабочей памяти."""
        response_with_memory = (
            "Ответ.\n\n```memory\n"
            '{"working": {"facts": ["AI важен"]}, '
            '"long_term": {"knowledge": [{"topic": "AI", "content": "тренд"}]}}\n'
            "```"
        )
        client = _make_mock_client(response_with_memory)
        mm = MemoryManager(long_term_path="/tmp/test_pa_mem.json")
        agent = PersonalizedAgent(
            client=client,
            memory_manager=mm,
            verbose=False,
        )
        agent.chat("Расскажи про AI")

        assert "AI важен" in mm.working.facts
        knowledge = mm.long_term.get_all_knowledge()
        assert len(knowledge) > 0
        assert knowledge[0]["topic"] == "AI"

    def test_set_profile_changes_behavior(self):
        """set_profile меняет профиль на лету."""
        client = _make_mock_client()
        agent = PersonalizedAgent(client=client, verbose=False)

        agent.set_profile(BUILTIN_PROFILES["system_analyst"])
        assert agent.profile.name == "Системный аналитик"

        agent.set_profile(BUILTIN_PROFILES["journalist"])
        assert agent.profile.name == "Журналист"

    def test_clear_profile(self):
        """clear_profile убирает профиль."""
        client = _make_mock_client()
        agent = PersonalizedAgent(
            client=client,
            profile=BUILTIN_PROFILES["system_analyst"],
            verbose=False,
        )
        agent.clear_profile()
        assert agent.profile is None

    def test_get_profile_info_active(self):
        """get_profile_info возвращает данные активного профиля."""
        client = _make_mock_client()
        agent = PersonalizedAgent(
            client=client,
            profile=BUILTIN_PROFILES["journalist"],
            verbose=False,
        )
        info = agent.get_profile_info()
        assert info["active"] is True
        assert info["name"] == "Журналист"
        assert info["constraints_count"] > 0

    def test_get_profile_info_inactive(self):
        """get_profile_info без профиля."""
        client = _make_mock_client()
        agent = PersonalizedAgent(client=client, verbose=False)
        info = agent.get_profile_info()
        assert info["active"] is False

    def test_reset_working_preserves_profile(self):
        """reset_working_memory не сбрасывает профиль."""
        client = _make_mock_client()
        agent = PersonalizedAgent(
            client=client,
            profile=BUILTIN_PROFILES["system_analyst"],
            verbose=False,
        )
        agent.reset_working_memory()
        assert agent.profile is not None
        assert agent.profile.name == "Системный аналитик"

    def test_reset_all_clears_profile(self):
        """reset_all сбрасывает и профиль."""
        client = _make_mock_client()
        agent = PersonalizedAgent(
            client=client,
            profile=BUILTIN_PROFILES["system_analyst"],
            verbose=False,
        )
        agent.reset_all()
        assert agent.profile is None

    def test_token_usage_tracked(self):
        """Токены корректно накапливаются."""
        client = _make_mock_client()
        agent = PersonalizedAgent(client=client, verbose=False)
        agent.chat("Тест 1")
        agent.chat("Тест 2")
        usage = agent.get_token_usage()
        assert usage["total_tokens"] == 300  # 150 * 2

    def test_profile_history(self):
        """История смен профилей записывается."""
        client = _make_mock_client()
        agent = PersonalizedAgent(client=client, verbose=False)
        agent.set_profile(BUILTIN_PROFILES["system_analyst"])
        agent.set_profile(BUILTIN_PROFILES["journalist"])
        agent.clear_profile()

        state = agent.get_memory_state()
        assert len(state["profile_history"]) == 3
        assert state["profile_history"][0]["to"] == "Системный аналитик"
        assert state["profile_history"][1]["to"] == "Журналист"
        assert state["profile_history"][2]["action"] == "clear_profile"


# ─────────────────────────────────────────────────────────────
# MEMORY_BLOCK_PATTERN regex
# ─────────────────────────────────────────────────────────────

class TestMemoryBlockPattern:
    """Тесты regex для извлечения блока memory."""

    def test_extracts_valid_json(self):
        text = 'Ответ.\n\n```memory\n{"working": {"facts": ["f1"]}}\n```'
        match = MEMORY_BLOCK_PATTERN.search(text)
        assert match is not None
        data = json.loads(match.group(1))
        assert data["working"]["facts"] == ["f1"]

    def test_no_match_without_block(self):
        text = "Обычный ответ без блока memory."
        match = MEMORY_BLOCK_PATTERN.search(text)
        assert match is None

    def test_removes_block_from_text(self):
        text = 'Видимый текст.\n\n```memory\n{"working": {}}\n```\nЕщё текст.'
        cleaned = MEMORY_BLOCK_PATTERN.sub("", text).strip()
        assert "```memory" not in cleaned
        assert "Видимый текст." in cleaned
