"""
Тесты для context_strategies — три стратегии управления контекстом.

Покрывает:
  - SlidingWindowStrategy: отсечение, сохранение system, сериализация
  - StickyFactsStrategy: формирование API-сообщений, парсинг фактов, сериализация
  - BranchingStrategy: checkpoint, branch, switch, leave, сериализация
  - create_strategy / strategy_from_dict — реестр и десериализация
"""

import json
import pytest

from news_agent.context_strategies import (
    ContextStrategy,
    SlidingWindowStrategy,
    StickyFactsStrategy,
    BranchingStrategy,
    create_strategy,
    strategy_from_dict,
)


# ─────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────

def _make_messages(n_user_assistant_pairs: int, system: bool = True):
    """Создаёт тестовый массив messages."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": "Ты помощник."})
    for i in range(1, n_user_assistant_pairs + 1):
        msgs.append({"role": "user", "content": f"Вопрос {i}"})
        msgs.append({"role": "assistant", "content": f"Ответ {i}"})
    return msgs


# ─────────────────────────────────────────────────────────────
# SlidingWindowStrategy
# ─────────────────────────────────────────────────────────────

class TestSlidingWindow:

    def test_small_history_no_drop(self):
        """Если сообщений меньше окна — ничего не отбрасывается."""
        s = SlidingWindowStrategy(window_size=10)
        msgs = _make_messages(3)  # system + 6 chat
        api_msgs = s.get_messages_for_api(msgs)
        assert len(api_msgs) == 7  # 1 system + 6 chat
        assert s.get_stats()["messages_dropped"] == 0

    def test_large_history_drops(self):
        """Если сообщений больше окна — лишние отбрасываются."""
        s = SlidingWindowStrategy(window_size=4)
        msgs = _make_messages(5)  # system + 10 chat
        api_msgs = s.get_messages_for_api(msgs)
        # system + 4 последних chat
        assert len(api_msgs) == 5
        assert api_msgs[0]["role"] == "system"
        assert api_msgs[1]["content"] == "Вопрос 4"
        assert s.get_stats()["messages_dropped"] == 6

    def test_system_always_included(self):
        """System-сообщения всегда передаются."""
        s = SlidingWindowStrategy(window_size=2)
        msgs = _make_messages(5)
        api_msgs = s.get_messages_for_api(msgs)
        assert api_msgs[0]["role"] == "system"

    def test_no_system(self):
        """Работает без system-сообщения."""
        s = SlidingWindowStrategy(window_size=4)
        msgs = _make_messages(5, system=False)
        api_msgs = s.get_messages_for_api(msgs)
        assert len(api_msgs) == 4
        assert all(m["role"] in ("user", "assistant") for m in api_msgs)

    def test_serialization(self):
        """Сериализация и десериализация."""
        s = SlidingWindowStrategy(window_size=8)
        s._messages_dropped = 5
        d = s.to_dict()
        s2 = SlidingWindowStrategy.from_dict(d)
        assert s2.window_size == 8
        assert s2._messages_dropped == 5

    def test_window_size_1(self):
        """Минимальное окно — 1 сообщение."""
        s = SlidingWindowStrategy(window_size=1)
        msgs = _make_messages(10)
        api_msgs = s.get_messages_for_api(msgs)
        assert len(api_msgs) == 2  # system + 1 chat
        assert api_msgs[1]["content"] == "Ответ 10"


# ─────────────────────────────────────────────────────────────
# StickyFactsStrategy
# ─────────────────────────────────────────────────────────────

class TestStickyFacts:

    def test_no_facts_initially(self):
        """Без фактов — только system + последние N."""
        s = StickyFactsStrategy(window_size=4)
        msgs = _make_messages(5)
        api_msgs = s.get_messages_for_api(msgs)
        # system + 4 последних chat (no facts block)
        assert len(api_msgs) == 5
        assert api_msgs[0]["role"] == "system"

    def test_with_facts(self):
        """С фактами — добавляется facts_block."""
        s = StickyFactsStrategy(window_size=4)
        s.facts = {"goal": "мобильное приложение", "budget": "2 млн"}
        msgs = _make_messages(5)
        api_msgs = s.get_messages_for_api(msgs)
        # system + facts_block + 4 chat
        assert len(api_msgs) == 6
        assert api_msgs[0]["role"] == "system"
        assert api_msgs[1]["role"] == "system"
        assert "мобильное приложение" in api_msgs[1]["content"]
        assert "2 млн" in api_msgs[1]["content"]

    def test_parse_facts_valid_json(self):
        """Парсинг корректного JSON."""
        raw = '{"goal": "приложение", "budget": "1M"}'
        result = StickyFactsStrategy._parse_facts(raw)
        assert result == {"goal": "приложение", "budget": "1M"}

    def test_parse_facts_markdown_wrapped(self):
        """Парсинг JSON обёрнутого в markdown."""
        raw = '```json\n{"goal": "app"}\n```'
        result = StickyFactsStrategy._parse_facts(raw)
        assert result == {"goal": "app"}

    def test_parse_facts_with_text(self):
        """Парсинг JSON внутри текста."""
        raw = 'Вот факты: {"goal": "app", "budget": "5M"} конец.'
        result = StickyFactsStrategy._parse_facts(raw)
        assert result == {"goal": "app", "budget": "5M"}

    def test_parse_facts_invalid(self):
        """Невалидный JSON — None."""
        result = StickyFactsStrategy._parse_facts("это не json")
        assert result is None

    def test_serialization(self):
        """Сериализация и десериализация с фактами."""
        s = StickyFactsStrategy(window_size=8, max_facts=15)
        s.facts = {"goal": "test", "budget": "100"}
        s._facts_update_count = 3
        d = s.to_dict()
        s2 = StickyFactsStrategy.from_dict(d)
        assert s2.window_size == 8
        assert s2.max_facts == 15
        assert s2.facts == {"goal": "test", "budget": "100"}
        assert s2._facts_update_count == 3

    def test_small_history(self):
        """Если сообщений меньше окна — все передаются."""
        s = StickyFactsStrategy(window_size=10)
        msgs = _make_messages(2)
        api_msgs = s.get_messages_for_api(msgs)
        assert len(api_msgs) == 5  # system + 4 chat


# ─────────────────────────────────────────────────────────────
# BranchingStrategy
# ─────────────────────────────────────────────────────────────

class TestBranching:

    def test_checkpoint(self):
        """Создание checkpoint."""
        s = BranchingStrategy()
        msgs = _make_messages(3)
        result = s.save_checkpoint("cp1", msgs)
        assert "cp1" in result
        assert "cp1" in s.checkpoints
        assert len(s.checkpoints["cp1"]) == 7

    def test_create_branch(self):
        """Создание ветки от checkpoint."""
        s = BranchingStrategy()
        msgs = _make_messages(3)
        s.save_checkpoint("cp1", msgs)
        result = s.create_branch("branch_a", "cp1")
        assert "создана" in result
        assert "branch_a" in s.branches

    def test_create_branch_no_checkpoint(self):
        """Создание ветки от несуществующего checkpoint."""
        s = BranchingStrategy()
        result = s.create_branch("branch_a", "nonexistent")
        assert "не найден" in result

    def test_switch_branch(self):
        """Переключение на ветку."""
        s = BranchingStrategy()
        msgs = _make_messages(3)
        s.save_checkpoint("cp1", msgs)
        s.create_branch("branch_a", "cp1")
        result = s.switch_branch("branch_a")
        assert s.current_branch == "branch_a"
        assert "Переключено" in result

    def test_switch_nonexistent(self):
        """Переключение на несуществующую ветку."""
        s = BranchingStrategy()
        result = s.switch_branch("nope")
        assert "не найдена" in result

    def test_leave_branch(self):
        """Выход из ветки."""
        s = BranchingStrategy()
        msgs = _make_messages(3)
        s.save_checkpoint("cp1", msgs)
        s.create_branch("b1", "cp1")
        s.switch_branch("b1")
        result = s.leave_branch()
        assert s.current_branch is None
        assert "основной" in result

    def test_messages_for_api_in_branch(self):
        """В ветке — сообщения из ветки."""
        s = BranchingStrategy()
        msgs = _make_messages(3)
        s.save_checkpoint("cp1", msgs)
        s.create_branch("b1", "cp1")
        s.switch_branch("b1")

        # Добавляем сообщения в ветку
        s.on_user_message("вопрос в ветке", msgs)
        s.on_assistant_message("ответ в ветке")

        api_msgs = s.get_messages_for_api(msgs)
        # base (7) + 2 added
        assert len(api_msgs) == 9
        assert api_msgs[-2]["content"] == "вопрос в ветке"
        assert api_msgs[-1]["content"] == "ответ в ветке"

    def test_messages_for_api_no_branch(self):
        """Без активной ветки — обычные сообщения."""
        s = BranchingStrategy()
        msgs = _make_messages(3)
        api_msgs = s.get_messages_for_api(msgs)
        assert api_msgs == msgs

    def test_independent_branches(self):
        """Ветки независимы."""
        s = BranchingStrategy()
        msgs = _make_messages(3)
        s.save_checkpoint("cp1", msgs)
        s.create_branch("b1", "cp1")
        s.create_branch("b2", "cp1")

        s.switch_branch("b1")
        s.on_user_message("b1 question", msgs)
        s.on_assistant_message("b1 answer")

        s.switch_branch("b2")
        s.on_user_message("b2 question", msgs)

        b1_msgs = s.branches["b1"].added_messages
        b2_msgs = s.branches["b2"].added_messages
        assert len(b1_msgs) == 2
        assert len(b2_msgs) == 1
        assert b1_msgs[0]["content"] == "b1 question"
        assert b2_msgs[0]["content"] == "b2 question"

    def test_serialization(self):
        """Сериализация с checkpoint'ами и ветками."""
        s = BranchingStrategy()
        msgs = _make_messages(2)
        s.save_checkpoint("cp1", msgs)
        s.create_branch("b1", "cp1")
        s.switch_branch("b1")
        s.on_user_message("test", msgs)

        d = s.to_dict()
        s2 = BranchingStrategy.from_dict(d)
        assert "cp1" in s2.checkpoints
        assert "b1" in s2.branches
        assert s2.current_branch == "b1"
        assert len(s2.branches["b1"].added_messages) == 1

    def test_list_info(self):
        """list_info содержит информацию о ветках."""
        s = BranchingStrategy()
        msgs = _make_messages(2)
        s.save_checkpoint("cp1", msgs)
        s.create_branch("b1", "cp1")
        info = s.list_info()
        assert "cp1" in info
        assert "b1" in info


# ─────────────────────────────────────────────────────────────
# Реестр и фабрика
# ─────────────────────────────────────────────────────────────

class TestRegistry:

    def test_create_sliding(self):
        s = create_strategy("sliding_window", window_size=5)
        assert isinstance(s, SlidingWindowStrategy)
        assert s.window_size == 5

    def test_create_facts(self):
        s = create_strategy("sticky_facts", window_size=8)
        assert isinstance(s, StickyFactsStrategy)
        assert s.window_size == 8

    def test_create_branching(self):
        s = create_strategy("branching")
        assert isinstance(s, BranchingStrategy)

    def test_create_unknown(self):
        with pytest.raises(ValueError, match="Неизвестная стратегия"):
            create_strategy("unknown_strategy")

    def test_from_dict_sliding(self):
        data = {"name": "sliding_window", "window_size": 12}
        s = strategy_from_dict(data)
        assert isinstance(s, SlidingWindowStrategy)
        assert s.window_size == 12

    def test_from_dict_facts(self):
        data = {"name": "sticky_facts", "window_size": 4, "facts": {"a": "b"}}
        s = strategy_from_dict(data)
        assert isinstance(s, StickyFactsStrategy)
        assert s.facts == {"a": "b"}

    def test_from_dict_branching(self):
        data = {"name": "branching", "checkpoints": {}, "branches": {}}
        s = strategy_from_dict(data)
        assert isinstance(s, BranchingStrategy)

    def test_from_dict_unknown(self):
        result = strategy_from_dict({"name": "nope"})
        assert result is None
