"""Тесты для core/ — config и persistence."""

import json
import os
import pytest
from unittest.mock import patch
from pathlib import Path


class TestConfig:
    """Тесты для core.config."""

    def test_get_config_returns_appconfig(self):
        import core.config as cfg
        cfg._config = None  # сбросить кэш
        with patch.dict(os.environ, {"CLOUD_API_KEY": "test-key"}, clear=False):
            config = cfg.get_config()
            assert config.cloud_api_key == "test-key"
            cfg._config = None

    def test_get_config_singleton(self):
        import core.config as cfg
        cfg._config = None
        with patch.dict(os.environ, {"CLOUD_API_KEY": "k1"}, clear=False):
            c1 = cfg.get_config()
            c2 = cfg.get_config()
            assert c1 is c2
            cfg._config = None

    def test_primary_api_key_prefers_cloud(self):
        import core.config as cfg
        cfg._config = None
        with patch.dict(os.environ, {
            "CLOUD_API_KEY": "cloud-key",
            "OPENAI_API_KEY": "openai-key",
        }, clear=False):
            config = cfg.get_config()
            assert config.primary_api_key == "cloud-key"
            cfg._config = None

    def test_primary_api_key_falls_back_to_openai(self):
        import core.config as cfg
        cfg._config = None
        with patch.dict(os.environ, {
            "CLOUD_API_KEY": "",
            "OPENAI_API_KEY": "openai-key",
        }, clear=False):
            config = cfg.get_config()
            assert config.primary_api_key == "openai-key"
            cfg._config = None

    def test_has_cloud_key(self):
        import core.config as cfg
        cfg._config = None
        with patch.dict(os.environ, {"CLOUD_API_KEY": "x"}, clear=False):
            assert cfg.get_config().has_cloud_key is True
            cfg._config = None

    def test_default_paths(self):
        import core.config as cfg
        cfg._config = None
        config = cfg.get_config()
        assert config.logs_path.name == "logs"
        assert config.sessions_path.name == "sessions"
        cfg._config = None


class TestPersistence:
    """Тесты для core.persistence."""

    def test_load_json_nonexistent_returns_default(self, tmp_path):
        from core.persistence import load_json
        result = load_json(tmp_path / "missing.json", default={"empty": True})
        assert result == {"empty": True}

    def test_save_and_load_json(self, tmp_path):
        from core.persistence import load_json, save_json
        data = {"key": "значение", "list": [1, 2, 3]}
        path = tmp_path / "test.json"
        save_json(path, data)
        loaded = load_json(path)
        assert loaded == data

    def test_save_json_creates_directories(self, tmp_path):
        from core.persistence import save_json
        path = tmp_path / "sub" / "dir" / "file.json"
        save_json(path, [1, 2])
        assert path.exists()

    def test_save_json_unicode(self, tmp_path):
        from core.persistence import save_json, load_json
        data = {"текст": "Привет мир 🌍"}
        path = tmp_path / "unicode.json"
        save_json(path, data)
        loaded = load_json(path)
        assert loaded["текст"] == "Привет мир 🌍"

    def test_load_json_default_none(self, tmp_path):
        from core.persistence import load_json
        result = load_json(tmp_path / "nope.json")
        assert result is None
