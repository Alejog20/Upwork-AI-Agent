"""Tests for `ulysses.config.settings`."""

from __future__ import annotations

from pathlib import Path

import pytest

from ulysses.config.settings import LlmProvider, Settings, get_settings


def _set_required_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    required = {
        "ULYSSES_IMAP_USER": "me@gmail.com",
        "ULYSSES_IMAP_APP_PASSWORD": "secret",
        "ULYSSES_TELEGRAM_BOT_TOKEN": "token",
        "ULYSSES_TELEGRAM_CHAT_ID": "123456",
    }
    required.update(overrides)
    for key, value in required.items():
        monkeypatch.setenv(key, value)


class TestImapHostResolution:
    def test_defaults_to_gmail_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_env(monkeypatch)
        assert Settings().imap_host == "imap.gmail.com"

    def test_icloud_provider_resolves_icloud_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_env(monkeypatch, ULYSSES_IMAP_PROVIDER="icloud")
        assert Settings().imap_host == "imap.mail.me.com"

    def test_explicit_override_wins_over_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_env(
            monkeypatch,
            ULYSSES_IMAP_PROVIDER="icloud",
            ULYSSES_IMAP_HOST_OVERRIDE="imap.example.com",
        )
        assert Settings().imap_host == "imap.example.com"


class TestLlmProvider:
    def test_defaults_to_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_env(monkeypatch)
        # `_env_file=None` isolates this from whatever this machine's own
        # `.env` happens to set -- we want the field's bare default, not
        # whatever the developer last configured.
        assert Settings(_env_file=None).llm_provider is LlmProvider.OPENAI

    def test_can_be_set_to_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_env(monkeypatch, ULYSSES_LLM_PROVIDER="anthropic")
        assert Settings().llm_provider is LlmProvider.ANTHROPIC

    def test_embedding_api_key_is_independent_of_llm_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_required_env(
            monkeypatch,
            ULYSSES_LLM_API_KEY="anthropic-chat-key",
            ULYSSES_LLM_EMBEDDING_API_KEY="gemini-embedding-key",
        )
        settings = Settings()
        assert settings.llm_api_key == "anthropic-chat-key"
        assert settings.llm_embedding_api_key == "gemini-embedding-key"


class TestDerivedPaths:
    def test_db_and_log_paths_derive_from_ulysses_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_required_env(monkeypatch, ULYSSES_ULYSSES_HOME=str(tmp_path))
        settings = Settings()
        assert settings.db_path == tmp_path / "ulysses.db"
        assert settings.log_dir == tmp_path / "logs"
        assert settings.log_path == tmp_path / "logs" / "ulysses.log"


class TestGetSettingsCaching:
    def test_returns_the_same_cached_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_env(monkeypatch)
        get_settings.cache_clear()
        try:
            assert get_settings() is get_settings()
        finally:
            get_settings.cache_clear()
