"""Tests for `hermes.config.settings`."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.config.settings import Settings, get_settings


def _set_required_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    required = {
        "HERMES_IMAP_USER": "me@gmail.com",
        "HERMES_IMAP_APP_PASSWORD": "secret",
        "HERMES_TELEGRAM_BOT_TOKEN": "token",
        "HERMES_TELEGRAM_CHAT_ID": "123456",
    }
    required.update(overrides)
    for key, value in required.items():
        monkeypatch.setenv(key, value)


class TestImapHostResolution:
    def test_defaults_to_gmail_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_env(monkeypatch)
        assert Settings().imap_host == "imap.gmail.com"

    def test_icloud_provider_resolves_icloud_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_env(monkeypatch, HERMES_IMAP_PROVIDER="icloud")
        assert Settings().imap_host == "imap.mail.me.com"

    def test_explicit_override_wins_over_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_env(
            monkeypatch,
            HERMES_IMAP_PROVIDER="icloud",
            HERMES_IMAP_HOST_OVERRIDE="imap.example.com",
        )
        assert Settings().imap_host == "imap.example.com"


class TestDerivedPaths:
    def test_db_and_log_paths_derive_from_hermes_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_required_env(monkeypatch, HERMES_HERMES_HOME=str(tmp_path))
        settings = Settings()
        assert settings.db_path == tmp_path / "hermes.db"
        assert settings.log_dir == tmp_path / "logs"
        assert settings.log_path == tmp_path / "logs" / "hermes.log"


class TestGetSettingsCaching:
    def test_returns_the_same_cached_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_env(monkeypatch)
        get_settings.cache_clear()
        try:
            assert get_settings() is get_settings()
        finally:
            get_settings.cache_clear()
