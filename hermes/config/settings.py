"""Environment-driven application settings, loaded from `.env` via pydantic-settings.

Credentials and infrastructure config (IMAP, Telegram, LLM, file paths) live here.
Freelancer profile data (skills, repos, scoring thresholds) lives in `profile.yaml`
and is loaded separately via `hermes.config.profile.load_profile`.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["ImapProvider", "Settings", "get_settings"]


class ImapProvider(StrEnum):
    """Supported IMAP providers with well-known hosts."""

    GMAIL = "gmail"
    ICLOUD = "icloud"


_IMAP_HOSTS: dict[ImapProvider, str] = {
    ImapProvider.GMAIL: "imap.gmail.com",
    ImapProvider.ICLOUD: "imap.mail.me.com",
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="HERMES_",
        extra="ignore",
    )

    # IMAP — always use an app-specific password, never the account password.
    imap_provider: ImapProvider = ImapProvider.GMAIL
    imap_host_override: str | None = None
    imap_port: int = 993
    imap_user: str
    imap_app_password: str
    imap_mailbox: str = "INBOX"

    # Telegram
    telegram_bot_token: str
    telegram_chat_id: str

    # LLM (see `hermes.tools.llm.get_llm` — added in Phase 2)
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None

    # Paths
    hermes_home: Path = Path.home() / ".hermes"
    profile_path: Path = Path(__file__).resolve().parent / "profile.yaml"

    # Scout polling
    email_poll_interval_seconds: int = 180

    @property
    def imap_host(self) -> str:
        """Resolve the IMAP host: an explicit override, or the provider default."""
        return self.imap_host_override or _IMAP_HOSTS[self.imap_provider]

    @property
    def db_path(self) -> Path:
        """Path to the SQLite database file."""
        return self.hermes_home / "hermes.db"

    @property
    def log_dir(self) -> Path:
        """Directory where rotating log files are written."""
        return self.hermes_home / "logs"

    @property
    def log_path(self) -> Path:
        """Path to the main Hermes log file."""
        return self.log_dir / "hermes.log"


@lru_cache
def get_settings() -> Settings:
    """Return the cached process-wide `Settings` instance."""
    return Settings()
