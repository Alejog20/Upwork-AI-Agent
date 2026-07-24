"""Tests for the Telegram startup/shutdown resilience helpers in `ulysses.cli.main`.

Scope is intentionally narrow: full CLI command testing (Typer's `CliRunner`
over `start`/`status`/`draft`/`build`/`go`) is a Phase 4 concern. These tests
cover the error-handling behavior added to make `ulysses start` resilient to
transient Telegram network failures instead of crashing the whole process,
plus the pure disk-writing helper used by `build`/`go`.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture
from telegram.error import InvalidToken, NetworkError, TimedOut
from typer.testing import CliRunner

from ulysses.cli.main import (
    _shutdown_telegram,
    _start_telegram_with_retry,
    _write_prototype_to_disk,
    app,
)
from ulysses.config.profile import DEFAULT_PROFILE_PATH, load_profile
from ulysses.config.settings import get_settings
from ulysses.models import GeneratedPrototype
from ulysses.tools.db import Job, JobStatus, UlyssesDB

runner = CliRunner()


def _telegram_app() -> MagicMock:
    app = MagicMock()
    app.initialize = AsyncMock()
    app.start = AsyncMock()
    app.updater = MagicMock()
    app.updater.start_polling = AsyncMock()
    app.updater.stop = AsyncMock()
    app.stop = AsyncMock()
    app.shutdown = AsyncMock()
    return app


class TestStartTelegramWithRetry:
    async def test_starts_immediately_on_success(self) -> None:
        app = _telegram_app()
        await _start_telegram_with_retry(app)
        app.initialize.assert_awaited_once()
        app.start.assert_awaited_once()
        app.updater.start_polling.assert_awaited_once()

    async def test_retries_on_network_error_then_succeeds(self, mocker: MockerFixture) -> None:
        mocker.patch("asyncio.sleep", AsyncMock())
        app = _telegram_app()
        app.initialize = AsyncMock(side_effect=[TimedOut(), None])
        await _start_telegram_with_retry(app)
        assert app.initialize.await_count == 2
        app.start.assert_awaited_once()

    async def test_does_not_retry_on_invalid_token(self, mocker: MockerFixture) -> None:
        sleep_mock = mocker.patch("asyncio.sleep", AsyncMock())
        app = _telegram_app()
        app.initialize = AsyncMock(side_effect=InvalidToken("bad token"))
        with pytest.raises(InvalidToken):
            await _start_telegram_with_retry(app)
        app.initialize.assert_awaited_once()
        sleep_mock.assert_not_awaited()

    async def test_keeps_retrying_indefinitely_on_repeated_network_errors(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch("asyncio.sleep", AsyncMock())
        app = _telegram_app()
        app.initialize = AsyncMock(side_effect=[NetworkError("a"), NetworkError("b"), None])
        await _start_telegram_with_retry(app)
        assert app.initialize.await_count == 3


class TestShutdownTelegram:
    async def test_stops_and_shuts_down_when_running(self) -> None:
        app = _telegram_app()
        app.updater.running = True
        app.running = True

        await _shutdown_telegram(app)

        app.updater.stop.assert_awaited_once()
        app.stop.assert_awaited_once()
        app.shutdown.assert_awaited_once()

    async def test_skips_stop_calls_when_never_started(self) -> None:
        app = _telegram_app()
        app.updater.running = False
        app.running = False

        await _shutdown_telegram(app)

        app.updater.stop.assert_not_awaited()
        app.stop.assert_not_awaited()
        app.shutdown.assert_awaited_once()

    async def test_swallows_exceptions_instead_of_raising(self) -> None:
        app = _telegram_app()
        app.updater.running = True
        app.running = True
        app.stop = AsyncMock(side_effect=NetworkError("boom"))

        await _shutdown_telegram(app)  # must not raise


class TestWritePrototypeToDisk:
    def test_writes_all_four_files_under_output_job_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        prototype = GeneratedPrototype(
            job_id="job-1",
            category="scraper",
            demo_script="print('hi')",
            requirements_txt="requests==2.32.3\n",
            readme_md="# Demo\n",
            config_example_env="# none needed\n",
            zip_filename="ulysses_demo_job-1.zip",
        )

        output_dir = _write_prototype_to_disk(prototype, "job-1")

        assert output_dir == Path("output") / "job-1"
        assert (output_dir / "demo.py").read_text() == "print('hi')"
        assert (output_dir / "requirements.txt").read_text() == "requests==2.32.3\n"
        assert (output_dir / "README.md").read_text() == "# Demo\n"
        assert (output_dir / "config.example.env").read_text() == "# none needed\n"


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point every CLI-level test at a throwaway DB/home dir, never the real one."""
    monkeypatch.setenv("ULYSSES_IMAP_USER", "me@gmail.com")
    monkeypatch.setenv("ULYSSES_IMAP_APP_PASSWORD", "secret")
    monkeypatch.setenv("ULYSSES_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("ULYSSES_TELEGRAM_CHAT_ID", "123456")
    monkeypatch.setenv("ULYSSES_ULYSSES_HOME", str(tmp_path / "home"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _seed_job(db_path: Path, **overrides: object) -> None:
    job_id = overrides.get("id", "job-1")
    defaults: dict[str, object] = {
        "id": job_id,
        "title": "Python scraper",
        "description": "desc",
        "url": f"https://www.upwork.com/jobs/~{job_id}",
        "score": 80.0,
        "category": "tier1",
        "status": JobStatus.NEW,
        "posted_at": datetime.now(UTC),
    }
    defaults.update(overrides)
    db = UlyssesDB(db_path)
    await db.init()
    await db.upsert_job(Job(**defaults))
    await db.dispose()


class TestStatusCommand:
    def test_shows_zero_counts_on_fresh_db(self) -> None:
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "total" in result.stdout


class TestQueueCommand:
    def test_shows_message_when_no_jobs_match(self) -> None:
        result = runner.invoke(app, ["queue"])
        assert result.exit_code == 0
        assert "No jobs match" in result.stdout

    def test_lists_seeded_jobs_filtered_by_min_score(self) -> None:
        settings = get_settings()
        asyncio.run(_seed_job(settings.db_path, id="job-1", title="High score job", score=90.0))
        asyncio.run(_seed_job(settings.db_path, id="job-2", title="Low score job", score=10.0))

        result = runner.invoke(app, ["queue", "--min-score", "50"])

        assert result.exit_code == 0
        assert "High score job" in result.stdout
        assert "Low score job" not in result.stdout


class TestArchiveCommand:
    def test_errors_for_unknown_job(self) -> None:
        result = runner.invoke(app, ["archive", "no-such-id"])
        assert result.exit_code == 1
        assert "No job found" in result.stdout

    def test_archives_a_seeded_job(self) -> None:
        settings = get_settings()
        asyncio.run(_seed_job(settings.db_path, id="job-1", title="Some job"))

        result = runner.invoke(app, ["archive", "job-1"])

        assert result.exit_code == 0
        assert "Archived" in result.stdout

        async def _check_status() -> JobStatus:
            db = UlyssesDB(settings.db_path)
            await db.init()
            job = await db.get_job("job-1")
            await db.dispose()
            assert job is not None
            return job.status

        assert asyncio.run(_check_status()) == JobStatus.ARCHIVED


class TestConfigCommands:
    @pytest.fixture(autouse=True)
    def _tmp_profile(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        # Never let a test write to the real profile.yaml on disk.
        tmp_profile_path = tmp_path / "profile.yaml"
        tmp_profile_path.write_text(
            DEFAULT_PROFILE_PATH.read_text(encoding="utf-8"), encoding="utf-8"
        )
        monkeypatch.setenv("ULYSSES_PROFILE_PATH", str(tmp_profile_path))
        get_settings.cache_clear()
        yield tmp_profile_path
        get_settings.cache_clear()

    def test_config_show_prints_yaml(self) -> None:
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "freelancer" in result.stdout

    def test_config_set_updates_and_persists(self, _tmp_profile: Path) -> None:
        result = runner.invoke(app, ["config", "set", "freelancer.rate_usd_hr", "42"])

        assert result.exit_code == 0
        assert "Set" in result.stdout
        assert load_profile(_tmp_profile).freelancer.rate_usd_hr == 42.0

    def test_config_set_unknown_key_errors(self) -> None:
        result = runner.invoke(app, ["config", "set", "nonexistent.key", "x"])
        assert result.exit_code == 1


class TestInstallUninstallCommands:
    def test_install_success(self, mocker: MockerFixture) -> None:
        install_mock = mocker.patch(
            "ulysses.cli.main.install_launch_agent", return_value=Path("/fake/path.plist")
        )
        result = runner.invoke(app, ["install"])
        assert result.exit_code == 0
        assert "Installed" in result.stdout
        install_mock.assert_called_once()

    def test_install_failure_shows_a_clean_error(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "ulysses.cli.main.install_launch_agent", side_effect=RuntimeError("no uv found")
        )
        result = runner.invoke(app, ["install"])
        assert result.exit_code == 1
        assert "Failed to install" in result.stdout

    def test_uninstall_when_installed(self, mocker: MockerFixture) -> None:
        mocker.patch("ulysses.cli.main.uninstall_launch_agent", return_value=True)
        result = runner.invoke(app, ["uninstall"])
        assert result.exit_code == 0
        assert "removed" in result.stdout.lower()

    def test_uninstall_when_nothing_installed(self, mocker: MockerFixture) -> None:
        mocker.patch("ulysses.cli.main.uninstall_launch_agent", return_value=False)
        result = runner.invoke(app, ["uninstall"])
        assert result.exit_code == 0
        assert "No LaunchAgent" in result.stdout
