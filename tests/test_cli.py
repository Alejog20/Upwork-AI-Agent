"""Tests for the Telegram startup/shutdown resilience helpers in `ulysses.cli.main`.

Scope is intentionally narrow: full CLI command testing (Typer's `CliRunner`
over `start`/`status`/`draft`/`build`/`go`) is a Phase 4 concern. These tests
cover the error-handling behavior added to make `ulysses start` resilient to
transient Telegram network failures instead of crashing the whole process,
plus the pure disk-writing helper used by `build`/`go`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture
from telegram.error import InvalidToken, NetworkError, TimedOut

from ulysses.cli.main import (
    _shutdown_telegram,
    _start_telegram_with_retry,
    _write_prototype_to_disk,
)
from ulysses.models import GeneratedPrototype


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
