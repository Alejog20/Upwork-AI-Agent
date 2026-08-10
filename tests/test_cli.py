"""Tests for the Telegram startup/shutdown resilience helpers in `ulysses.cli.main`.

Scope is intentionally narrow: full CLI command testing (Typer's `CliRunner`
over `start`/`status`/`draft`/`build`/`go`) is a Phase 4 concern. These tests
cover the error-handling behavior added to make `ulysses start` resilient to
transient Telegram network failures instead of crashing the whole process,
plus the pure disk-writing helper used by `build`/`go`.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture
from telegram.error import InvalidToken, NetworkError, TimedOut
from typer.testing import CliRunner

from ulysses.agents.scorer import score_job
from ulysses.cli.main import (
    _make_build_handler,
    _make_draft_handler,
    _read_pasted_job_listings,
    _shutdown_telegram,
    _start_telegram_with_retry,
    _write_prototype_to_disk,
    app,
)
from ulysses.config.profile import DEFAULT_PROFILE_PATH, Profile, load_profile
from ulysses.config.settings import get_settings
from ulysses.models import BudgetRange, BudgetType, GeneratedPrototype, JobPost, JobScore, Milestone
from ulysses.tools.db import Job, JobStatus, UlyssesDB
from ulysses.tools.manual_job import ManualJobParseError

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
    monkeypatch.setenv("ULYSSES_LLM_API_KEY", "test-key")
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


async def _seed_full_job(db_path: Path, job: JobPost, score: JobScore) -> None:
    """Seed a job with `job_json`/`score_json` populated, as `_lookup_full_job` requires."""
    await _seed_job(
        db_path,
        id=job.id,
        title=job.title,
        description=job.description,
        url=job.url,
        score=score.total_score,
        category=score.gig_category.value,
        posted_at=job.posted_at,
        job_json=job.model_dump_json(),
        score_json=score.model_dump_json(),
    )


class TestMakeDraftHandler:
    """Unit tests for the Telegram "Draft Proposal" button's callback logic."""

    async def test_drafts_and_sends_for_a_known_job(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        db = AsyncMock()
        db.get_full_job = AsyncMock(return_value=(fresh_job, score))
        proposal_agent = AsyncMock()
        proposal_agent.generate = AsyncMock(return_value=_mock_proposal())
        notifier = AsyncMock()

        handler = _make_draft_handler(db, proposal_agent, notifier, profile)
        await handler(fresh_job.id)

        proposal_agent.generate.assert_awaited_once_with(fresh_job, score, profile)
        db.add_proposal_draft.assert_awaited_once_with(fresh_job.id, "Generated proposal text.")
        notifier.send_proposal_draft.assert_awaited_once_with(
            fresh_job.id, "Generated proposal text."
        )

    async def test_sends_error_message_for_a_job_predating_detailed_storage(self) -> None:
        db = AsyncMock()
        db.get_full_job = AsyncMock(return_value=None)
        proposal_agent = AsyncMock()
        notifier = AsyncMock()

        handler = _make_draft_handler(db, proposal_agent, notifier, MagicMock())
        await handler("unknown-id")

        notifier.send_error_message.assert_awaited_once()
        proposal_agent.generate.assert_not_awaited()


class TestMakeBuildHandler:
    """Unit tests for the Telegram "Build Demo" button's callback logic."""

    async def test_builds_and_sends_for_a_known_job(
        self, fresh_job: JobPost, profile: Profile, mocker: MockerFixture
    ) -> None:
        score = score_job(fresh_job, profile)
        db = AsyncMock()
        db.get_full_job = AsyncMock(return_value=(fresh_job, score))
        prototype_agent = AsyncMock()
        prototype = _mock_prototype(fresh_job.id)
        prototype_agent.generate = AsyncMock(return_value=prototype)
        notifier = AsyncMock()
        mocker.patch("ulysses.cli.main.build_prototype_zip", return_value=b"zip-bytes")

        handler = _make_build_handler(db, prototype_agent, notifier, profile)
        await handler(fresh_job.id)

        prototype_agent.generate.assert_awaited_once_with(fresh_job, score, profile)
        assert db.add_prototype_file.await_count == 4
        notifier.send_prototype_zip.assert_awaited_once_with(fresh_job.id, prototype, b"zip-bytes")

    async def test_sends_error_message_for_a_job_predating_detailed_storage(self) -> None:
        db = AsyncMock()
        db.get_full_job = AsyncMock(return_value=None)
        prototype_agent = AsyncMock()
        notifier = AsyncMock()

        handler = _make_build_handler(db, prototype_agent, notifier, MagicMock())
        await handler("unknown-id")

        notifier.send_error_message.assert_awaited_once()
        prototype_agent.generate.assert_not_awaited()


class TestDraftCommand:
    def test_errors_for_unknown_url(self) -> None:
        result = runner.invoke(app, ["draft", "https://www.upwork.com/jobs/~unknown"])

        assert result.exit_code == 1
        assert "No job found for URL" in result.stdout

    def test_errors_for_job_predating_detailed_storage(self) -> None:
        settings = get_settings()
        url = "https://www.upwork.com/jobs/~job-1"
        asyncio.run(_seed_job(settings.db_path, id="job-1", url=url))

        result = runner.invoke(app, ["draft", url])

        assert result.exit_code == 1
        assert "predates detailed storage" in result.stdout

    def test_drafts_a_proposal_for_a_known_job(
        self, mocker: MockerFixture, fresh_job: JobPost, profile: Profile
    ) -> None:
        settings = get_settings()
        score = score_job(fresh_job, profile)
        asyncio.run(_seed_full_job(settings.db_path, fresh_job, score))
        mocker.patch(
            "ulysses.cli.main.ProposalAgent",
            return_value=MagicMock(generate=AsyncMock(return_value=_mock_proposal())),
        )

        result = runner.invoke(app, ["draft", fresh_job.url])

        assert result.exit_code == 0
        assert "Generated proposal text." in result.stdout

    def test_prints_a_milestones_panel_when_present(
        self, mocker: MockerFixture, fresh_job: JobPost, profile: Profile
    ) -> None:
        settings = get_settings()
        score = score_job(fresh_job, profile)
        asyncio.run(_seed_full_job(settings.db_path, fresh_job, score))
        proposal = _mock_proposal()
        proposal.milestones = [Milestone(description="Set it up", amount_usd=100.0, days=2)]
        mocker.patch(
            "ulysses.cli.main.ProposalAgent",
            return_value=MagicMock(generate=AsyncMock(return_value=proposal)),
        )

        result = runner.invoke(app, ["draft", fresh_job.url])

        assert result.exit_code == 0
        assert "Suggested Milestones" in result.stdout
        assert "Set it up" in result.stdout


class TestBuildCommand:
    def test_errors_for_unknown_url(self) -> None:
        result = runner.invoke(app, ["build", "https://www.upwork.com/jobs/~unknown"])

        assert result.exit_code == 1
        assert "No job found for URL" in result.stdout

    def test_builds_a_prototype_for_a_known_job(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mocker: MockerFixture,
        fresh_job: JobPost,
        profile: Profile,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        settings = get_settings()
        score = score_job(fresh_job, profile)
        asyncio.run(_seed_full_job(settings.db_path, fresh_job, score))
        mocker.patch(
            "ulysses.cli.main.PrototypeAgent",
            return_value=MagicMock(generate=AsyncMock(return_value=_mock_prototype(fresh_job.id))),
        )

        result = runner.invoke(app, ["build", fresh_job.url])

        assert result.exit_code == 0
        assert "# Demo" in result.stdout
        assert (Path("output") / fresh_job.id / "demo.py").read_text() == "print('demo')"


class TestGoCommand:
    def test_errors_for_unknown_url(self) -> None:
        result = runner.invoke(app, ["go", "https://www.upwork.com/jobs/~unknown"])

        assert result.exit_code == 1
        assert "No job found for URL" in result.stdout

    def test_drafts_and_builds_for_a_known_job(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mocker: MockerFixture,
        fresh_job: JobPost,
        profile: Profile,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        settings = get_settings()
        score = score_job(fresh_job, profile)
        asyncio.run(_seed_full_job(settings.db_path, fresh_job, score))
        mocker.patch(
            "ulysses.cli.main.ProposalAgent",
            return_value=MagicMock(generate=AsyncMock(return_value=_mock_proposal())),
        )
        mocker.patch(
            "ulysses.cli.main.PrototypeAgent",
            return_value=MagicMock(generate=AsyncMock(return_value=_mock_prototype(fresh_job.id))),
        )

        result = runner.invoke(app, ["go", fresh_job.url])

        assert result.exit_code == 0
        assert "Generated proposal text." in result.stdout
        assert "# Demo" in result.stdout
        output_dir = Path("output") / fresh_job.id
        assert output_dir.exists()
        assert (output_dir / "proposal.txt").read_text() == "Generated proposal text."


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


class TestWonLostCommands:
    def test_won_errors_for_unknown_job(self) -> None:
        result = runner.invoke(app, ["won", "no-such-id"])
        assert result.exit_code == 1
        assert "No job found" in result.stdout

    def test_won_marks_job_and_records_outcome(self) -> None:
        settings = get_settings()
        asyncio.run(_seed_job(settings.db_path, id="job-1", title="Some job"))

        result = runner.invoke(app, ["won", "job-1", "--value", "500", "--note", "great client"])

        assert result.exit_code == 0
        assert "Won" in result.stdout

        async def _check() -> tuple[JobStatus, float | None]:
            db = UlyssesDB(settings.db_path)
            await db.init()
            job = await db.get_job("job-1")
            outcomes = await db.list_outcomes()
            await db.dispose()
            assert job is not None
            return job.status, outcomes[0].contract_value_usd

        status, value = asyncio.run(_check())
        assert status == JobStatus.WON
        assert value == 500.0

    def test_lost_errors_for_unknown_job(self) -> None:
        result = runner.invoke(app, ["lost", "no-such-id"])
        assert result.exit_code == 1
        assert "No job found" in result.stdout

    def test_lost_marks_job(self) -> None:
        settings = get_settings()
        asyncio.run(_seed_job(settings.db_path, id="job-1", title="Some job"))

        result = runner.invoke(app, ["lost", "job-1", "--note", "went with someone else"])

        assert result.exit_code == 0
        assert "Lost" in result.stdout

        async def _check_status() -> JobStatus:
            db = UlyssesDB(settings.db_path)
            await db.init()
            job = await db.get_job("job-1")
            await db.dispose()
            assert job is not None
            return job.status

        assert asyncio.run(_check_status()) == JobStatus.LOST


class TestAnalyticsCommand:
    def test_shows_message_when_no_outcomes_recorded(self) -> None:
        result = runner.invoke(app, ["analytics"])
        assert result.exit_code == 0
        assert "No outcomes recorded yet" in result.stdout

    def test_shows_win_rate_breakdown_after_recording_outcomes(self) -> None:
        settings = get_settings()
        asyncio.run(_seed_job(settings.db_path, id="job-1", title="Won job", score=90.0))
        asyncio.run(_seed_job(settings.db_path, id="job-2", title="Lost job", score=40.0))
        runner.invoke(app, ["won", "job-1"])
        runner.invoke(app, ["lost", "job-2"])

        result = runner.invoke(app, ["analytics"])

        assert result.exit_code == 0
        assert "2 outcomes recorded" in result.stdout
        assert "Win Rate by Category" in result.stdout
        assert "Scoring weight suggestions" in result.stdout


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


class TestStartCommand:
    """`start()` itself just wraps `run_forever` with top-level exception handling --
    the actual scout/score/notify orchestration loop is exercised via its own
    components (ScoutAgent, NotifierAgent, etc.), not simulated here."""

    def test_keyboard_interrupt_exits_cleanly(self, mocker: MockerFixture) -> None:
        mocker.patch("ulysses.cli.main.run_forever", new=AsyncMock(side_effect=KeyboardInterrupt()))

        result = runner.invoke(app, ["start"])

        assert result.exit_code == 0
        assert "Ulysses stopped" in result.stdout

    def test_invalid_token_exits_with_a_clean_error(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "ulysses.cli.main.run_forever",
            new=AsyncMock(side_effect=InvalidToken("bad token")),
        )

        result = runner.invoke(app, ["start"])

        assert result.exit_code == 1
        assert "Telegram rejected the bot token" in result.stdout

    def test_unexpected_exception_exits_with_a_clean_error(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "ulysses.cli.main.run_forever", new=AsyncMock(side_effect=RuntimeError("boom"))
        )

        result = runner.invoke(app, ["start"])

        assert result.exit_code == 1
        assert "Ulysses crashed" in result.stdout


def _mock_pasted_job(**overrides: object) -> JobPost:
    defaults: dict[str, object] = {
        "id": "manual-job-1",
        "title": "Python scraper for real estate listings",
        "description": "Scrape three sites daily and dedupe results.",
        "budget": BudgetRange(),
        "skills_required": ["python", "web scraping"],
        "client_hires": 0,
        "payment_verified": True,
        "proposals_count": 3,
        "posted_at": datetime.now(UTC),
        "url": "manual://fixed-for-test",
    }
    defaults.update(overrides)
    return JobPost(**defaults)


def _mock_proposal() -> MagicMock:
    proposal = MagicMock()
    proposal.full_text = "Generated proposal text."
    proposal.milestones = []
    proposal.category = "scraping"
    proposal.timeline = "3 days"
    proposal.bid_usd = 200.0
    return proposal


def _mock_prototype(job_id: str) -> MagicMock:
    prototype = MagicMock()
    prototype.job_id = job_id
    prototype.demo_script = "print('demo')"
    prototype.requirements_txt = "requests==2.32.3\n"
    prototype.readme_md = "# Demo\n"
    prototype.config_example_env = "# none\n"
    return prototype


def _mock_narrator_agent(mocker: MockerFixture, *blurbs: str) -> MagicMock:
    """Patch `NarratorAgent` so chat tests never make a real LLM call for narration."""
    texts = blurbs or ("A short verdict blurb.",)
    return mocker.patch(
        "ulysses.cli.main.NarratorAgent",
        return_value=MagicMock(narrate=AsyncMock(side_effect=list(texts))),
    )


class TestChatCommand:
    def test_quitting_immediately_prints_goodbye_and_touches_nothing(
        self, mocker: MockerFixture
    ) -> None:
        extract_mock = mocker.patch("ulysses.cli.main.extract_job_from_text", new=AsyncMock())

        result = runner.invoke(app, ["chat"], input="quit\n")

        assert result.exit_code == 0
        assert "Goodbye" in result.stdout
        extract_mock.assert_not_awaited()

    def test_eof_before_any_input_leaves_cleanly(self) -> None:
        result = runner.invoke(app, ["chat"], input="")

        assert result.exit_code == 0
        assert "Goodbye" in result.stdout

    def test_processes_one_pasted_job_end_to_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.chdir(tmp_path)
        job = _mock_pasted_job()
        mocker.patch("ulysses.cli.main.extract_job_from_text", new=AsyncMock(return_value=job))
        mocker.patch(
            "ulysses.cli.main.ProposalAgent",
            return_value=MagicMock(generate=AsyncMock(return_value=_mock_proposal())),
        )
        mocker.patch(
            "ulysses.cli.main.PrototypeAgent",
            return_value=MagicMock(generate=AsyncMock(return_value=_mock_prototype(job.id))),
        )
        _mock_narrator_agent(mocker)

        result = runner.invoke(app, ["chat"], input="Some pasted job text here.\n")

        assert result.exit_code == 0
        assert "Generated proposal text." in result.stdout
        assert "# Demo" in result.stdout
        assert (Path("output") / job.id / "proposal.txt").read_text() == "Generated proposal text."

        settings = get_settings()

        async def _check() -> Job | None:
            db = UlyssesDB(settings.db_path)
            await db.init()
            stored = await db.get_job(job.id)
            await db.dispose()
            return stored

        stored_job = asyncio.run(_check())
        assert stored_job is not None
        assert stored_job.title == job.title

    def test_narrator_failure_does_not_block_drafting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.chdir(tmp_path)
        job = _mock_pasted_job()
        mocker.patch("ulysses.cli.main.extract_job_from_text", new=AsyncMock(return_value=job))
        mocker.patch(
            "ulysses.cli.main.ProposalAgent",
            return_value=MagicMock(generate=AsyncMock(return_value=_mock_proposal())),
        )
        mocker.patch(
            "ulysses.cli.main.PrototypeAgent",
            return_value=MagicMock(generate=AsyncMock(return_value=_mock_prototype(job.id))),
        )
        mocker.patch(
            "ulysses.cli.main.NarratorAgent",
            return_value=MagicMock(narrate=AsyncMock(side_effect=RuntimeError("boom"))),
        )

        result = runner.invoke(app, ["chat"], input="Some pasted job text here.\n")

        assert result.exit_code == 0
        assert "Generated proposal text." in result.stdout  # drafting still happened
        assert "# Demo" in result.stdout

    def test_skip_recommended_job_never_drafts_or_builds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # Stale, saturated with proposals, an experienced client, no skill overlap,
        # and a budget far outside the target range -- scores well below
        # min_score_to_notify, so score_job recommends SKIP.
        weak_job = _mock_pasted_job(
            description="Simple task, shouldn't take long -- basic data entry.",
            posted_at=datetime.now(UTC) - timedelta(hours=10),
            proposals_count=50,
            client_hires=10,
            skills_required=["cobol"],
            budget=BudgetRange(type=BudgetType.FIXED, min_amount=5000, max_amount=5000),
        )
        mocker.patch("ulysses.cli.main.extract_job_from_text", new=AsyncMock(return_value=weak_job))
        proposal_agent_mock = mocker.patch("ulysses.cli.main.ProposalAgent")
        prototype_agent_mock = mocker.patch("ulysses.cli.main.PrototypeAgent")
        _mock_narrator_agent(mocker, "I'd skip this one.")

        result = runner.invoke(app, ["chat"], input="A weak job listing.\n")

        assert result.exit_code == 0
        assert "I'd skip this one." in result.stdout
        assert "Not drafting a proposal" in result.stdout
        assert "Generated proposal text." not in result.stdout
        proposal_agent_mock.assert_not_called()
        prototype_agent_mock.assert_not_called()
        assert not (Path("output") / weak_job.id).exists()

    def test_extraction_failure_shows_friendly_error_and_continues_session(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "ulysses.cli.main.extract_job_from_text",
            new=AsyncMock(side_effect=ManualJobParseError("nope")),
        )

        result = runner.invoke(app, ["chat"], input="not a real job listing\n")

        assert result.exit_code == 0
        assert "Couldn't read that listing" in result.stdout
        assert "nope" in result.stdout
        assert "Goodbye" in result.stdout

    def test_processes_two_jobs_in_one_session_via_typed_sentinel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        # Unlike EOF, the typed SUBMITJOB sentinel is just more text in the
        # same piped stream, so (unlike the EOF-only path) a multi-job
        # session is exercisable end-to-end through a single runner.invoke().
        monkeypatch.chdir(tmp_path)
        job1 = _mock_pasted_job(id="job-1", url="manual://job-1")
        job2 = _mock_pasted_job(id="job-2", url="manual://job-2", title="Second job")
        mocker.patch(
            "ulysses.cli.main.extract_job_from_text",
            new=AsyncMock(side_effect=[job1, job2]),
        )
        mocker.patch(
            "ulysses.cli.main.ProposalAgent",
            return_value=MagicMock(
                generate=AsyncMock(side_effect=[_mock_proposal(), _mock_proposal()])
            ),
        )
        mocker.patch(
            "ulysses.cli.main.PrototypeAgent",
            return_value=MagicMock(
                generate=AsyncMock(side_effect=[_mock_prototype("job-1"), _mock_prototype("job-2")])
            ),
        )
        _mock_narrator_agent(mocker, "Blurb one.", "Blurb two.")

        result = runner.invoke(
            app,
            ["chat"],
            input="job one textSUBMITJOB\njob two textSUBMITJOB\nquit\n",
        )

        assert result.exit_code == 0
        assert (Path("output") / "job-1").exists()
        assert (Path("output") / "job-2").exists()

    def test_processes_two_jobs_from_one_paste_via_next_job_sentinel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        # Both listings are queued with NEXTJOB in a single paste, then
        # submitted together with one SUBMITJOB -- the batch path, as
        # opposed to the sequential-SUBMITJOB path exercised above.
        monkeypatch.chdir(tmp_path)
        job1 = _mock_pasted_job(id="job-1", url="manual://job-1")
        job2 = _mock_pasted_job(id="job-2", url="manual://job-2", title="Second job")
        mocker.patch(
            "ulysses.cli.main.extract_job_from_text",
            new=AsyncMock(side_effect=[job1, job2]),
        )
        mocker.patch(
            "ulysses.cli.main.ProposalAgent",
            return_value=MagicMock(
                generate=AsyncMock(side_effect=[_mock_proposal(), _mock_proposal()])
            ),
        )
        mocker.patch(
            "ulysses.cli.main.PrototypeAgent",
            return_value=MagicMock(
                generate=AsyncMock(side_effect=[_mock_prototype("job-1"), _mock_prototype("job-2")])
            ),
        )
        _mock_narrator_agent(mocker, "Blurb one.", "Blurb two.")

        result = runner.invoke(
            app,
            ["chat"],
            input="job one textNEXTJOB\njob two textSUBMITJOB\nquit\n",
        )

        assert result.exit_code == 0
        assert "Processing job 1 of 2" in result.stdout
        assert "Processing job 2 of 2" in result.stdout
        assert (Path("output") / "job-1").exists()
        assert (Path("output") / "job-2").exists()

    def test_submitting_nothing_shows_a_nudge_and_continues(self) -> None:
        result = runner.invoke(app, ["chat"], input="SUBMITJOB\nquit\n")

        assert result.exit_code == 0
        assert "Nothing pasted" in result.stdout
        assert "Goodbye" in result.stdout

    def test_unexpected_error_shows_friendly_message_and_continues_the_batch(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "ulysses.cli.main.extract_job_from_text",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        )

        result = runner.invoke(app, ["chat"], input="a pasted job listing\n")

        assert result.exit_code == 0
        assert "Something went wrong processing that listing" in result.stdout
        assert "Goodbye" in result.stdout


class TestReadPastedJobListings:
    """Unit tests for the paste reader (typed sentinels or Ctrl+D) against mocked `input()`.

    A real terminal's Ctrl+D is a "soft", per-read EOF -- the process can
    call `input()` again afterward and keep going, unlike a pipe/file EOF
    (which is permanent). `CliRunner`'s piped `input=` behaves like the
    latter, so a multi-job chat session can only be exercised at this level,
    by mocking `input()` itself with a sequence of return values and
    `EOFError`s, not through a full `runner.invoke(...)` call.
    """

    def test_sentinel_on_its_own_line_submits(self, mocker: MockerFixture) -> None:
        mocker.patch("builtins.input", side_effect=["line one", "line two", "SUBMITJOB"])

        assert _read_pasted_job_listings() == ["line one\nline two"]

    def test_sentinel_merged_onto_last_pasted_line_still_submits(
        self, mocker: MockerFixture
    ) -> None:
        # Simulates a real paste with no trailing newline, followed by typing
        # the sentinel immediately after with no separating Enter -- the
        # exact bug an exact-line-match sentinel ("END") missed.
        mocker.patch(
            "builtins.input", side_effect=["line one", "line two no trailing newlineSUBMITJOB"]
        )

        assert _read_pasted_job_listings() == ["line one\nline two no trailing newline"]

    def test_sentinel_matching_is_case_insensitive(self, mocker: MockerFixture) -> None:
        mocker.patch("builtins.input", side_effect=["some text", "submitjob"])

        assert _read_pasted_job_listings() == ["some text"]

    def test_single_line_paste_submitted_by_eof(self, mocker: MockerFixture) -> None:
        mocker.patch("builtins.input", side_effect=["Some job text", EOFError()])

        assert _read_pasted_job_listings() == ["Some job text"]

    def test_multi_line_paste_submitted_by_eof(self, mocker: MockerFixture) -> None:
        mocker.patch("builtins.input", side_effect=["line one", "line two", EOFError()])

        assert _read_pasted_job_listings() == ["line one\nline two"]

    def test_immediate_eof_with_nothing_typed_returns_none(self, mocker: MockerFixture) -> None:
        mocker.patch("builtins.input", side_effect=EOFError())

        assert _read_pasted_job_listings() is None

    def test_quit_as_first_line_returns_none(self, mocker: MockerFixture) -> None:
        mocker.patch("builtins.input", side_effect=["quit"])

        assert _read_pasted_job_listings() is None

    def test_exit_as_first_line_is_case_insensitive(self, mocker: MockerFixture) -> None:
        mocker.patch("builtins.input", side_effect=["EXIT"])

        assert _read_pasted_job_listings() is None

    def test_two_separate_calls_each_get_their_own_paste(self, mocker: MockerFixture) -> None:
        input_mock = mocker.patch(
            "builtins.input", side_effect=["job one text", EOFError(), "job two text", EOFError()]
        )

        assert _read_pasted_job_listings() == ["job one text"]
        assert _read_pasted_job_listings() == ["job two text"]
        assert input_mock.call_count == 4

    def test_next_job_sentinel_queues_multiple_listings_in_one_batch(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "builtins.input",
            side_effect=["job one text", "NEXTJOB", "job two text", "SUBMITJOB"],
        )

        assert _read_pasted_job_listings() == ["job one text", "job two text"]

    def test_next_job_sentinel_merged_onto_last_pasted_line(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "builtins.input",
            side_effect=["job one textNEXTJOB", "job two text", EOFError()],
        )

        assert _read_pasted_job_listings() == ["job one text", "job two text"]

    def test_three_listings_via_two_next_job_sentinels(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "builtins.input",
            side_effect=[
                "job one",
                "NEXTJOB",
                "job two",
                "nextjob",
                "job three",
                "SUBMITJOB",
            ],
        )

        assert _read_pasted_job_listings() == ["job one", "job two", "job three"]

    def test_empty_chunk_between_next_job_sentinels_is_dropped(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "builtins.input",
            side_effect=["job one", "NEXTJOB", "NEXTJOB", "job two", "SUBMITJOB"],
        )

        assert _read_pasted_job_listings() == ["job one", "job two"]

    def test_submit_with_nothing_pasted_returns_empty_list_not_none(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch("builtins.input", side_effect=["SUBMITJOB"])

        assert _read_pasted_job_listings() == []

    def test_quit_after_a_next_job_flush_is_treated_as_pasted_text(
        self, mocker: MockerFixture
    ) -> None:
        # Once content has been queued via NEXTJOB, "quit" is no longer a
        # special leave-the-chat command -- it would be ambiguous.
        mocker.patch(
            "builtins.input",
            side_effect=["job one", "NEXTJOB", "quit", "SUBMITJOB"],
        )

        assert _read_pasted_job_listings() == ["job one", "quit"]
