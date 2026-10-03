"""Tests for `ulysses.agents.scout.ScoutAgent`, with a mocked `EmailReader`."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ulysses.agents.scout import ScoutAgent, _parse_received_at
from ulysses.config.profile import Profile
from ulysses.tools.db import UlyssesDB
from ulysses.tools.email_reader import RawEmail

VALID_EMAIL_HTML = """
<html><body>
<a href="https://www.upwork.com/jobs/~0112345678901234">Python scraper for real estate listings</a>
<p>We need someone to build a scraper that pulls real estate listings daily.</p>
<div>Budget: $150 fixed price</div>
<div>Skills: <ul><li>Python</li><li>Web Scraping</li></ul></div>
<div>Posted 5 minutes ago</div>
</body></html>
"""

STALE_EMAIL_HTML = """
<html><body>
<a href="https://www.upwork.com/jobs/~0199999999999999">A job posted long ago</a>
<p>We need someone to build a scraper that pulls real estate listings daily.</p>
<div>Budget: $150 fixed price</div>
</body></html>
"""

UNPARSEABLE_EMAIL_HTML = "<html><body><p>Not a job posting at all.</p></body></html>"


class TestParseReceivedAt:
    def test_parses_a_valid_rfc_2822_date(self) -> None:
        parsed = _parse_received_at("Wed, 29 Jul 2026 17:16:21 +0000")
        assert parsed == datetime(2026, 7, 29, 17, 16, 21, tzinfo=UTC)

    def test_returns_none_for_an_empty_header(self) -> None:
        assert _parse_received_at("") is None

    def test_returns_none_for_a_malformed_header(self) -> None:
        assert _parse_received_at("not a real date") is None

    def test_assumes_utc_when_the_header_has_no_timezone(self) -> None:
        parsed = _parse_received_at("Wed, 29 Jul 2026 17:16:21")
        assert parsed is not None
        assert parsed.tzinfo is UTC


@pytest.fixture
async def db(tmp_path: Path) -> UlyssesDB:
    database = UlyssesDB(tmp_path / "scout-test.db")
    await database.init()
    yield database
    await database.dispose()


def _reader_returning(*raw_emails: RawEmail) -> AsyncMock:
    reader = AsyncMock()
    reader.fetch_new_upwork_emails = AsyncMock(return_value=list(raw_emails))
    return reader


class TestRunOnce:
    async def test_scores_and_persists_new_job(self, db: UlyssesDB, profile: Profile) -> None:
        reader = _reader_returning(RawEmail("1", "subj", VALID_EMAIL_HTML, ""))
        scout = ScoutAgent(email_reader=reader, db=db, profile=profile)

        results = await scout.run_once()

        assert len(results) == 1
        job, score = results[0]
        assert job.title == "Python scraper for real estate listings"
        assert score.total_score > 0
        stored = await db.get_job(job.id)
        assert stored is not None
        assert stored.score == score.total_score

        full = await db.get_full_job(job.id)
        assert full is not None
        restored_job, restored_score = full
        assert restored_job.title == job.title
        assert restored_score.total_score == score.total_score

    async def test_skips_already_seen_jobs(self, db: UlyssesDB, profile: Profile) -> None:
        reader = _reader_returning(RawEmail("1", "subj", VALID_EMAIL_HTML, ""))
        scout = ScoutAgent(email_reader=reader, db=db, profile=profile)

        first_pass = await scout.run_once()
        second_pass = await scout.run_once()

        assert len(first_pass) == 1
        assert len(second_pass) == 0

    async def test_skips_unparseable_emails(self, db: UlyssesDB, profile: Profile) -> None:
        reader = _reader_returning(RawEmail("2", "subj", UNPARSEABLE_EMAIL_HTML, ""))
        scout = ScoutAgent(email_reader=reader, db=db, profile=profile)

        results = await scout.run_once()

        assert results == []

    async def test_returns_empty_list_when_no_new_emails(
        self, db: UlyssesDB, profile: Profile
    ) -> None:
        reader = _reader_returning()
        scout = ScoutAgent(email_reader=reader, db=db, profile=profile)

        assert await scout.run_once() == []

    async def test_skips_jobs_older_than_skip_if_posted_hours_ago(
        self, db: UlyssesDB, profile: Profile
    ) -> None:
        # No "posted N ago" phrase in STALE_EMAIL_HTML, so posted_at falls
        # back entirely to this header -- `profile.scoring.skip_if_posted_hours_ago`
        # is 6 (see conftest.py), well under this 48-hour-old email.
        stale_header = format_datetime(datetime.now(UTC) - timedelta(hours=48))
        reader = _reader_returning(RawEmail("3", "subj", STALE_EMAIL_HTML, stale_header))
        scout = ScoutAgent(email_reader=reader, db=db, profile=profile)

        results = await scout.run_once()

        assert results == []
        stored = await db.get_job("0199999999999999")
        assert stored is None

    async def test_processes_jobs_within_skip_if_posted_hours_ago(
        self, db: UlyssesDB, profile: Profile
    ) -> None:
        fresh_header = format_datetime(datetime.now(UTC) - timedelta(hours=1))
        reader = _reader_returning(RawEmail("4", "subj", STALE_EMAIL_HTML, fresh_header))
        scout = ScoutAgent(email_reader=reader, db=db, profile=profile)

        results = await scout.run_once()

        assert len(results) == 1


class TestRunForever:
    async def test_invokes_callback_for_each_scored_job_then_sleeps(
        self, db: UlyssesDB, profile: Profile, mocker
    ) -> None:
        reader = _reader_returning(RawEmail("1", "subj", VALID_EMAIL_HTML, ""))
        scout = ScoutAgent(email_reader=reader, db=db, profile=profile)
        on_scored_job = AsyncMock()

        async def stop_after_sleep(_seconds: float) -> None:
            raise asyncio.CancelledError

        mocker.patch("asyncio.sleep", side_effect=stop_after_sleep)

        with pytest.raises(asyncio.CancelledError):
            await scout.run_forever(poll_interval_seconds=1, on_scored_job=on_scored_job)

        on_scored_job.assert_awaited_once()

    async def test_recovers_from_run_once_exception(
        self, db: UlyssesDB, profile: Profile, mocker
    ) -> None:
        scout = ScoutAgent(email_reader=AsyncMock(), db=db, profile=profile)
        mocker.patch.object(scout, "run_once", AsyncMock(side_effect=RuntimeError("boom")))
        on_scored_job = AsyncMock()

        async def stop_after_sleep(_seconds: float) -> None:
            raise asyncio.CancelledError

        mocker.patch("asyncio.sleep", side_effect=stop_after_sleep)

        with pytest.raises(asyncio.CancelledError):
            await scout.run_forever(poll_interval_seconds=1, on_scored_job=on_scored_job)

        on_scored_job.assert_not_awaited()

    async def test_stop_event_ends_the_loop_immediately(
        self, db: UlyssesDB, profile: Profile
    ) -> None:
        scout = ScoutAgent(email_reader=AsyncMock(), db=db, profile=profile)
        stop_event = threading.Event()
        stop_event.set()

        # Should return without ever polling or sleeping -- no mocking needed.
        await scout.run_forever(
            poll_interval_seconds=1, on_scored_job=AsyncMock(), stop_event=stop_event
        )

    async def test_paused_event_skips_polling(
        self, db: UlyssesDB, profile: Profile, mocker
    ) -> None:
        reader = _reader_returning(RawEmail("1", "subj", VALID_EMAIL_HTML, ""))
        scout = ScoutAgent(email_reader=reader, db=db, profile=profile)
        on_scored_job = AsyncMock()
        paused_event = threading.Event()
        paused_event.set()

        async def stop_after_sleep(_seconds: float) -> None:
            raise asyncio.CancelledError

        mocker.patch("asyncio.sleep", side_effect=stop_after_sleep)

        with pytest.raises(asyncio.CancelledError):
            await scout.run_forever(
                poll_interval_seconds=1, on_scored_job=on_scored_job, paused_event=paused_event
            )

        on_scored_job.assert_not_awaited()
        reader.fetch_new_upwork_emails.assert_not_awaited()
