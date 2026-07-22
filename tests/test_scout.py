"""Tests for `ulysses.agents.scout.ScoutAgent`, with a mocked `EmailReader`."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ulysses.agents.scout import ScoutAgent
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

UNPARSEABLE_EMAIL_HTML = "<html><body><p>Not a job posting at all.</p></body></html>"


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
