"""Tests for the async SQLite persistence layer in `hermes.tools.db`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from hermes.tools.db import HermesDB, Job, JobStatus


@pytest.fixture
async def db(tmp_path: Path) -> HermesDB:
    database = HermesDB(tmp_path / "hermes-test.db")
    await database.init()
    yield database
    await database.dispose()


def _job(job_id: str = "job-1", **overrides: object) -> Job:
    defaults = dict(
        id=job_id,
        title="Python scraper",
        description="Scrape listings",
        url=f"https://www.upwork.com/jobs/~{job_id}",
        score=80.0,
        category="tier1",
        status=JobStatus.NEW,
        posted_at=datetime.now(UTC),
    )
    defaults.update(overrides)
    return Job(**defaults)


class TestUpsertAndGet:
    async def test_upsert_inserts_new_job(self, db: HermesDB) -> None:
        job = await db.upsert_job(_job())
        assert job.id == "job-1"

    async def test_get_job_by_id(self, db: HermesDB) -> None:
        await db.upsert_job(_job())
        fetched = await db.get_job("job-1")
        assert fetched is not None
        assert fetched.title == "Python scraper"

    async def test_get_job_returns_none_when_missing(self, db: HermesDB) -> None:
        assert await db.get_job("does-not-exist") is None

    async def test_get_job_by_url(self, db: HermesDB) -> None:
        await db.upsert_job(_job())
        fetched = await db.get_job_by_url("https://www.upwork.com/jobs/~job-1")
        assert fetched is not None
        assert fetched.id == "job-1"

    async def test_upsert_updates_existing_job(self, db: HermesDB) -> None:
        await db.upsert_job(_job())
        updated = await db.upsert_job(_job(score=95.0, category="tier1"))
        assert updated.score == 95.0
        all_jobs = await db.list_jobs()
        assert len(all_jobs) == 1


class TestJobExists:
    async def test_true_for_seen_url(self, db: HermesDB) -> None:
        await db.upsert_job(_job())
        assert await db.job_exists("https://www.upwork.com/jobs/~job-1") is True

    async def test_false_for_unseen_url(self, db: HermesDB) -> None:
        assert await db.job_exists("https://www.upwork.com/jobs/~unseen") is False


class TestListJobs:
    async def test_filters_by_min_score(self, db: HermesDB) -> None:
        await db.upsert_job(_job("low", score=20.0))
        await db.upsert_job(_job("high", score=90.0))
        results = await db.list_jobs(min_score=50.0)
        assert {job.id for job in results} == {"high"}

    async def test_filters_by_category(self, db: HermesDB) -> None:
        await db.upsert_job(_job("t1", category="tier1"))
        await db.upsert_job(_job("t3", category="tier3"))
        results = await db.list_jobs(category="tier1")
        assert {job.id for job in results} == {"t1"}

    async def test_filters_by_status(self, db: HermesDB) -> None:
        await db.upsert_job(_job("new-job", status=JobStatus.NEW))
        await db.upsert_job(_job("skipped-job", status=JobStatus.SKIPPED))
        results = await db.list_jobs(status=JobStatus.SKIPPED)
        assert {job.id for job in results} == {"skipped-job"}


class TestUpdateStatus:
    async def test_updates_status(self, db: HermesDB) -> None:
        await db.upsert_job(_job())
        await db.update_status("job-1", JobStatus.ARCHIVED)
        job = await db.get_job("job-1")
        assert job.status == JobStatus.ARCHIVED

    async def test_no_op_for_missing_job(self, db: HermesDB) -> None:
        await db.update_status("does-not-exist", JobStatus.ARCHIVED)  # should not raise


class TestProposalDraftsAndPrototypeFiles:
    async def test_add_and_get_proposal_drafts(self, db: HermesDB) -> None:
        await db.upsert_job(_job())
        await db.add_proposal_draft("job-1", "Draft one")
        await db.add_proposal_draft("job-1", "Draft two")
        drafts = await db.get_proposal_drafts("job-1")
        assert [draft.content for draft in drafts] == ["Draft one", "Draft two"]

    async def test_add_and_get_prototype_files(self, db: HermesDB) -> None:
        await db.upsert_job(_job())
        await db.add_prototype_file("job-1", "demo.py", "print('hi')")
        files = await db.get_prototype_files("job-1")
        assert len(files) == 1
        assert files[0].filename == "demo.py"


class TestStats:
    async def test_counts_by_status_and_total(self, db: HermesDB) -> None:
        await db.upsert_job(_job("a", status=JobStatus.NEW))
        await db.upsert_job(_job("b", status=JobStatus.NEW))
        await db.upsert_job(_job("c", status=JobStatus.SKIPPED))
        stats = await db.stats()
        assert stats["new"] == 2
        assert stats["skipped"] == 1
        assert stats["total"] == 3
