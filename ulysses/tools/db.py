"""Async SQLite persistence layer. All database queries live in this module.

Uses SQLModel over aiosqlite. The `Job` table's primary columns are a flattened,
queryable subset of the richer Pydantic domain models in `ulysses.models`
(for listing/filtering); it also stores the full `JobPost`/`JobScore` payloads
as JSON so callers without access to the original graph state — e.g. the
`ulysses draft <url>` CLI command, or a Telegram button pressed long after the
job was scored — can reconstruct them via `get_full_job`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from ulysses.models import JobPost, JobScore

__all__ = [
    "Job",
    "JobStatus",
    "ProposalDraft",
    "PrototypeFile",
    "UlyssesDB",
    "sync_read_menubar_stats",
]


class JobStatus(StrEnum):
    """Lifecycle status of a job as it moves through the Ulysses pipeline."""

    NEW = "new"
    NOTIFIED = "notified"
    DRAFTED = "drafted"
    BUILT = "built"
    SKIPPED = "skipped"
    ARCHIVED = "archived"
    WON = "won"
    LOST = "lost"


class Job(SQLModel, table=True):
    """A scored Upwork job, as persisted for deduplication and status tracking."""

    id: str = Field(primary_key=True)
    title: str
    description: str
    url: str = Field(unique=True, index=True)
    score: float
    category: str
    status: JobStatus = Field(default=JobStatus.NEW)
    posted_at: datetime
    seen_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    job_json: str | None = None
    score_json: str | None = None


class ProposalDraft(SQLModel, table=True):
    """A generated proposal draft for a job (Phase 2+)."""

    id: int | None = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PrototypeFile(SQLModel, table=True):
    """A single generated file belonging to a job's demo prototype (Phase 3+)."""

    id: int | None = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    filename: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class UlyssesDB:
    """Async SQLite database handle. Every query Ulysses runs goes through here."""

    def __init__(self, db_path: Path) -> None:
        """Create a DB handle backed by the SQLite file at `db_path`.

        The engine is created eagerly, but no connection is opened and no
        tables are created until `init()` is awaited.
        """
        self._db_path = db_path
        self._engine: AsyncEngine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    async def init(self) -> None:
        """Create the parent directory (if needed), all tables, and any missing columns.

        `SQLModel.metadata.create_all` only creates tables that don't exist yet
        — it won't add new columns to a table from an older version of the
        schema. Since this project doesn't run a full migration tool (Alembic),
        `_add_missing_columns` covers the one thing that actually happens in
        practice: additive, nullable columns (like `job_json`/`score_json`)
        showing up on an existing SQLite file from a prior run.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
            await conn.run_sync(_add_missing_columns)

    async def dispose(self) -> None:
        """Dispose of the underlying connection pool."""
        await self._engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a scoped `AsyncSession` bound to this DB's engine."""
        async with AsyncSession(self._engine) as session:
            yield session

    async def job_exists(self, url: str) -> bool:
        """Return whether a job with the given URL has already been seen."""
        return await self.get_job_by_url(url) is not None

    async def upsert_job(self, job: Job) -> Job:
        """Insert a new job, or update an existing one with the same id."""
        async with self.session() as session:
            existing = await session.get(Job, job.id)
            if existing is not None:
                for field_name in (
                    "title",
                    "description",
                    "url",
                    "score",
                    "category",
                    "posted_at",
                    "job_json",
                    "score_json",
                ):
                    setattr(existing, field_name, getattr(job, field_name))
                session.add(existing)
                await session.commit()
                await session.refresh(existing)
                return existing
            session.add(job)
            await session.commit()
            await session.refresh(job)
            return job

    async def get_job(self, job_id: str) -> Job | None:
        """Fetch a job by its primary key, or `None` if not found."""
        async with self.session() as session:
            return await session.get(Job, job_id)

    async def get_job_by_url(self, url: str) -> Job | None:
        """Fetch a job by its Upwork URL, or `None` if not found."""
        async with self.session() as session:
            result = await session.exec(select(Job).where(Job.url == url))
            return result.first()

    async def get_full_job(self, job_id: str) -> tuple[JobPost, JobScore] | None:
        """Reconstruct the full `JobPost`/`JobScore` for a job, if it was stored.

        Returns `None` if the job doesn't exist, or if it predates the
        `job_json`/`score_json` columns being populated.
        """
        job = await self.get_job(job_id)
        if job is None or job.job_json is None or job.score_json is None:
            return None
        return JobPost.model_validate_json(job.job_json), JobScore.model_validate_json(
            job.score_json
        )

    async def list_jobs(
        self,
        *,
        min_score: float = 0.0,
        category: str | None = None,
        status: JobStatus | None = None,
    ) -> list[Job]:
        """List jobs matching the given filters, most recently seen first."""
        async with self.session() as session:
            query = select(Job).where(Job.score >= min_score)
            if category is not None:
                query = query.where(Job.category == category)
            if status is not None:
                query = query.where(Job.status == status)
            query = query.order_by(Job.seen_at.desc())
            result = await session.exec(query)
            return list(result.all())

    async def update_status(self, job_id: str, status: JobStatus) -> None:
        """Update a job's lifecycle status."""
        async with self.session() as session:
            job = await session.get(Job, job_id)
            if job is None:
                return
            job.status = status
            session.add(job)
            await session.commit()

    async def add_proposal_draft(self, job_id: str, content: str) -> ProposalDraft:
        """Persist a generated proposal draft for a job."""
        async with self.session() as session:
            draft = ProposalDraft(job_id=job_id, content=content)
            session.add(draft)
            await session.commit()
            await session.refresh(draft)
            return draft

    async def add_prototype_file(self, job_id: str, filename: str, content: str) -> PrototypeFile:
        """Persist a single generated prototype file for a job."""
        async with self.session() as session:
            prototype_file = PrototypeFile(job_id=job_id, filename=filename, content=content)
            session.add(prototype_file)
            await session.commit()
            await session.refresh(prototype_file)
            return prototype_file

    async def get_proposal_drafts(self, job_id: str) -> list[ProposalDraft]:
        """List all proposal drafts generated for a job, oldest first."""
        async with self.session() as session:
            result = await session.exec(
                select(ProposalDraft)
                .where(ProposalDraft.job_id == job_id)
                .order_by(ProposalDraft.created_at)
            )
            return list(result.all())

    async def get_prototype_files(self, job_id: str) -> list[PrototypeFile]:
        """List all prototype files generated for a job, oldest first."""
        async with self.session() as session:
            result = await session.exec(
                select(PrototypeFile)
                .where(PrototypeFile.job_id == job_id)
                .order_by(PrototypeFile.created_at)
            )
            return list(result.all())

    async def stats(self) -> dict[str, int]:
        """Return summary counts used by `ulysses status`."""
        async with self.session() as session:
            all_jobs = list((await session.exec(select(Job))).all())
        counts = {status.value: 0 for status in JobStatus}
        for job in all_jobs:
            counts[job.status.value] += 1
        counts["total"] = len(all_jobs)
        return counts


def _add_missing_columns(sync_conn: Connection) -> None:
    """Add any column present in the models but missing from the live SQLite tables."""
    inspector = inspect(sync_conn)
    existing_tables = set(inspector.get_table_names())
    for table in SQLModel.metadata.tables.values():
        if table.name not in existing_tables:
            continue
        existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing_columns:
                continue
            column_type = column.type.compile(sync_conn.dialect)
            sync_conn.execute(
                text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column_type}')
            )


def sync_read_menubar_stats(db_path: Path) -> dict[str, int]:
    """Read menu-bar summary stats via a plain sync `sqlite3` connection.

    Only for the rumps menu bar app's periodic UI refresh timer, which runs
    on a different thread than the main asyncio event loop -- using the
    async `UlyssesDB` API there would require cross-thread coroutine
    scheduling for what's just a few small read-only COUNT queries.
    "Jobs today" is computed in UTC, matching how the rest of Ulysses stores
    timestamps.
    """
    if not db_path.exists():
        return {"jobs_today": 0, "proposals_drafted": 0, "prototypes_built": 0}

    with sqlite3.connect(db_path) as conn:
        jobs_today = conn.execute(
            "SELECT COUNT(*) FROM job WHERE date(seen_at) = date('now')"
        ).fetchone()[0]
        proposals_drafted = conn.execute(
            "SELECT COUNT(DISTINCT job_id) FROM proposaldraft"
        ).fetchone()[0]
        prototypes_built = conn.execute(
            "SELECT COUNT(DISTINCT job_id) FROM prototypefile"
        ).fetchone()[0]

    return {
        "jobs_today": jobs_today,
        "proposals_drafted": proposals_drafted,
        "prototypes_built": prototypes_built,
    }
