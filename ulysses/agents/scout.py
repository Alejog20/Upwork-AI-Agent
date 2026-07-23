"""Scout Agent — watches email for new Upwork jobs, scores them, and persists them.

Scoring emission is done via an injected callback rather than a direct import
of the Notifier Agent, keeping agents decoupled from one another (the graph
layer is responsible for wiring scout output to the notifier).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable

from loguru import logger

from ulysses.agents.scorer import score_job
from ulysses.config.profile import Profile
from ulysses.models import JobPost, JobScore
from ulysses.tools.db import Job, JobStatus, UlyssesDB
from ulysses.tools.email_reader import EmailReader
from ulysses.tools.job_parser import parse_job_email

__all__ = ["ScoredJobCallback", "ScoutAgent"]

ScoredJobCallback = Callable[[JobPost, JobScore], Awaitable[None]]


class ScoutAgent:
    """Polls the mailbox for new Upwork jobs and pushes scored, deduped jobs onward."""

    def __init__(self, email_reader: EmailReader, db: UlyssesDB, profile: Profile) -> None:
        """Create a Scout Agent.

        Args:
            email_reader: The IMAP client used to fetch unread Upwork emails.
            db: Persistence layer, used for deduplication and storage.
            profile: The freelancer's profile, used for scoring.
        """
        self._email_reader = email_reader
        self._db = db
        self._profile = profile

    async def run_once(self) -> list[tuple[JobPost, JobScore]]:
        """Fetch, parse, score, and persist any new jobs since the last poll.

        Returns:
            The `(JobPost, JobScore)` pairs for jobs not previously seen,
            in the order they were fetched.
        """
        raw_emails = await self._email_reader.fetch_new_upwork_emails()
        logger.debug("Scout fetched {} candidate email(s)", len(raw_emails))

        scored_jobs: list[tuple[JobPost, JobScore]] = []
        for raw_email in raw_emails:
            job, error = parse_job_email(raw_email.html_body)
            if error is not None:
                logger.warning("Failed to parse email uid={}: {}", raw_email.uid, error)
                continue

            if await self._db.job_exists(job.url):
                logger.debug("Skipping already-seen job: {}", job.url)
                continue

            score = score_job(job, self._profile)
            await self._db.upsert_job(
                Job(
                    id=job.id,
                    title=job.title,
                    description=job.description,
                    url=job.url,
                    score=score.total_score,
                    category=score.gig_category.value,
                    status=JobStatus.NEW,
                    posted_at=job.posted_at,
                    job_json=job.model_dump_json(),
                    score_json=score.model_dump_json(),
                )
            )
            logger.bind(job_id=job.id, agent="scout").info(
                "Scored new job: {} ({}/100)", job.title, score.total_score
            )
            scored_jobs.append((job, score))

        return scored_jobs

    async def run_forever(
        self,
        poll_interval_seconds: int,
        on_scored_job: ScoredJobCallback,
        *,
        stop_event: threading.Event | None = None,
        paused_event: threading.Event | None = None,
    ) -> None:
        """Poll indefinitely, invoking `on_scored_job` for each newly scored job.

        `stop_event`/`paused_event` are plain `threading.Event`s (not
        `asyncio.Event`s) because the menu bar app toggles them from rumps'
        native Cocoa event loop, which runs on a different thread than this
        coroutine -- `threading.Event.is_set()` is safe to poll cross-thread.

        Args:
            poll_interval_seconds: Delay between polls.
            on_scored_job: Async callback invoked once per new `(JobPost, JobScore)`.
            stop_event: If set, the loop exits before the next poll. Defaults
                to never stopping (plain CLI usage doesn't need this).
            paused_event: While set, polling is skipped for that cycle, but
                the loop keeps checking `stop_event` on schedule. Defaults to
                never paused.
        """
        while stop_event is None or not stop_event.is_set():
            if paused_event is not None and paused_event.is_set():
                logger.debug("Scout is paused; skipping this poll cycle")
            else:
                try:
                    scored_jobs = await self.run_once()
                except Exception:
                    logger.exception("Scout polling cycle failed; will retry next interval")
                    scored_jobs = []

                for job, score in scored_jobs:
                    await on_scored_job(job, score)

            await asyncio.sleep(poll_interval_seconds)
