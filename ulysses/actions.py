"""Surface-agnostic job actions shared by the Telegram bot, the CLI, and the dashboard.

Each function here does one unit of real work (generate a draft, mark a job
skipped, record an outcome) against the DB/agents directly, independent of
how it was triggered. Callers -- `ulysses.cli.main`'s Telegram handler
closures and CLI commands, `ulysses.dashboard.api`'s HTTP routes -- decide
how to report a `JobNotFoundError` and whether to also notify (e.g. via
Telegram). Deliberately its own module rather than living in `cli.main`:
`ulysses.dashboard.api` needs these functions, and `cli.main` needs to build
the dashboard app, so neither of those two modules can import from the
other without a cycle -- this module sits below both.
"""

from __future__ import annotations

from ulysses.agents.proposal import ProposalAgent
from ulysses.agents.prototype import PrototypeAgent, build_prototype_zip
from ulysses.config.profile import Profile
from ulysses.models import GeneratedProposal, GeneratedPrototype, JobNotFoundError
from ulysses.tools.db import Job, JobStatus, Outcome, UlyssesDB
from ulysses.tools.events import DashboardEventBus

__all__ = [
    "archive_job",
    "build_job",
    "draft_job",
    "persist_prototype_files",
    "record_job_outcome",
    "skip_job",
]


async def persist_prototype_files(
    db: UlyssesDB, job_id: str, prototype: GeneratedPrototype
) -> None:
    """Write a generated prototype's four files to the DB as `PrototypeFile` rows."""
    for filename, content in (
        ("demo.py", prototype.demo_script),
        ("requirements.txt", prototype.requirements_txt),
        ("README.md", prototype.readme_md),
        ("config.example.env", prototype.config_example_env),
    ):
        await db.add_prototype_file(job_id, filename, content)


async def draft_job(
    db: UlyssesDB,
    proposal_agent: ProposalAgent,
    profile: Profile,
    job_id: str,
    *,
    events: DashboardEventBus | None = None,
) -> GeneratedProposal:
    """Generate and persist a proposal draft for an already-scored job.

    Used by both the Telegram Draft/Regenerate button (via
    `ulysses.cli.main._make_draft_handler`) and the dashboard's
    `POST /api/jobs/{id}/draft` route. Raises `JobNotFoundError` if `job_id`
    has no detailed (job_json/score_json) row to draft from -- callers
    decide how to report that themselves.
    """
    full = await db.get_full_job(job_id)
    if full is None:
        raise JobNotFoundError(job_id)
    job, score = full
    draft = await proposal_agent.generate(job, score, profile)
    await db.add_proposal_draft(job_id, draft.full_text)
    await db.update_status(job_id, JobStatus.DRAFTED)
    if events is not None:
        await events.broadcast({"type": "job_updated", "job_id": job_id})
    return draft


async def build_job(
    db: UlyssesDB,
    prototype_agent: PrototypeAgent,
    profile: Profile,
    job_id: str,
    *,
    events: DashboardEventBus | None = None,
) -> tuple[GeneratedPrototype, bytes]:
    """Generate, persist, and zip a demo prototype for an already-scored job.

    Same shape as `draft_job`. Raises `JobNotFoundError` if `job_id` has no
    detailed row to build from.
    """
    full = await db.get_full_job(job_id)
    if full is None:
        raise JobNotFoundError(job_id)
    job, score = full
    prototype = await prototype_agent.generate(job, score, profile)
    await persist_prototype_files(db, job_id, prototype)
    await db.update_status(job_id, JobStatus.BUILT)
    if events is not None:
        await events.broadcast({"type": "job_updated", "job_id": job_id})
    return prototype, build_prototype_zip(prototype)


async def skip_job(db: UlyssesDB, job_id: str, *, events: DashboardEventBus | None = None) -> Job:
    """Mark a job as skipped. Raises `JobNotFoundError` if `job_id` is unknown."""
    job = await db.get_job(job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    await db.update_status(job_id, JobStatus.SKIPPED)
    if events is not None:
        await events.broadcast({"type": "job_updated", "job_id": job_id})
    job.status = JobStatus.SKIPPED
    return job


async def archive_job(
    db: UlyssesDB, job_id: str, *, events: DashboardEventBus | None = None
) -> Job:
    """Mark a job as archived. Raises `JobNotFoundError` if `job_id` is unknown."""
    job = await db.get_job(job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    await db.update_status(job_id, JobStatus.ARCHIVED)
    if events is not None:
        await events.broadcast({"type": "job_updated", "job_id": job_id})
    job.status = JobStatus.ARCHIVED
    return job


async def record_job_outcome(
    db: UlyssesDB,
    job_id: str,
    *,
    won: bool,
    value: float | None = None,
    connects_spent: int | None = None,
    note: str | None = None,
    events: DashboardEventBus | None = None,
) -> Outcome:
    """Record a job's won/lost outcome. Raises `JobNotFoundError` if `job_id` is unknown."""
    job = await db.get_job(job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    outcome = await db.record_outcome(
        job_id, won=won, contract_value_usd=value, connects_spent=connects_spent, note=note
    )
    if events is not None:
        await events.broadcast({"type": "job_updated", "job_id": job_id})
    return outcome
