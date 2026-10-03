"""Tests for `ulysses.agents.chat`: the Chat Agent and its context-digest helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ulysses.agents.chat import (
    CHAT_SYSTEM_PROMPT,
    ChatAgent,
    build_advisor_profile_summary,
    build_job_context_message,
    build_job_lookup_message,
    build_queue_digest,
)
from ulysses.agents.scorer import score_job
from ulysses.config.profile import Profile
from ulysses.models import GeneratedProposal, GeneratedPrototype, JobPost, JobScore
from ulysses.tools.db import Job, JobStatus, UlyssesDB


def _mock_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.bind = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content=content))
    return llm


class TestChatAgentReply:
    async def test_returns_stripped_reply_text(self) -> None:
        llm = _mock_llm("  Sure, here's a shorter hook.  ")
        agent = ChatAgent(llm=llm)

        reply = await agent.reply([], "Make the hook shorter.", system_prompt=CHAT_SYSTEM_PROMPT)

        assert reply == "Sure, here's a shorter hook."

    async def test_sends_system_prompt_then_history_then_new_message(self) -> None:
        llm = _mock_llm("ok")
        agent = ChatAgent(llm=llm)
        history = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]

        await agent.reply(history, "second question", system_prompt="PERSONA")

        sent = llm.ainvoke.await_args.args[0]
        assert sent == [
            {"role": "system", "content": "PERSONA"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]

    async def test_does_not_mutate_the_callers_history_list(self) -> None:
        llm = _mock_llm("ok")
        agent = ChatAgent(llm=llm)
        history = [{"role": "user", "content": "hi"}]

        await agent.reply(history, "next", system_prompt=CHAT_SYSTEM_PROMPT)

        assert history == [{"role": "user", "content": "hi"}]

    async def test_binds_cost_control_kwargs(self) -> None:
        llm = _mock_llm("ok")
        agent = ChatAgent(llm=llm)

        await agent.reply([], "hi", system_prompt=CHAT_SYSTEM_PROMPT)

        _, kwargs = llm.bind.call_args
        assert kwargs["thinking"] == {"type": "disabled"}
        assert kwargs["cache_control"] == {"type": "ephemeral"}
        assert isinstance(kwargs["max_tokens"], int)
        assert kwargs["max_tokens"] <= 1000


class TestBuildJobContextMessage:
    def test_includes_core_job_and_score_facts(self, fresh_job: JobPost, profile: Profile) -> None:
        score = score_job(fresh_job, profile)

        message = build_job_context_message(fresh_job, score, None, None)

        assert fresh_job.title in message
        assert fresh_job.url in message
        assert f"{score.total_score:.0f}" in message
        assert score.recommendation.value in message

    def test_posted_hours_ago_renders_as_hours(self, fresh_job: JobPost, profile: Profile) -> None:
        job = fresh_job.model_copy(update={"posted_at": datetime.now(UTC) - timedelta(hours=5)})
        score = score_job(job, profile)

        message = build_job_context_message(job, score, None, None)

        assert "Posted: 5 hours ago" in message

    def test_no_red_flags_renders_as_none(self, fresh_job: JobPost, profile: Profile) -> None:
        score = score_job(fresh_job, profile)
        assert score.red_flags == []

        message = build_job_context_message(fresh_job, score, None, None)

        assert "Red flags: none" in message

    def test_unknown_proposals_count_renders_as_not_shown(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        job = fresh_job.model_copy(update={"proposals_count": None})
        score = score_job(job, profile)

        message = build_job_context_message(job, score, None, None)

        assert "Proposals so far: not shown" in message

    def test_omits_proposal_and_prototype_sections_when_absent(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)

        message = build_job_context_message(fresh_job, score, None, None)

        assert "Current proposal draft" not in message
        assert "demo prototype" not in message

    def test_includes_proposal_full_text_when_present(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        proposal = GeneratedProposal(
            job_id=fresh_job.id,
            category="scraping",
            hook="hook",
            plan_bullets=["step one"],
            close="close",
            proof_repo="repo",
            proof_repo_url="https://example.com/repo",
            timeline="3-5 days",
            bid_usd=150.0,
            full_text="The full proposal text goes here.",
        )

        message = build_job_context_message(fresh_job, score, proposal, None)

        assert "The full proposal text goes here." in message
        assert "Current proposal draft (scraping):" in message

    def test_includes_prototype_readme_when_present(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        prototype = GeneratedPrototype(
            job_id=fresh_job.id,
            category="scraper",
            demo_script="print('demo')",
            requirements_txt="requests",
            readme_md="# Demo\nThis scrapes listings.",
            config_example_env="API_KEY=",
            zip_filename="demo.zip",
        )

        message = build_job_context_message(fresh_job, score, None, prototype)

        assert "# Demo\nThis scrapes listings." in message
        assert "print('demo')" not in message  # demo source itself is deliberately omitted


class TestBuildQueueDigest:
    def _job(self, **overrides: object) -> Job:
        defaults: dict[str, object] = {
            "id": "job-1",
            "title": "A job",
            "description": "desc",
            "url": "https://example.com/job-1",
            "score": 80.0,
            "category": "tier1",
            "status": JobStatus.NEW,
            "posted_at": datetime.now(UTC) - timedelta(minutes=5),
        }
        defaults.update(overrides)
        return Job(**defaults)

    def test_empty_queue_renders_a_plain_message(self) -> None:
        assert build_queue_digest([]) == "The job queue is currently empty."

    def test_renders_one_line_per_job_with_key_fields(self) -> None:
        job = self._job()

        digest = build_queue_digest([job])

        assert "[80]" in digest
        assert "tier1" in digest
        assert "new" in digest
        assert job.title in digest
        assert job.url in digest

    def test_caps_to_the_given_limit(self) -> None:
        jobs = [self._job(id=f"job-{i}", url=f"https://example.com/job-{i}") for i in range(5)]

        digest = build_queue_digest(jobs, limit=2)

        assert digest.count("- [") == 2
        assert "2 of 5 shown" in digest


class TestBuildAdvisorProfileSummary:
    def test_includes_rate_skills_and_target_budget(self, profile: Profile) -> None:
        summary = build_advisor_profile_summary(profile)

        assert f"${profile.freelancer.rate_usd_hr:.0f}/hr" in summary
        assert profile.skills.primary[0] in summary
        assert f"${profile.scoring.target_budget_min:.0f}" in summary
        assert f"${profile.scoring.target_budget_max:.0f}" in summary


@pytest.fixture
async def db(tmp_path: Path) -> UlyssesDB:
    database = UlyssesDB(tmp_path / "ulysses-test.db")
    await database.init()
    yield database
    await database.dispose()


async def _seed_full_job(db: UlyssesDB, job: JobPost, score: JobScore) -> None:
    await db.upsert_job(
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


class TestBuildJobLookupMessage:
    async def test_returns_none_for_unknown_identifier(self, db: UlyssesDB) -> None:
        assert await build_job_lookup_message(db, "does-not-exist") is None

    async def test_looks_up_by_id_or_url(
        self, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_full_job(db, fresh_job, score)

        by_id = await build_job_lookup_message(db, fresh_job.id)
        by_url = await build_job_lookup_message(db, fresh_job.url)

        assert by_id is not None
        assert fresh_job.title in by_id
        assert by_url is not None
        assert fresh_job.title in by_url

    async def test_returns_none_when_job_predates_full_json_storage(self, db: UlyssesDB) -> None:
        await db.upsert_job(
            Job(
                id="pre-phase-2",
                title="An old job",
                description="desc",
                url="manual://pre-phase-2",
                score=50.0,
                category="tier2",
                status=JobStatus.NEW,
                posted_at=datetime.now(UTC),
            )
        )

        assert await build_job_lookup_message(db, "pre-phase-2") is None

    async def test_includes_most_recent_proposal_draft(
        self, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_full_job(db, fresh_job, score)
        await db.add_proposal_draft(fresh_job.id, "An older draft.")
        await db.add_proposal_draft(fresh_job.id, "The newest draft.")

        message = await build_job_lookup_message(db, fresh_job.id)

        assert message is not None
        assert "The newest draft." in message
        assert "Most recent saved proposal draft:" in message

    async def test_includes_prototype_readme_when_present(
        self, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_full_job(db, fresh_job, score)
        await db.add_prototype_file(fresh_job.id, "README.md", "# A generated demo README.")

        message = await build_job_lookup_message(db, fresh_job.id)

        assert message is not None
        assert "# A generated demo README." in message

    async def test_omits_draft_and_readme_sections_when_neither_exists(
        self, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_full_job(db, fresh_job, score)

        message = await build_job_lookup_message(db, fresh_job.id)

        assert message is not None
        assert "Most recent saved proposal draft:" not in message
        assert "Prototype README:" not in message
