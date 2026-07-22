"""Tests for the LangGraph node factories in `ulysses.graph.nodes`."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ulysses.agents.scorer import score_job
from ulysses.config.profile import Profile
from ulysses.graph.nodes import (
    build_done_node,
    build_notifier_node,
    build_proposal_node,
    build_prototype_node,
    build_scorer_node,
    build_scout_node,
)
from ulysses.graph.state import UlyssesState
from ulysses.models import GeneratedProposal, GeneratedPrototype, JobPost


def _state(job: JobPost, **overrides: object) -> UlyssesState:
    base: UlyssesState = {
        "job": job,
        "score": None,
        "user_action": None,
        "proposal_draft": None,
        "prototype_files": None,
        "notification_sent": False,
        "completed": False,
    }
    base.update(overrides)
    return base


class TestScoutNode:
    async def test_passes_state_through_unchanged(self, fresh_job: JobPost) -> None:
        node = build_scout_node()
        state = _state(fresh_job)
        result = await node(state)
        assert result == state


class TestScorerNode:
    async def test_populates_score_in_state(self, fresh_job: JobPost, profile: Profile) -> None:
        node = build_scorer_node(profile)
        result = await node(_state(fresh_job))
        assert result["score"] is not None
        assert result["score"].total_score > 0


class TestNotifierNode:
    async def test_calls_notifier_and_marks_notification_sent(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        notifier = AsyncMock()
        node = build_notifier_node(notifier, profile)

        result = await node(_state(fresh_job, score=score))

        notifier.handle_scored_job.assert_awaited_once_with(fresh_job, score, profile.scoring)
        assert result["notification_sent"] is True

    async def test_raises_if_score_missing(self, fresh_job: JobPost, profile: Profile) -> None:
        notifier = AsyncMock()
        node = build_notifier_node(notifier, profile)
        with pytest.raises(ValueError, match="score"):
            await node(_state(fresh_job, score=None))


class TestProposalNode:
    async def test_generates_persists_sends_and_updates_state(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        generated = GeneratedProposal(
            job_id=fresh_job.id,
            category="automation",
            hook="hook text",
            plan_bullets=["one", "two", "three"],
            proof_repo="repo",
            proof_repo_url="https://github.com/Alejog20/repo",
            timeline="3 days",
            bid_usd=150.0,
            full_text="the full proposal text",
        )
        proposal_agent = AsyncMock()
        proposal_agent.generate = AsyncMock(return_value=generated)
        notifier = AsyncMock()
        db = AsyncMock()

        node = build_proposal_node(proposal_agent, notifier, db, profile)
        result = await node(_state(fresh_job, score=score))

        proposal_agent.generate.assert_awaited_once_with(fresh_job, score, profile)
        db.add_proposal_draft.assert_awaited_once_with(fresh_job.id, "the full proposal text")
        notifier.send_proposal_draft.assert_awaited_once_with(
            fresh_job.id, "the full proposal text"
        )
        assert result["proposal_draft"] == "the full proposal text"

    async def test_raises_if_score_missing(self, fresh_job: JobPost, profile: Profile) -> None:
        node = build_proposal_node(AsyncMock(), AsyncMock(), AsyncMock(), profile)
        with pytest.raises(ValueError, match="score"):
            await node(_state(fresh_job, score=None))


class TestPrototypeNode:
    async def test_generates_persists_sends_and_updates_state(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        generated = GeneratedPrototype(
            job_id=fresh_job.id,
            category="scraper",
            demo_script="print('hi')",
            requirements_txt="requests==2.32.3\n",
            readme_md="# Demo\n",
            config_example_env="# none needed\n",
            zip_filename=f"ulysses_demo_{fresh_job.id}.zip",
        )
        prototype_agent = AsyncMock()
        prototype_agent.generate = AsyncMock(return_value=generated)
        notifier = AsyncMock()
        db = AsyncMock()

        node = build_prototype_node(prototype_agent, notifier, db, profile)
        result = await node(_state(fresh_job, score=score))

        prototype_agent.generate.assert_awaited_once_with(fresh_job, score, profile)
        assert db.add_prototype_file.await_count == 4
        notifier.send_prototype_zip.assert_awaited_once()
        assert notifier.send_prototype_zip.call_args.args[0] == fresh_job.id
        assert notifier.send_prototype_zip.call_args.args[1] == generated
        assert result["prototype_files"] == {
            "demo.py": "print('hi')",
            "requirements.txt": "requests==2.32.3\n",
            "README.md": "# Demo\n",
            "config.example.env": "# none needed\n",
        }

    async def test_raises_if_score_missing(self, fresh_job: JobPost, profile: Profile) -> None:
        node = build_prototype_node(AsyncMock(), AsyncMock(), AsyncMock(), profile)
        with pytest.raises(ValueError, match="score"):
            await node(_state(fresh_job, score=None))


class TestDoneNode:
    async def test_marks_completed(self, fresh_job: JobPost) -> None:
        node = build_done_node()
        result = await node(_state(fresh_job))
        assert result["completed"] is True
