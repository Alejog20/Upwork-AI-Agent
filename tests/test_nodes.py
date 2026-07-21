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
from ulysses.models import JobPost


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


class TestProposalAndPrototypeStubs:
    async def test_proposal_node_sets_none_draft(self, fresh_job: JobPost) -> None:
        node = build_proposal_node()
        result = await node(_state(fresh_job))
        assert result["proposal_draft"] is None

    async def test_prototype_node_sets_none_files(self, fresh_job: JobPost) -> None:
        node = build_prototype_node()
        result = await node(_state(fresh_job))
        assert result["prototype_files"] is None


class TestDoneNode:
    async def test_marks_completed(self, fresh_job: JobPost) -> None:
        node = build_done_node()
        result = await node(_state(fresh_job))
        assert result["completed"] is True
