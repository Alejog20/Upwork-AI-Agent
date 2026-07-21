"""Structural tests for `ulysses.graph.graph.build_graph`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from ulysses.agents.notifier import NotifierAgent
from ulysses.config.profile import Profile
from ulysses.graph.graph import build_graph


@pytest.fixture
def notifier(mocker: MockerFixture) -> NotifierAgent:
    mocker.patch("ulysses.agents.notifier.Bot")
    return NotifierAgent(bot_token="fake-token", chat_id="123456", db=MagicMock())


class TestBuildGraph:
    def test_compiled_graph_contains_all_expected_nodes(
        self, profile: Profile, notifier: NotifierAgent
    ) -> None:
        graph = build_graph(profile, notifier)
        node_names = set(graph.get_graph().nodes.keys())
        assert {"scout", "scorer", "notifier", "proposal", "prototype", "done"} <= node_names

    async def test_runs_through_to_the_notifier_interrupt(
        self, profile: Profile, notifier: NotifierAgent, fresh_job
    ) -> None:
        notifier.handle_scored_job = AsyncMock()
        graph = build_graph(profile, notifier)
        config = {"configurable": {"thread_id": fresh_job.id}}

        result = await graph.ainvoke(
            {
                "job": fresh_job,
                "score": None,
                "user_action": None,
                "proposal_draft": None,
                "prototype_files": None,
                "notification_sent": False,
                "completed": False,
            },
            config=config,
        )

        assert result["notification_sent"] is True
        assert result["score"] is not None
        notifier.handle_scored_job.assert_awaited_once()
