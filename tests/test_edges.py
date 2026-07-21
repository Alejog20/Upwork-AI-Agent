"""Tests for the conditional routing logic in `hermes.graph.edges`."""

from __future__ import annotations

from langgraph.types import Send

from hermes.graph.edges import route_user_action
from hermes.graph.state import HermesState


def _state(user_action: str | None) -> HermesState:
    return {
        "job": None,
        "score": None,
        "user_action": user_action,
        "proposal_draft": None,
        "prototype_files": None,
        "notification_sent": True,
        "completed": False,
    }


class TestRouteUserAction:
    def test_draft_routes_to_proposal(self) -> None:
        assert route_user_action(_state("draft")) == "proposal"

    def test_build_routes_to_prototype(self) -> None:
        assert route_user_action(_state("build")) == "prototype"

    def test_skip_routes_to_done(self) -> None:
        assert route_user_action(_state("skip")) == "done"

    def test_archive_routes_to_done(self) -> None:
        assert route_user_action(_state("archive")) == "done"

    def test_none_routes_to_done(self) -> None:
        assert route_user_action(_state(None)) == "done"

    def test_both_routes_to_parallel_sends(self) -> None:
        result = route_user_action(_state("both"))
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, Send) for item in result)
        assert {send.node for send in result} == {"proposal", "prototype"}
