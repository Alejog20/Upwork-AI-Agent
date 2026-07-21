"""Node functions for the Ulysses LangGraph pipeline.

Each `build_*_node` factory closes over its dependencies (profile, DB, agent
instances) and returns a plain async function of `UlyssesState -> UlyssesState`,
per LangGraph's node contract. This keeps dependency injection explicit
(constructor-style, via closures) instead of importing global singletons.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from loguru import logger

from ulysses.agents.notifier import NotifierAgent
from ulysses.agents.scorer import score_job
from ulysses.config.profile import Profile
from ulysses.graph.state import UlyssesState

__all__ = [
    "build_done_node",
    "build_notifier_node",
    "build_proposal_node",
    "build_prototype_node",
    "build_scorer_node",
    "build_scout_node",
]

GraphNode = Callable[[UlyssesState], Awaitable[UlyssesState]]


def build_scout_node() -> GraphNode:
    """Build the `scout` node.

    The actual mailbox polling and dedup happens in `agents.scout.ScoutAgent`,
    run by the CLI/orchestrator *before* the graph is invoked for a given job.
    This node just asserts the invariant that a job is already present in
    state by the time the graph runs.
    """

    async def scout_node(state: UlyssesState) -> UlyssesState:
        logger.bind(job_id=state["job"].id, agent="scout").debug("Job entering graph")
        return state

    return scout_node


def build_scorer_node(profile: Profile) -> GraphNode:
    """Build the `scorer` node, bound to a freelancer `Profile`."""

    async def scorer_node(state: UlyssesState) -> UlyssesState:
        score = score_job(state["job"], profile)
        logger.bind(job_id=state["job"].id, agent="scorer").info(
            "Scored {}/100 -> {}", score.total_score, score.recommendation.value
        )
        return {**state, "score": score}

    return scorer_node


def build_notifier_node(notifier: NotifierAgent, profile: Profile) -> GraphNode:
    """Build the `notifier` node. This is the graph's human-in-the-loop interrupt point."""

    async def notifier_node(state: UlyssesState) -> UlyssesState:
        score = state["score"]
        if score is None:
            raise ValueError("notifier_node requires state['score'] to be set by scorer_node first")
        await notifier.handle_scored_job(state["job"], score, profile.scoring)
        return {**state, "notification_sent": True}

    return notifier_node


def build_proposal_node() -> GraphNode:
    """Build the `proposal` node. Stubbed until Phase 2 implements the Proposal Agent."""

    async def proposal_node(state: UlyssesState) -> UlyssesState:
        logger.bind(job_id=state["job"].id, agent="proposal").warning(
            "Proposal Agent not implemented until Phase 2; skipping"
        )
        return {**state, "proposal_draft": None}

    return proposal_node


def build_prototype_node() -> GraphNode:
    """Build the `prototype` node. Stubbed until Phase 3 implements the Prototype Agent."""

    async def prototype_node(state: UlyssesState) -> UlyssesState:
        logger.bind(job_id=state["job"].id, agent="prototype").warning(
            "Prototype Agent not implemented until Phase 3; skipping"
        )
        return {**state, "prototype_files": None}

    return prototype_node


def build_done_node() -> GraphNode:
    """Build the terminal `done` node."""

    async def done_node(state: UlyssesState) -> UlyssesState:
        logger.bind(job_id=state["job"].id, agent="graph").debug("Job pipeline complete")
        return {**state, "completed": True}

    return done_node
