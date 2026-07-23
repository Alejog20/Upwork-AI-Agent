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
from ulysses.agents.proposal import ProposalAgent
from ulysses.agents.prototype import PrototypeAgent, build_prototype_zip
from ulysses.agents.scorer import score_job
from ulysses.config.profile import Profile
from ulysses.graph.state import UlyssesState
from ulysses.tools.db import UlyssesDB

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


def build_proposal_node(
    proposal_agent: ProposalAgent, notifier: NotifierAgent, db: UlyssesDB, profile: Profile
) -> GraphNode:
    """Build the `proposal` node: generates a draft, persists it, and sends it to Telegram."""

    async def proposal_node(state: UlyssesState) -> UlyssesState:
        score = state["score"]
        if score is None:
            raise ValueError("proposal_node requires state['score'] to be set by scorer_node first")
        draft = await proposal_agent.generate(state["job"], score, profile)
        await db.add_proposal_draft(state["job"].id, draft.full_text)
        await notifier.send_proposal_draft(state["job"].id, draft.full_text)
        logger.bind(job_id=state["job"].id, agent="proposal").info("Draft generated and sent")
        return {**state, "proposal_draft": draft.full_text}

    return proposal_node


def build_prototype_node(
    prototype_agent: PrototypeAgent, notifier: NotifierAgent, db: UlyssesDB, profile: Profile
) -> GraphNode:
    """Build the `prototype` node: generates a demo, persists it, and sends it to Telegram."""

    async def prototype_node(state: UlyssesState) -> UlyssesState:
        score = state["score"]
        if score is None:
            raise ValueError(
                "prototype_node requires state['score'] to be set by scorer_node first"
            )
        prototype = await prototype_agent.generate(state["job"], score, profile)
        files = {
            "demo.py": prototype.demo_script,
            "requirements.txt": prototype.requirements_txt,
            "README.md": prototype.readme_md,
            "config.example.env": prototype.config_example_env,
        }
        for filename, content in files.items():
            await db.add_prototype_file(state["job"].id, filename, content)
        zip_bytes = build_prototype_zip(prototype)
        await notifier.send_prototype_zip(state["job"].id, prototype, zip_bytes)
        logger.bind(job_id=state["job"].id, agent="prototype").info("Prototype generated and sent")
        return {**state, "prototype_files": files}

    return prototype_node


def build_done_node() -> GraphNode:
    """Build the terminal `done` node."""

    async def done_node(state: UlyssesState) -> UlyssesState:
        logger.bind(job_id=state["job"].id, agent="graph").debug("Job pipeline complete")
        return {**state, "completed": True}

    return done_node
