"""Assembles the Ulysses LangGraph `StateGraph`.

Human-in-the-loop: the graph interrupts after `notifier` so the CLI/bot layer
can wait for a Telegram button press (which sets `state["user_action"]`)
before resuming into `proposal`/`prototype`/`done`.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ulysses.agents.notifier import NotifierAgent
from ulysses.config.profile import Profile
from ulysses.graph.edges import route_user_action
from ulysses.graph.nodes import (
    build_done_node,
    build_notifier_node,
    build_proposal_node,
    build_prototype_node,
    build_scorer_node,
    build_scout_node,
)
from ulysses.graph.state import UlyssesState

__all__ = ["build_graph"]


def build_graph(
    profile: Profile,
    notifier: NotifierAgent,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the Ulysses pipeline graph.

    Args:
        profile: The freelancer's profile, injected into the scorer/notifier nodes.
        notifier: The Telegram Notifier Agent, injected into the notifier node.
        checkpointer: Persists state across the human-in-the-loop interrupt.
            Defaults to an in-memory saver, which does not survive a process
            restart while a job is awaiting a button press — swap in a
            durable checkpointer (e.g. SQLite-backed) if that matters.

    Returns:
        A compiled LangGraph graph that interrupts after `notifier`, awaiting
        `state["user_action"]` before it can be resumed.
    """
    graph = StateGraph(UlyssesState)

    graph.add_node("scout", build_scout_node())
    graph.add_node("scorer", build_scorer_node(profile))
    graph.add_node("notifier", build_notifier_node(notifier, profile))
    graph.add_node("proposal", build_proposal_node())
    graph.add_node("prototype", build_prototype_node())
    graph.add_node("done", build_done_node())

    graph.set_entry_point("scout")
    graph.add_edge("scout", "scorer")
    graph.add_edge("scorer", "notifier")
    graph.add_conditional_edges(
        "notifier",
        route_user_action,
        {"proposal": "proposal", "prototype": "prototype", "done": "done"},
    )
    graph.add_edge("proposal", "done")
    graph.add_edge("prototype", "done")
    graph.add_edge("done", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver(), interrupt_after=["notifier"])
