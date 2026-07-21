"""Conditional routing logic for the Hermes LangGraph pipeline."""

from __future__ import annotations

from langgraph.types import Send

from hermes.graph.state import HermesState

__all__ = ["ROUTE_MAP", "route_user_action"]

ROUTE_MAP: dict[str, str | list[str]] = {
    "draft": "proposal",
    "build": "prototype",
    "both": ["proposal", "prototype"],
    "skip": "done",
    "archive": "done",
}


def route_user_action(state: HermesState) -> str | list[Send]:
    """Route from `notifier` to `proposal`/`prototype`/`done` based on `state["user_action"]`.

    Returns a single node name for `draft`/`build`/`skip`/`archive`, or a list
    of `Send` dispatches for `both` so `proposal` and `prototype` run in
    parallel branches, per LangGraph's `Send` API.
    """
    action = state.get("user_action")
    if action == "both":
        return [Send("proposal", state), Send("prototype", state)]
    return ROUTE_MAP.get(action, "done")
