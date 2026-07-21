"""Typed state shared across every node in the Hermes LangGraph pipeline."""

from __future__ import annotations

from typing import Literal, TypedDict

from hermes.models import JobPost, JobScore

__all__ = ["HermesState", "UserAction"]

type UserAction = Literal["draft", "build", "both", "skip", "archive"]


class HermesState(TypedDict):
    """State threaded through the scout -> scorer -> notifier -> {proposal,prototype} graph."""

    job: JobPost
    score: JobScore | None
    user_action: UserAction | None
    proposal_draft: str | None
    prototype_files: dict[str, str] | None
    notification_sent: bool
    completed: bool
