"""Keyword-pattern detector for common Upwork scam and bait-and-switch red flags."""

from __future__ import annotations

__all__ = ["RED_FLAGS", "detect_red_flags"]

RED_FLAGS: tuple[str, ...] = (
    "will pay after results",
    "prove yourself",
    "simple task",
    "previous freelancer disappeared",
    "long-term if this works out",
    "per hour once hired",
)


def detect_red_flags(description: str) -> list[str]:
    """Return every red-flag phrase found in a job description.

    Args:
        description: The raw job description text.

    Returns:
        The subset of `RED_FLAGS` phrases present in `description`, in the
        order they're defined. Case-insensitive substring match.
    """
    lowered = description.lower()
    return [flag for flag in RED_FLAGS if flag in lowered]
