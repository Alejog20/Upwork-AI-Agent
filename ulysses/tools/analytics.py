"""Pure, deterministic win-rate analytics over recorded job outcomes — no LLM calls, no I/O.

Every function here takes the `(Job, Outcome)` pairs already loaded by
`tools.db.UlyssesDB.list_jobs_with_outcomes` and returns plain data (dicts/lists),
so this module stays fast, free, and fully unit-testable, mirroring `agents.scorer`.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import mean

from ulysses.models import JobScore
from ulysses.tools.db import Job, Outcome

__all__ = [
    "average_score_won_vs_lost",
    "scoring_weight_suggestions",
    "win_rate_by_category",
    "win_rate_by_red_flags",
    "win_rate_by_score_bucket",
]

_SCORE_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 25.0, "0-25"),
    (25.0, 50.0, "25-50"),
    (50.0, 75.0, "50-75"),
    (75.0, 100.0, "75-100"),
)

_SCORE_COMPONENTS: dict[str, Callable[[JobScore], float]] = {
    "freshness": lambda score: score.freshness_score,
    "proposal count": lambda score: score.proposal_score,
    "client history": lambda score: score.client_score,
    "skill match": lambda score: score.skill_score,
    "budget match": lambda score: score.budget_score,
}

_MIN_COMPONENT_GAP = 2.0


def win_rate_by_category(pairs: list[tuple[Job, Outcome]]) -> dict[str, float]:
    """Return win rate (0-1) grouped by the job's tier category (tier1/tier2/tier3)."""
    categories = sorted({job.category for job, _ in pairs})
    return {category: _win_rate(_filter_by_category(pairs, category)) for category in categories}


def win_rate_by_score_bucket(pairs: list[tuple[Job, Outcome]]) -> dict[str, float]:
    """Return win rate (0-1) grouped by 25-point score buckets (e.g. "50-75")."""
    grouped: dict[str, list[tuple[Job, Outcome]]] = {label: [] for *_, label in _SCORE_BUCKETS}
    for job, outcome in pairs:
        grouped[_bucket_for_score(job.score)].append((job, outcome))
    return {label: _win_rate(items) for label, items in grouped.items()}


def win_rate_by_red_flags(pairs: list[tuple[Job, Outcome]]) -> dict[str, float]:
    """Return win rate (0-1) for jobs with vs. without any detected red flags.

    Jobs that predate the `score_json` column (so red flags can't be recovered) are
    skipped for this specific breakdown.
    """
    with_flags: list[tuple[Job, Outcome]] = []
    without_flags: list[tuple[Job, Outcome]] = []
    for job, outcome in pairs:
        score = _parse_score(job)
        if score is None:
            continue
        (with_flags if score.red_flags else without_flags).append((job, outcome))
    return {"has_red_flags": _win_rate(with_flags), "no_red_flags": _win_rate(without_flags)}


def average_score_won_vs_lost(pairs: list[tuple[Job, Outcome]]) -> dict[str, float]:
    """Return the average total score for won jobs vs. lost jobs."""
    won_scores = [job.score for job, outcome in pairs if outcome.won]
    lost_scores = [job.score for job, outcome in pairs if not outcome.won]
    return {
        "won": round(mean(won_scores), 2) if won_scores else 0.0,
        "lost": round(mean(lost_scores), 2) if lost_scores else 0.0,
    }


def scoring_weight_suggestions(
    pairs: list[tuple[Job, Outcome]], *, min_sample_size: int = 5
) -> list[str]:
    """Suggest which scoring components correlate most with wins, as data to review.

    Never applied automatically -- surfaced for Alejandro to act on or ignore, per
    AGENTS.md's "never override a score without logging why." Compares the average
    value of each score component (freshness/proposal/client/skill/budget) between
    won and lost jobs; a component with a notably wider gap is more predictive and
    worth weighting up, a component with almost no gap is worth weighting down.
    Returns a single explanatory string (not a weight change) if there aren't at
    least `min_sample_size` of both won and lost jobs with recoverable score
    breakdowns -- too few data points to say anything meaningful yet.
    """
    won_scores: list[JobScore] = []
    lost_scores: list[JobScore] = []
    for job, outcome in pairs:
        score = _parse_score(job)
        if score is None:
            continue
        (won_scores if outcome.won else lost_scores).append(score)

    if len(won_scores) < min_sample_size or len(lost_scores) < min_sample_size:
        return [
            f"Not enough data yet: need at least {min_sample_size} won and {min_sample_size} "
            "lost jobs with recoverable score breakdowns to suggest weight changes."
        ]

    suggestions: list[str] = []
    for name, extractor in _SCORE_COMPONENTS.items():
        won_avg = mean(extractor(score) for score in won_scores)
        lost_avg = mean(extractor(score) for score in lost_scores)
        gap = won_avg - lost_avg
        if gap > _MIN_COMPONENT_GAP:
            suggestions.append(
                f"'{name}' averages {won_avg:.1f} on won jobs vs {lost_avg:.1f} on lost jobs -- "
                "consider weighting it up."
            )
        elif gap < -_MIN_COMPONENT_GAP:
            suggestions.append(
                f"'{name}' averages {won_avg:.1f} on won jobs vs {lost_avg:.1f} on lost jobs -- "
                "higher values didn't help here; consider weighting it down."
            )
    if not suggestions:
        suggestions.append(
            "No component shows a clear win/loss gap yet -- current weights look reasonable."
        )
    return suggestions


def _win_rate(pairs: list[tuple[Job, Outcome]]) -> float:
    if not pairs:
        return 0.0
    wins = sum(1 for _, outcome in pairs if outcome.won)
    return round(wins / len(pairs), 3)


def _filter_by_category(
    pairs: list[tuple[Job, Outcome]], category: str
) -> list[tuple[Job, Outcome]]:
    return [(job, outcome) for job, outcome in pairs if job.category == category]


def _bucket_for_score(score: float) -> str:
    for low, high, label in _SCORE_BUCKETS:
        if low <= score < high:
            return label
    return _SCORE_BUCKETS[-1][2]  # scores >= 100 fall into the top bucket


def _parse_score(job: Job) -> JobScore | None:
    """Reconstruct a job's `JobScore`, or `None` if it predates the `score_json` column."""
    if job.score_json is None:
        return None
    return JobScore.model_validate_json(job.score_json)
