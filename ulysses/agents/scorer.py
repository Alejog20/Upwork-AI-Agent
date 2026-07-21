"""Pure, deterministic job scoring engine — no LLM calls, no I/O.

Implements the weighted scoring formula from `ULYSSES-ARQUITECHTURE.md`:

    score = freshness(posted_at)      # 0-30
          + low_proposal_count(count) # 0-25
          + new_client(client_hires)  # 0-20
          + skill_match(skills)       # 0-15
          + budget_match(budget)      # 0-10

Each component function already returns points on its own weighted scale
(they sum to a 100-point total), so `score_job` simply adds them.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ulysses.config.profile import Profile
from ulysses.models import BudgetRange, GigCategory, JobPost, JobScore, Recommendation
from ulysses.tools.github_mapper import rank_matching_repos
from ulysses.tools.red_flag import detect_red_flags

__all__ = ["score_job"]

_FRESHNESS_UNDER_15_MIN = 30.0
_FRESHNESS_UNDER_1_HOUR = 20.0
_FRESHNESS_STALE = 5.0

_PROPOSALS_UNDER_5 = 25.0
_PROPOSALS_5_TO_15 = 15.0
_PROPOSALS_OVER_15 = 5.0
_PROPOSALS_UNKNOWN = 15.0  # No proposal count is visible; assume the middle tier.

_CLIENT_NO_HIRES = 20.0
_CLIENT_1_TO_3_HIRES = 12.0
_CLIENT_OVER_3_HIRES = 5.0

_SKILL_MATCH_MAX_POINTS = 15.0
_BUDGET_MATCH_MAX_POINTS = 10.0
_BUDGET_UNKNOWN_POINTS = 5.0  # Neutral score when no budget is listed at all.


def score_job(job: JobPost, profile: Profile, *, now: datetime | None = None) -> JobScore:
    """Score a job against the freelancer's profile.

    Args:
        job: The structured job posting to score.
        profile: The freelancer's profile (skills, repos, scoring thresholds).
        now: Clock override for freshness scoring, primarily for tests.
            Defaults to the current UTC time.

    Returns:
        The full `JobScore` breakdown, including matched repos, red flags,
        gig category, and a recommended action.
    """
    now = now or datetime.now(UTC)

    freshness_score = _score_freshness(job.posted_at, now)
    proposal_score = _score_proposal_count(job.proposals_count)
    client_score = _score_client_history(job.client_hires)
    skill_score = _score_skill_match(job.skills_required, profile.skills.all)
    budget_score = _score_budget_match(
        job.budget, profile.scoring.target_budget_min, profile.scoring.target_budget_max
    )

    total_score = round(
        freshness_score + proposal_score + client_score + skill_score + budget_score, 2
    )

    red_flags = detect_red_flags(job.description)
    matched_repos = rank_matching_repos(job.skills_required, profile.repos)
    gig_category = _categorize(
        total_score, profile.scoring.min_score_to_notify, profile.scoring.instant_alert_threshold
    )
    recommendation = _recommend(
        total_score,
        red_flags,
        profile.scoring.min_score_to_notify,
        profile.scoring.instant_alert_threshold,
    )

    return JobScore(
        total_score=total_score,
        freshness_score=freshness_score,
        proposal_score=proposal_score,
        client_score=client_score,
        skill_score=skill_score,
        budget_score=budget_score,
        matched_repos=matched_repos,
        gig_category=gig_category,
        red_flags=red_flags,
        recommendation=recommendation,
    )


def _score_freshness(posted_at: datetime, now: datetime) -> float:
    age_minutes = (now - posted_at).total_seconds() / 60
    if age_minutes < 15:
        return _FRESHNESS_UNDER_15_MIN
    if age_minutes < 60:
        return _FRESHNESS_UNDER_1_HOUR
    return _FRESHNESS_STALE


def _score_proposal_count(proposals_count: int | None) -> float:
    if proposals_count is None:
        return _PROPOSALS_UNKNOWN
    if proposals_count < 5:
        return _PROPOSALS_UNDER_5
    if proposals_count <= 15:
        return _PROPOSALS_5_TO_15
    return _PROPOSALS_OVER_15


def _score_client_history(client_hires: int) -> float:
    if client_hires == 0:
        return _CLIENT_NO_HIRES
    if client_hires <= 3:
        return _CLIENT_1_TO_3_HIRES
    return _CLIENT_OVER_3_HIRES


def _score_skill_match(skills_required: list[str], profile_skills: list[str]) -> float:
    if not skills_required:
        return 0.0
    required = {skill.strip().lower() for skill in skills_required}
    known = {skill.strip().lower() for skill in profile_skills}
    if not required:
        return 0.0
    overlap_fraction = len(required & known) / len(required)
    return round(overlap_fraction * _SKILL_MATCH_MAX_POINTS, 2)


def _score_budget_match(budget: BudgetRange, target_min: float, target_max: float) -> float:
    midpoint = budget.midpoint
    if midpoint is None:
        return _BUDGET_UNKNOWN_POINTS
    if target_min <= midpoint <= target_max:
        return _BUDGET_MATCH_MAX_POINTS
    if midpoint < target_min:
        scaled = _BUDGET_MATCH_MAX_POINTS * (midpoint / target_min)
    else:
        scaled = _BUDGET_MATCH_MAX_POINTS * (target_max / midpoint)
    return round(max(0.0, min(_BUDGET_MATCH_MAX_POINTS, scaled)), 2)


def _categorize(
    total_score: float, min_score_to_notify: float, instant_alert_threshold: float
) -> GigCategory:
    if total_score >= instant_alert_threshold:
        return GigCategory.TIER_1
    if total_score >= min_score_to_notify:
        return GigCategory.TIER_2
    return GigCategory.TIER_3


def _recommend(
    total_score: float,
    red_flags: list[str],
    min_score_to_notify: float,
    instant_alert_threshold: float,
) -> Recommendation:
    if total_score < min_score_to_notify:
        return Recommendation.SKIP
    if total_score >= instant_alert_threshold and not red_flags:
        return Recommendation.APPLY_NOW
    return Recommendation.REVIEW
