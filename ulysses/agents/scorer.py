"""Pure, deterministic job scoring engine — no LLM calls, no I/O.

Implements the weighted scoring formula from `ULYSSES-ARQUITECHTURE.md`:

    score = freshness(posted_at)      # 0-30
          + low_proposal_count(count) # 0-25
          + new_client(client_hires)  # 0-20
          + skill_match(skills)       # 0-15
          + budget_match(budget)      # 0-10

Each component function returns points on its own weighted scale (they sum to
a 100-point total by default), so `score_job` simply adds them. The point
values themselves live in `profile.scoring.weights` (see
`config.profile.ScoringWeights`), not as module constants here, so they can be
tuned via `ulysses config set` based on observed win rate instead of a code
change.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ulysses.config.profile import Profile, ScoringWeights
from ulysses.models import BudgetRange, GigCategory, JobPost, JobScore, Recommendation
from ulysses.tools.github_mapper import rank_matching_repos
from ulysses.tools.red_flag import detect_red_flags

__all__ = ["score_job"]


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
    weights = profile.scoring.weights

    freshness_score = _score_freshness(job.posted_at, now, weights)
    proposal_score = _score_proposal_count(job.proposals_count, weights)
    client_score = _score_client_history(job.client_hires, weights)
    skill_score = _score_skill_match(job.skills_required, profile.skills.all, weights)
    budget_score = _score_budget_match(
        job.budget, profile.scoring.target_budget_min, profile.scoring.target_budget_max, weights
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


def _score_freshness(posted_at: datetime, now: datetime, weights: ScoringWeights) -> float:
    age_minutes = (now - posted_at).total_seconds() / 60
    if age_minutes < 15:
        return weights.freshness_under_15_min
    if age_minutes < 60:
        return weights.freshness_under_1_hour
    return weights.freshness_stale


def _score_proposal_count(proposals_count: int | None, weights: ScoringWeights) -> float:
    if proposals_count is None:
        return weights.proposals_unknown  # No proposal count visible; assume the middle tier.
    if proposals_count < 5:
        return weights.proposals_under_5
    if proposals_count <= 15:
        return weights.proposals_5_to_15
    return weights.proposals_over_15


def _score_client_history(client_hires: int, weights: ScoringWeights) -> float:
    if client_hires == 0:
        return weights.client_no_hires
    if client_hires <= 3:
        return weights.client_1_to_3_hires
    return weights.client_over_3_hires


def _score_skill_match(
    skills_required: list[str], profile_skills: list[str], weights: ScoringWeights
) -> float:
    if not skills_required:
        return 0.0
    required = {skill.strip().lower() for skill in skills_required}
    known = {skill.strip().lower() for skill in profile_skills}
    if not required:
        return 0.0
    overlap_fraction = len(required & known) / len(required)
    return round(overlap_fraction * weights.skill_match_max_points, 2)


def _score_budget_match(
    budget: BudgetRange, target_min: float, target_max: float, weights: ScoringWeights
) -> float:
    midpoint = budget.midpoint
    max_points = weights.budget_match_max_points
    if midpoint is None:
        return weights.budget_unknown_points  # Neutral score when no budget is listed at all.
    if target_min <= midpoint <= target_max:
        return max_points
    if midpoint < target_min:
        scaled = max_points * (midpoint / target_min)
    else:
        scaled = max_points * (target_max / midpoint)
    return round(max(0.0, min(max_points, scaled)), 2)


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
