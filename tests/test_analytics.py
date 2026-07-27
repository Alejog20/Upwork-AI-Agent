"""Tests for `ulysses.tools.analytics`: pure win-rate/weight-suggestion functions."""

from __future__ import annotations

from datetime import UTC, datetime

from ulysses.models import GigCategory, JobScore, Recommendation
from ulysses.tools.analytics import (
    average_score_won_vs_lost,
    scoring_weight_suggestions,
    win_rate_by_category,
    win_rate_by_red_flags,
    win_rate_by_score_bucket,
)
from ulysses.tools.db import Job, Outcome


def _score(
    *,
    freshness: float = 30.0,
    proposal: float = 20.0,
    client: float = 15.0,
    skill: float = 10.0,
    budget: float = 5.0,
    red_flags: list[str] | None = None,
) -> JobScore:
    return JobScore(
        total_score=freshness + proposal + client + skill + budget,
        freshness_score=freshness,
        proposal_score=proposal,
        client_score=client,
        skill_score=skill,
        budget_score=budget,
        gig_category=GigCategory.TIER_1,
        red_flags=red_flags or [],
        recommendation=Recommendation.APPLY_NOW,
    )


def _job(job_id: str, *, score: float, category: str, score_json: str | None) -> Job:
    return Job(
        id=job_id,
        title=f"Job {job_id}",
        description="desc",
        url=f"https://www.upwork.com/jobs/~{job_id}",
        score=score,
        category=category,
        posted_at=datetime.now(UTC),
        score_json=score_json,
    )


def _pair(
    job_id: str,
    *,
    score: float,
    won: bool,
    category: str = "tier1",
    job_score: JobScore | None = None,
) -> tuple[Job, Outcome]:
    score_json = job_score.model_dump_json() if job_score is not None else None
    return _job(job_id, score=score, category=category, score_json=score_json), Outcome(
        job_id=job_id, won=won
    )


class TestWinRateByCategory:
    def test_computes_rate_per_category(self) -> None:
        pairs = [
            _pair("a", score=80, category="tier1", won=True),
            _pair("b", score=70, category="tier1", won=False),
            _pair("c", score=40, category="tier3", won=True),
        ]
        assert win_rate_by_category(pairs) == {"tier1": 0.5, "tier3": 1.0}

    def test_empty_input_returns_empty_dict(self) -> None:
        assert win_rate_by_category([]) == {}


class TestWinRateByScoreBucket:
    def test_buckets_by_25_point_ranges(self) -> None:
        pairs = [
            _pair("a", score=10, won=True),
            _pair("b", score=60, won=False),
            _pair("c", score=90, won=True),
        ]
        rates = win_rate_by_score_bucket(pairs)
        assert rates["0-25"] == 1.0
        assert rates["25-50"] == 0.0
        assert rates["50-75"] == 0.0
        assert rates["75-100"] == 1.0

    def test_boundary_score_of_100_falls_in_top_bucket(self) -> None:
        rates = win_rate_by_score_bucket([_pair("a", score=100.0, won=True)])
        assert rates["75-100"] == 1.0


class TestWinRateByRedFlags:
    def test_splits_by_presence_of_red_flags(self) -> None:
        pairs = [
            _pair("a", score=80, won=True, job_score=_score(red_flags=["simple task"])),
            _pair("b", score=70, won=False, job_score=_score(red_flags=["simple task"])),
            _pair("c", score=90, won=True, job_score=_score(red_flags=[])),
        ]
        assert win_rate_by_red_flags(pairs) == {"has_red_flags": 0.5, "no_red_flags": 1.0}

    def test_skips_jobs_without_recoverable_score_json(self) -> None:
        pairs = [_pair("a", score=80, won=True, job_score=None)]
        assert win_rate_by_red_flags(pairs) == {"has_red_flags": 0.0, "no_red_flags": 0.0}


class TestAverageScoreWonVsLost:
    def test_computes_averages_for_each_group(self) -> None:
        pairs = [
            _pair("a", score=80, won=True),
            _pair("b", score=90, won=True),
            _pair("c", score=40, won=False),
        ]
        assert average_score_won_vs_lost(pairs) == {"won": 85.0, "lost": 40.0}

    def test_empty_group_averages_to_zero(self) -> None:
        assert average_score_won_vs_lost([_pair("a", score=80, won=True)]) == {
            "won": 80.0,
            "lost": 0.0,
        }


class TestScoringWeightSuggestions:
    def test_returns_insufficient_data_message_below_min_sample(self) -> None:
        pairs = [
            _pair("a", score=80, won=True, job_score=_score()),
            _pair("b", score=40, won=False, job_score=_score()),
        ]
        suggestions = scoring_weight_suggestions(pairs, min_sample_size=5)
        assert len(suggestions) == 1
        assert "Not enough data" in suggestions[0]

    def test_flags_a_component_with_a_wide_won_vs_lost_gap(self) -> None:
        won_pairs = [
            _pair(f"won-{i}", score=80, won=True, job_score=_score(freshness=30.0))
            for i in range(5)
        ]
        lost_pairs = [
            _pair(f"lost-{i}", score=40, won=False, job_score=_score(freshness=5.0))
            for i in range(5)
        ]
        suggestions = scoring_weight_suggestions(won_pairs + lost_pairs, min_sample_size=5)
        assert len(suggestions) == 1
        assert "freshness" in suggestions[0]
        assert "weighting it up" in suggestions[0]

    def test_flags_a_component_with_a_wide_gap_in_the_other_direction(self) -> None:
        won_pairs = [
            _pair(f"won-{i}", score=80, won=True, job_score=_score(budget=2.0)) for i in range(5)
        ]
        lost_pairs = [
            _pair(f"lost-{i}", score=40, won=False, job_score=_score(budget=10.0)) for i in range(5)
        ]
        suggestions = scoring_weight_suggestions(won_pairs + lost_pairs, min_sample_size=5)
        assert len(suggestions) == 1
        assert "budget match" in suggestions[0]
        assert "weighting it down" in suggestions[0]

    def test_skips_pairs_without_recoverable_score_json(self) -> None:
        won_pairs = [_pair(f"won-{i}", score=80, won=True, job_score=_score()) for i in range(5)]
        lost_pairs = [_pair(f"lost-{i}", score=40, won=False, job_score=_score()) for i in range(5)]
        unscored = [_pair("unscored", score=50, won=True, job_score=None)]

        suggestions = scoring_weight_suggestions(
            won_pairs + lost_pairs + unscored, min_sample_size=5
        )

        assert suggestions == [
            "No component shows a clear win/loss gap yet -- current weights look reasonable."
        ]

    def test_reports_no_clear_gap_when_components_are_similar(self) -> None:
        won_pairs = [_pair(f"won-{i}", score=80, won=True, job_score=_score()) for i in range(5)]
        lost_pairs = [_pair(f"lost-{i}", score=79, won=False, job_score=_score()) for i in range(5)]
        suggestions = scoring_weight_suggestions(won_pairs + lost_pairs, min_sample_size=5)
        assert suggestions == [
            "No component shows a clear win/loss gap yet -- current weights look reasonable."
        ]
