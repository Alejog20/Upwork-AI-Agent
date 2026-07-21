"""Unit tests for the pure scoring engine in `hermes.agents.scorer`."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from hermes.agents.scorer import score_job
from hermes.config.profile import Profile
from hermes.models import BudgetRange, BudgetType, GigCategory, JobPost, Recommendation


def _job(fresh_job: JobPost, **overrides: object) -> JobPost:
    return fresh_job.model_copy(update=overrides)


class TestFreshnessScoring:
    def test_under_15_minutes_scores_max(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(fresh_job, posted_at=now - timedelta(minutes=10))
        score = score_job(job, profile, now=now)
        assert score.freshness_score == 30.0

    def test_under_1_hour_scores_mid(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(fresh_job, posted_at=now - timedelta(minutes=45))
        score = score_job(job, profile, now=now)
        assert score.freshness_score == 20.0

    def test_over_1_hour_scores_low(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(fresh_job, posted_at=now - timedelta(hours=5))
        score = score_job(job, profile, now=now)
        assert score.freshness_score == 5.0


class TestProposalCountScoring:
    def test_under_5_scores_max(self, fresh_job: JobPost, profile: Profile, now: datetime) -> None:
        job = _job(fresh_job, proposals_count=2)
        assert score_job(job, profile, now=now).proposal_score == 25.0

    def test_5_to_15_scores_mid(self, fresh_job: JobPost, profile: Profile, now: datetime) -> None:
        job = _job(fresh_job, proposals_count=10)
        assert score_job(job, profile, now=now).proposal_score == 15.0

    def test_over_15_scores_low(self, fresh_job: JobPost, profile: Profile, now: datetime) -> None:
        job = _job(fresh_job, proposals_count=30)
        assert score_job(job, profile, now=now).proposal_score == 5.0

    def test_unknown_scores_mid(self, fresh_job: JobPost, profile: Profile, now: datetime) -> None:
        job = _job(fresh_job, proposals_count=None)
        assert score_job(job, profile, now=now).proposal_score == 15.0


class TestClientHistoryScoring:
    def test_zero_hires_scores_max(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(fresh_job, client_hires=0)
        assert score_job(job, profile, now=now).client_score == 20.0

    def test_1_to_3_hires_scores_mid(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(fresh_job, client_hires=2)
        assert score_job(job, profile, now=now).client_score == 12.0

    def test_over_3_hires_scores_low(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(fresh_job, client_hires=10)
        assert score_job(job, profile, now=now).client_score == 5.0


class TestSkillMatchScoring:
    def test_full_overlap_scores_max(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(fresh_job, skills_required=["python", "web scraping"])
        assert score_job(job, profile, now=now).skill_score == 15.0

    def test_partial_overlap_scores_proportionally(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(fresh_job, skills_required=["python", "react", "typescript", "rust"])
        score = score_job(job, profile, now=now)
        assert score.skill_score == pytest.approx(15.0 * (1 / 4))

    def test_no_overlap_scores_zero(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(fresh_job, skills_required=["cobol", "fortran"])
        assert score_job(job, profile, now=now).skill_score == 0.0

    def test_no_skills_listed_scores_zero(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(fresh_job, skills_required=[])
        assert score_job(job, profile, now=now).skill_score == 0.0


class TestBudgetMatchScoring:
    def test_within_target_range_scores_max(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(
            fresh_job, budget=BudgetRange(type=BudgetType.FIXED, min_amount=200, max_amount=200)
        )
        assert score_job(job, profile, now=now).budget_score == 10.0

    def test_below_range_scales_down(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(
            fresh_job, budget=BudgetRange(type=BudgetType.FIXED, min_amount=25, max_amount=25)
        )
        score = score_job(job, profile, now=now)
        assert score.budget_score == pytest.approx(10.0 * (25 / 50))

    def test_above_range_scales_down(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(
            fresh_job, budget=BudgetRange(type=BudgetType.FIXED, min_amount=1600, max_amount=1600)
        )
        score = score_job(job, profile, now=now)
        assert score.budget_score == pytest.approx(10.0 * (800 / 1600))

    def test_unknown_budget_scores_neutral(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(fresh_job, budget=BudgetRange())
        assert score_job(job, profile, now=now).budget_score == 5.0


class TestRedFlagsAndRecommendation:
    def test_no_red_flags_high_score_recommends_apply_now(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        score = score_job(fresh_job, profile, now=now)
        assert score.red_flags == []
        assert score.recommendation == Recommendation.APPLY_NOW

    def test_red_flag_downgrades_apply_now_to_review(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(fresh_job, description="Simple task, will pay after results are delivered.")
        score = score_job(job, profile, now=now)
        assert "simple task" in score.red_flags
        assert "will pay after results" in score.red_flags
        assert score.recommendation == Recommendation.REVIEW

    def test_low_score_recommends_skip(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(
            fresh_job,
            posted_at=now - timedelta(hours=5),
            proposals_count=50,
            client_hires=20,
            skills_required=["cobol"],
            budget=BudgetRange(),
        )
        score = score_job(job, profile, now=now)
        assert score.recommendation == Recommendation.SKIP
        assert score.gig_category == GigCategory.TIER_3


class TestGigCategoryAndMatchedRepos:
    def test_tier1_at_or_above_instant_alert_threshold(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        score = score_job(fresh_job, profile, now=now)
        assert score.total_score >= profile.scoring.instant_alert_threshold
        assert score.gig_category == GigCategory.TIER_1

    def test_matched_repos_are_ranked_by_relevance(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = _job(fresh_job, skills_required=["scraping", "python", "data extraction"])
        score = score_job(job, profile, now=now)
        assert score.matched_repos[0].repo_name == "Multiple_source_scraper"

    def test_total_score_is_sum_of_components(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        score = score_job(fresh_job, profile, now=now)
        expected = round(
            score.freshness_score
            + score.proposal_score
            + score.client_score
            + score.skill_score
            + score.budget_score,
            2,
        )
        assert score.total_score == expected
