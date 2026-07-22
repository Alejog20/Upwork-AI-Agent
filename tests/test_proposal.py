"""Tests for `ulysses.agents.proposal`: the Proposal Agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ulysses.agents.proposal import (
    ProposalAgent,
    classify_category,
    estimate_timeline_and_bid,
    validate_proposal_text,
)
from ulysses.agents.scorer import score_job
from ulysses.config.profile import Profile
from ulysses.models import BudgetRange, BudgetType, JobPost


def _job(fresh_job: JobPost, **overrides: object) -> JobPost:
    return fresh_job.model_copy(update=overrides)


class TestClassifyCategory:
    def test_scraping_keywords_classify_as_scraping(self, fresh_job: JobPost) -> None:
        job = _job(
            fresh_job,
            title="Need a web scraper",
            description="Build a scraper using BeautifulSoup and Playwright.",
            skills_required=["web scraping"],
        )
        assert classify_category(job) == "scraping"

    def test_api_keywords_classify_as_api_dev(self, fresh_job: JobPost) -> None:
        job = _job(
            fresh_job,
            title="REST API integration",
            description="Integrate our backend with a third-party API via webhook.",
            skills_required=["fastapi"],
        )
        assert classify_category(job) == "api_dev"

    def test_data_pipeline_keywords_classify_as_data_pipeline(self, fresh_job: JobPost) -> None:
        job = _job(
            fresh_job,
            title="ETL pipeline",
            description="Transform CSV and Excel data into a clean reporting pipeline with pandas.",
            skills_required=["pandas"],
        )
        assert classify_category(job) == "data_pipeline"

    def test_ai_keywords_classify_as_ai_integration(self, fresh_job: JobPost) -> None:
        job = _job(
            fresh_job,
            title="Build an AI chatbot",
            description="Use OpenAI and LangChain to build an LLM-powered agent.",
            skills_required=["langchain"],
        )
        assert classify_category(job) == "ai_integration"

    def test_no_clear_match_defaults_to_automation(self, fresh_job: JobPost) -> None:
        job = _job(
            fresh_job,
            title="Help with a task",
            description="Need general help with something.",
            skills_required=[],
        )
        assert classify_category(job) == "automation"


class TestEstimateTimelineAndBid:
    def test_fixed_budget_scales_timeline_and_quotes_midpoint(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        job = _job(
            fresh_job, budget=BudgetRange(type=BudgetType.FIXED, min_amount=300, max_amount=300)
        )
        timeline, bid = estimate_timeline_and_bid(job, profile)
        assert timeline == "4 days"
        assert bid == 300.0

    def test_fixed_budget_clamped_to_max_14_days(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        job = _job(
            fresh_job, budget=BudgetRange(type=BudgetType.FIXED, min_amount=5000, max_amount=5000)
        )
        timeline, _ = estimate_timeline_and_bid(job, profile)
        assert timeline == "14 days"

    def test_hourly_budget_is_ongoing(self, fresh_job: JobPost, profile: Profile) -> None:
        job = _job(
            fresh_job, budget=BudgetRange(type=BudgetType.HOURLY, min_amount=40, max_amount=40)
        )
        timeline, bid = estimate_timeline_and_bid(job, profile)
        assert timeline == "Ongoing (hourly)"
        assert bid == 40.0

    def test_unknown_budget_falls_back_to_rate_based_estimate(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        job = _job(fresh_job, budget=BudgetRange())
        timeline, bid = estimate_timeline_and_bid(job, profile)
        assert timeline == "3 days"
        assert bid == profile.freelancer.rate_usd_hr * 3 * 4


class TestValidateProposalText:
    def test_clean_text_has_no_violations(self) -> None:
        assert validate_proposal_text("A short, clean proposal with no issues.") == []

    def test_flags_forbidden_greeting(self) -> None:
        violations = validate_proposal_text("Hello, I'd love to help.")
        assert any("hello" in v for v in violations)

    def test_flags_i_am_interested(self) -> None:
        violations = validate_proposal_text("I am interested in your project.")
        assert any("i am interested" in v for v in violations)

    def test_flags_over_800_characters(self) -> None:
        text = "word " * 200  # 1000 characters
        violations = validate_proposal_text(text)
        assert any("800 characters" in v for v in violations)

    def test_allows_up_to_800_characters(self) -> None:
        text = "word " * 150  # 750 characters
        assert validate_proposal_text(text) == []

    def test_flags_more_than_two_emoji(self) -> None:
        violations = validate_proposal_text(
            "Great fit \U0001f680 for this \U0001f525 project \U00002705."
        )
        assert any("emoji" in v for v in violations)

    def test_allows_up_to_two_emoji(self) -> None:
        assert validate_proposal_text("Great fit \U0001f680 for this \U0001f525 project.") == []


class TestProposalAgentGenerate:
    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        structured_output = MagicMock()
        structured_output.hook = "Your listings are stale and it's costing you time."
        structured_output.plan_bullet_1 = "Set up a targeted scraper for the listing site."
        structured_output.plan_bullet_2 = "Validate and deduplicate with pandas."
        structured_output.plan_bullet_3 = "Schedule it via cron."

        structured_llm = AsyncMock()
        structured_llm.ainvoke = AsyncMock(return_value=structured_output)

        llm = MagicMock()
        llm.bind = MagicMock(return_value=llm)
        llm.with_structured_output = MagicMock(return_value=structured_llm)
        return llm

    async def test_generate_fills_template_with_llm_output(
        self, fresh_job: JobPost, profile: Profile, mock_llm: MagicMock
    ) -> None:
        score = score_job(fresh_job, profile)
        agent = ProposalAgent(llm=mock_llm)

        result = await agent.generate(fresh_job, score, profile)

        assert result.job_id == fresh_job.id
        assert "Your listings are stale" in result.full_text
        assert "Set up a targeted scraper" in result.full_text
        assert result.proof_repo in result.full_text
        assert result.proof_repo_url in result.full_text
        assert f"${result.bid_usd:.0f}" in result.full_text
        assert result.timeline in result.full_text

    async def test_generate_never_contains_forbidden_phrases(
        self, fresh_job: JobPost, profile: Profile, mock_llm: MagicMock
    ) -> None:
        score = score_job(fresh_job, profile)
        agent = ProposalAgent(llm=mock_llm)

        result = await agent.generate(fresh_job, score, profile)

        assert validate_proposal_text(result.full_text) == []

    async def test_generate_selects_template_by_classified_category(
        self, fresh_job: JobPost, profile: Profile, mock_llm: MagicMock
    ) -> None:
        job = _job(
            fresh_job,
            title="Need a web scraper",
            description="Build a scraper for real estate listings.",
            skills_required=["web scraping", "beautifulsoup"],
        )
        score = score_job(job, profile)
        agent = ProposalAgent(llm=mock_llm)

        result = await agent.generate(job, score, profile)

        assert result.category == "scraping"
        assert "validation, dedup" in result.full_text

    async def test_generate_enforces_char_budget_without_cutting_off_pricing(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        structured_output = MagicMock()
        structured_output.hook = "This pain point is exactly what I solve. " * 10
        structured_output.plan_bullet_1 = "A very long and detailed first step. " * 10
        structured_output.plan_bullet_2 = "A very long and detailed second step. " * 10
        structured_output.plan_bullet_3 = "A very long and detailed third step. " * 10

        structured_llm = AsyncMock()
        structured_llm.ainvoke = AsyncMock(return_value=structured_output)
        llm = MagicMock()
        llm.bind = MagicMock(return_value=llm)
        llm.with_structured_output = MagicMock(return_value=structured_llm)

        score = score_job(fresh_job, profile)
        agent = ProposalAgent(llm=llm)

        result = await agent.generate(fresh_job, score, profile)

        assert len(result.full_text) <= 800
        assert f"Timeline: {result.timeline}" in result.full_text
        assert f"${result.bid_usd:.0f}" in result.full_text
