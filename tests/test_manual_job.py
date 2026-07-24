"""Tests for `ulysses.tools.manual_job`: LLM-based extraction from pasted job text."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from ulysses.models import BudgetType
from ulysses.tools.manual_job import ManualJobParseError, extract_job_from_text

_SAMPLE_LISTING = """Python Web Scraper Needed for Real Estate Listings
Posted 3 hours ago

We need a scraper that pulls real estate listings from three sites daily and
writes them to a spreadsheet. Must handle pagination and dedupe.

Skills: Python, BeautifulSoup, Web Scraping
Payment method verified
5 hires
4.9 of 5 stars
Less than 5 proposals
"""


@pytest.fixture
def mock_llm() -> MagicMock:
    structured_output = MagicMock()
    structured_output.title = "Python Web Scraper Needed for Real Estate Listings"
    structured_output.description = (
        "We need a scraper that pulls real estate listings from three sites daily."
    )
    structured_output.budget_type = BudgetType.FIXED
    structured_output.budget_min = 150.0
    structured_output.budget_max = 150.0
    structured_output.budget_currency = "USD"
    structured_output.skills_required = ["Python", "BeautifulSoup", "Web Scraping"]
    structured_output.client_hires = 5
    structured_output.client_rating = 4.9
    structured_output.payment_verified = True
    structured_output.proposals_count = 5
    structured_output.posted_minutes_ago = 180
    structured_output.url = None

    structured_llm = AsyncMock()
    structured_llm.ainvoke = AsyncMock(return_value=structured_output)

    llm = MagicMock()
    llm.bind = MagicMock(return_value=llm)
    llm.with_structured_output = MagicMock(return_value=structured_llm)
    return llm


class TestExtractJobFromText:
    async def test_successful_extraction_populates_all_fields(self, mock_llm: MagicMock) -> None:
        job = await extract_job_from_text(_SAMPLE_LISTING, llm=mock_llm)

        assert job.title == "Python Web Scraper Needed for Real Estate Listings"
        assert job.budget.type is BudgetType.FIXED
        assert job.budget.min_amount == 150.0
        assert job.skills_required == ["Python", "BeautifulSoup", "Web Scraping"]
        assert job.client_hires == 5
        assert job.client_rating == 4.9
        assert job.payment_verified is True
        assert job.proposals_count == 5

    async def test_synthesizes_manual_url_when_llm_finds_none(self, mock_llm: MagicMock) -> None:
        job = await extract_job_from_text(_SAMPLE_LISTING, llm=mock_llm)

        assert job.url.startswith("manual://")
        assert job.id

    async def test_uses_found_url_when_llm_extracts_one(self, mock_llm: MagicMock) -> None:
        mock_llm.with_structured_output.return_value.ainvoke.return_value.url = (
            "https://www.upwork.com/jobs/~0112345678901234"
        )

        job = await extract_job_from_text(_SAMPLE_LISTING, llm=mock_llm)

        assert job.url == "https://www.upwork.com/jobs/~0112345678901234"

    async def test_same_found_url_produces_the_same_id_across_calls(
        self, mock_llm: MagicMock
    ) -> None:
        mock_llm.with_structured_output.return_value.ainvoke.return_value.url = (
            "https://www.upwork.com/jobs/~0112345678901234"
        )

        job1 = await extract_job_from_text(_SAMPLE_LISTING, llm=mock_llm)
        job2 = await extract_job_from_text(_SAMPLE_LISTING, llm=mock_llm)

        assert job1.id == job2.id

    async def test_posted_at_computed_from_posted_minutes_ago(self, mock_llm: MagicMock) -> None:
        before = datetime.now(UTC)

        job = await extract_job_from_text(_SAMPLE_LISTING, llm=mock_llm)

        after = datetime.now(UTC)
        assert before - timedelta(minutes=181) <= job.posted_at <= after - timedelta(minutes=179)

    async def test_posted_at_defaults_to_now_when_no_time_found(self, mock_llm: MagicMock) -> None:
        mock_llm.with_structured_output.return_value.ainvoke.return_value.posted_minutes_ago = None
        before = datetime.now(UTC)

        job = await extract_job_from_text(_SAMPLE_LISTING, llm=mock_llm)

        after = datetime.now(UTC)
        assert before <= job.posted_at <= after

    async def test_blank_input_raises_without_calling_the_llm(self, mock_llm: MagicMock) -> None:
        with pytest.raises(ManualJobParseError):
            await extract_job_from_text("   \n  ", llm=mock_llm)
        mock_llm.with_structured_output.return_value.ainvoke.assert_not_awaited()

    async def test_degenerate_llm_output_raises_manual_job_parse_error(
        self, mock_llm: MagicMock
    ) -> None:
        mock_llm.with_structured_output.return_value.ainvoke.return_value.title = ""
        mock_llm.with_structured_output.return_value.ainvoke.return_value.description = "x"

        with pytest.raises(ManualJobParseError):
            await extract_job_from_text(_SAMPLE_LISTING, llm=mock_llm)
