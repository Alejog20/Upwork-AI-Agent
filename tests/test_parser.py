"""Tests for `ulysses.tools.job_parser` against representative Upwork email HTML."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ulysses.models import BudgetType
from ulysses.tools.job_parser import JobParseError, parse_job_email

# Mirrors the real "New job alert: ..." template (confirmed against live
# emails, 2026-10): label-first budget ("Fixed-price • $200.00"), skill
# pills linking to a skill-search page rather than a text heading, the
# description truncated by an inline "more" link with a literal "Job
# Description:" marker glued onto the repeated title, and no relative
# "posted N ago" phrase anywhere -- `posted_at` must come from `received_at`.
CURRENT_TEMPLATE_HTML = """
<html><body>
<table><tr><td>
<a href="https://www.upwork.com/jobs/~022082515063060880349?link=title&utm_source=mailgun">
Python Developer Needed for Web Scraping</a>
</td></tr></table>
<table><tr><td>
<div>Fixed-price • $200.00</div>
</td></tr></table>
<table><tr><td>
<div>Python Developer Needed for Web ScrapingJob Description:I am looking for an experienced
Python developer to build a scraper for real estate listings.
<a href="https://www.upwork.com/jobs/~022082515063060880349?link=more">more</a></div>
</td></tr></table>
<table><tr><td>
<div>
<a href="https://www.upwork.com/nx/search/jobs/?frkscc=abc">Python</a>
<a href="https://www.upwork.com/nx/search/jobs/?frkscc=def">Web Scraping</a>
<a href="https://www.upwork.com/nx/search/jobs/?frkscc=ghi">BeautifulSoup</a>
</div>
</td></tr></table>
<div>Payment verified • 4.39 • $770 spent • United States</div>
</body></html>
"""

CURRENT_TEMPLATE_HOURLY_HTML = """
<html><body>
<a href="https://www.upwork.com/jobs/~022082515063060880350?link=title">
Ongoing bookkeeping support</a>
<div>Hourly • $20.00 - $35.00</div>
<div>Payment verified • 4.90 • $12,000 spent • Canada</div>
</body></html>
"""

FULL_EMAIL_HTML = """
<html><body>
<a href="https://www.upwork.com/jobs/~0112345678901234">Python scraper for real estate listings</a>
<p>We need someone to build a scraper that pulls real estate listings daily and exports to CSV.</p>
<div>Budget: $100 - $200</div>
<div>Skills: <ul><li>Python</li><li>Web Scraping</li><li>BeautifulSoup</li></ul></div>
<div>Client: 3 hires, 4.8 of 5 stars, Payment method verified</div>
<div>Less than 5 proposals</div>
<div>Posted 8 minutes ago</div>
</body></html>
"""

MISSING_BUDGET_HTML = """
<html><body>
<a href="https://www.upwork.com/jobs/~0198765432109876">Automate invoice generation</a>
<p>We need a script that turns our spreadsheet of orders into PDF invoices automatically.</p>
<div>Skills: <ul><li>Python</li><li>Pandas</li></ul></div>
<div>Posted 2 hours ago</div>
</body></html>
"""

MISSING_SKILLS_HTML = """
<html><body>
<a href="https://www.upwork.com/jobs/~0100000000000001">Small automation task</a>
<p>Need a quick script to rename files in a folder based on their contents.</p>
<div>Budget: $75 fixed price</div>
<div>Posted 30 minutes ago</div>
</body></html>
"""

NO_CLIENT_HISTORY_HTML = """
<html><body>
<a href="https://www.upwork.com/jobs/~0100000000000002">First-time client's automation project</a>
<p>Brand new to Upwork, need help automating a recurring reporting workflow.</p>
<div>Budget: $50/hr</div>
<div>Posted 1 minute ago</div>
</body></html>
"""

NO_JOB_LINK_HTML = """
<html><body>
<p>This is a promotional email with no job posting in it at all.</p>
</body></html>
"""

PROPOSAL_RANGE_HTML = """
<html><body>
<a href="https://www.upwork.com/jobs/~0100000000000003">Data pipeline job</a>
<p>Need an ETL pipeline built to move data between two systems reliably.</p>
<div>12 to 15 proposals</div>
<div>Posted 3 minutes ago</div>
</body></html>
"""

PROPOSAL_EXACT_HTML = """
<html><body>
<a href="https://www.upwork.com/jobs/~0100000000000004">API integration job</a>
<p>Integrate our backend with a third-party payment API and handle webhooks.</p>
<div>7 proposals</div>
<div>Posted 20 minutes ago</div>
</body></html>
"""


class TestFullyPopulatedEmail:
    def test_parses_without_error(self) -> None:
        job, error = parse_job_email(FULL_EMAIL_HTML)
        assert error is None
        assert job is not None

    def test_extracts_title_and_url(self) -> None:
        job, _ = parse_job_email(FULL_EMAIL_HTML)
        assert job.title == "Python scraper for real estate listings"
        assert job.url == "https://www.upwork.com/jobs/~0112345678901234"

    def test_extracts_job_id_from_url(self) -> None:
        job, _ = parse_job_email(FULL_EMAIL_HTML)
        assert job.id == "0112345678901234"

    def test_extracts_description(self) -> None:
        job, _ = parse_job_email(FULL_EMAIL_HTML)
        assert "scraper" in job.description.lower()

    def test_extracts_budget_range(self) -> None:
        job, _ = parse_job_email(FULL_EMAIL_HTML)
        assert job.budget.type == BudgetType.FIXED
        assert job.budget.min_amount == 100.0
        assert job.budget.max_amount == 200.0

    def test_extracts_skills(self) -> None:
        job, _ = parse_job_email(FULL_EMAIL_HTML)
        lowered_skills = {skill.lower() for skill in job.skills_required}
        assert {"python", "web scraping", "beautifulsoup"} <= lowered_skills

    def test_extracts_client_metadata(self) -> None:
        job, _ = parse_job_email(FULL_EMAIL_HTML)
        assert job.client_hires == 3
        assert job.client_rating == 4.8
        assert job.payment_verified is True

    def test_extracts_proposals_count(self) -> None:
        job, _ = parse_job_email(FULL_EMAIL_HTML)
        assert job.proposals_count == 5

    def test_extracts_recent_posted_at(self) -> None:
        job, _ = parse_job_email(FULL_EMAIL_HTML)
        age_seconds = (datetime.now(UTC) - job.posted_at).total_seconds()
        assert 7 * 60 <= age_seconds <= 9 * 60


class TestMissingBudget:
    def test_defaults_to_unknown_budget(self) -> None:
        job, error = parse_job_email(MISSING_BUDGET_HTML)
        assert error is None
        assert job.budget.type == BudgetType.UNKNOWN
        assert job.budget.midpoint is None


class TestMissingSkills:
    def test_defaults_to_empty_skills_list(self) -> None:
        job, error = parse_job_email(MISSING_SKILLS_HTML)
        assert error is None
        assert job.skills_required == []
        assert job.budget.type == BudgetType.FIXED
        assert job.budget.min_amount == 75.0


class TestNoClientHistory:
    def test_defaults_to_zero_hires_and_no_rating(self) -> None:
        job, error = parse_job_email(NO_CLIENT_HISTORY_HTML)
        assert error is None
        assert job.client_hires == 0
        assert job.client_rating is None
        assert job.payment_verified is False

    def test_hourly_budget_parsed(self) -> None:
        job, _ = parse_job_email(NO_CLIENT_HISTORY_HTML)
        assert job.budget.type == BudgetType.HOURLY
        assert job.budget.min_amount == 50.0


class TestMissingJobLink:
    def test_returns_error_result(self) -> None:
        job, error = parse_job_email(NO_JOB_LINK_HTML)
        assert job is None
        assert isinstance(error, JobParseError)


class TestProposalsCountVariants:
    def test_range_pattern_takes_upper_bound(self) -> None:
        job, _ = parse_job_email(PROPOSAL_RANGE_HTML)
        assert job.proposals_count == 15

    def test_exact_count_pattern(self) -> None:
        job, _ = parse_job_email(PROPOSAL_EXACT_HTML)
        assert job.proposals_count == 7


class TestJobIdFallback:
    def test_falls_back_to_hash_when_no_tilde_id(self) -> None:
        html = """
        <html><body>
        <a href="https://www.upwork.com/jobs/no-id-in-this-url">Odd URL job</a>
        <p>Description long enough to be picked up by the parser heuristics here.</p>
        </body></html>
        """
        job, error = parse_job_email(html)
        assert error is None
        assert len(job.id) == 16

    def test_same_url_produces_same_fallback_id(self) -> None:
        html = """
        <html><body>
        <a href="https://www.upwork.com/jobs/no-id-in-this-url">Odd URL job</a>
        <p>Description long enough to be picked up by the parser heuristics here.</p>
        </body></html>
        """
        first, _ = parse_job_email(html)
        second, _ = parse_job_email(html)
        assert first.id == second.id


class TestCurrentTemplate:
    """The real "New job alert: ..." template, confirmed against live emails."""

    def test_parses_without_error(self) -> None:
        job, error = parse_job_email(CURRENT_TEMPLATE_HTML)
        assert error is None
        assert job is not None

    def test_strips_tracking_params_from_the_url(self) -> None:
        job, _ = parse_job_email(CURRENT_TEMPLATE_HTML)
        assert job.url == "https://www.upwork.com/jobs/~022082515063060880349"
        assert "utm_source" not in job.url
        assert "&" not in job.url

    def test_extracts_the_real_description_not_the_boilerplate_preamble(self) -> None:
        job, _ = parse_job_email(CURRENT_TEMPLATE_HTML)
        assert "experienced" in job.description.lower()
        assert "job description" not in job.description.lower()
        assert "new job alert" not in job.description.lower()

    def test_extracts_label_first_fixed_budget(self) -> None:
        job, _ = parse_job_email(CURRENT_TEMPLATE_HTML)
        assert job.budget.type == BudgetType.FIXED
        assert job.budget.min_amount == 200.0
        assert job.budget.max_amount == 200.0

    def test_extracts_skill_pills_not_search_link_noise(self) -> None:
        job, _ = parse_job_email(CURRENT_TEMPLATE_HTML)
        assert job.skills_required == ["Python", "Web Scraping", "BeautifulSoup"]

    def test_excludes_the_overflow_count_pill(self) -> None:
        html = CURRENT_TEMPLATE_HTML.replace(
            '<a href="https://www.upwork.com/nx/search/jobs/?frkscc=ghi">BeautifulSoup</a>',
            '<a href="https://www.upwork.com/nx/search/jobs/?frkscc=ghi">BeautifulSoup</a>'
            '\n<a href="https://www.upwork.com/nx/search/jobs/?frkscc=xyz">+2</a>',
        )
        job, _ = parse_job_email(html)
        assert job.skills_required == ["Python", "Web Scraping", "BeautifulSoup"]

    def test_payment_verified_without_the_word_method(self) -> None:
        job, _ = parse_job_email(CURRENT_TEMPLATE_HTML)
        assert job.payment_verified is True

    def test_posted_at_falls_back_to_received_at_when_no_relative_phrase(self) -> None:
        received_at = datetime(2026, 7, 29, 17, 16, 21, tzinfo=UTC)
        job, _ = parse_job_email(CURRENT_TEMPLATE_HTML, received_at=received_at)
        assert job.posted_at == received_at

    def test_posted_at_falls_back_to_now_when_received_at_is_not_given(self) -> None:
        job, _ = parse_job_email(CURRENT_TEMPLATE_HTML)
        age_seconds = (datetime.now(UTC) - job.posted_at).total_seconds()
        assert 0 <= age_seconds <= 5

    def test_explicit_posted_ago_phrase_is_anchored_to_received_at_not_now(self) -> None:
        # The "posted N ago" phrase, when present, was itself written
        # relative to when the email was sent -- processing it days later
        # must not silently re-anchor it to the processing moment.
        html = CURRENT_TEMPLATE_HTML.replace(
            "<div>Fixed-price • $200.00</div>",
            "<div>Fixed-price • $200.00 Posted 8 minutes ago</div>",
        )
        received_at = datetime.now(UTC) - timedelta(days=10)
        job, _ = parse_job_email(html, received_at=received_at)
        assert abs((job.posted_at - (received_at - timedelta(minutes=8))).total_seconds()) < 1

    def test_hourly_label_first_budget(self) -> None:
        job, _ = parse_job_email(CURRENT_TEMPLATE_HOURLY_HTML)
        assert job.budget.type == BudgetType.HOURLY
        assert job.budget.min_amount == 20.0
        assert job.budget.max_amount == 35.0
