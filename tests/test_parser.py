"""Tests for `hermes.tools.job_parser` against representative Upwork email HTML."""

from __future__ import annotations

from datetime import UTC, datetime

from hermes.models import BudgetType
from hermes.tools.job_parser import JobParseError, parse_job_email

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
