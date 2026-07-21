"""Shared pytest fixtures for the Ulysses test suite."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ulysses.config.profile import (
    AlertsConfig,
    FreelancerProfile,
    Profile,
    RepoConfig,
    ScoringConfig,
    SkillsConfig,
)
from ulysses.models import BudgetRange, BudgetType, JobPost


@pytest.fixture
def now() -> datetime:
    """A fixed clock so freshness-dependent tests are deterministic."""
    return datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def profile() -> Profile:
    """A representative freelancer profile for scoring tests."""
    return Profile(
        freelancer=FreelancerProfile(
            name="Alejandro Garcia",
            title="I automate the boring stuff",
            github="https://github.com/Alejog20",
            rate_usd_hr=25,
        ),
        skills=SkillsConfig(
            primary=["python", "web scraping", "automation", "beautifulsoup"],
            secondary=["langchain", "sqlite"],
        ),
        repos=[
            RepoConfig(
                name="Multiple_source_scraper",
                url="https://github.com/Alejog20/Multiple_source_scraper",
                tags=["scraping", "python", "data extraction", "multi-platform"],
            ),
            RepoConfig(
                name="download-autoprocessor",
                url="https://github.com/Alejog20/download-autoprocessor",
                tags=["automation", "file processing", "watchdog", "python"],
            ),
        ],
        scoring=ScoringConfig(
            target_budget_min=50,
            target_budget_max=800,
            min_score_to_notify=50,
            instant_alert_threshold=75,
            skip_if_proposals_above=25,
            skip_if_posted_hours_ago=6,
        ),
        alerts=AlertsConfig(telegram_chat_id="123456", batch_interval_minutes=30),
    )


@pytest.fixture
def fresh_job(now: datetime) -> JobPost:
    """A strong candidate job: fresh, few proposals, new client, skills match, in-budget."""
    return JobPost(
        id="abc123",
        title="Python scraper for real estate listings",
        description="We need a scraper for listings. Straightforward scope.",
        budget=BudgetRange(type=BudgetType.FIXED, min_amount=150, max_amount=150),
        skills_required=["python", "web scraping", "beautifulsoup"],
        client_hires=0,
        client_rating=None,
        payment_verified=True,
        proposals_count=3,
        posted_at=now - timedelta(minutes=5),
        url="https://www.upwork.com/jobs/~0112345678901234",
    )
