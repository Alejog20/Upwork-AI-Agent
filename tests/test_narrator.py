"""Tests for `ulysses.agents.narrator`: the Narrator Agent."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from ulysses.agents.narrator import (
    NarratorAgent,
    _format_client_history,
    _format_posted_age,
    _format_skill_overlap,
)
from ulysses.agents.scorer import score_job
from ulysses.config.profile import Profile
from ulysses.models import JobPost


class TestFormatPostedAge:
    def test_minutes_ago(self) -> None:
        posted_at = datetime.now(UTC) - timedelta(minutes=9)
        assert _format_posted_age(posted_at) == "9 minutes ago"

    def test_hours_ago(self) -> None:
        posted_at = datetime.now(UTC) - timedelta(hours=5)
        assert _format_posted_age(posted_at) == "5 hours ago"

    def test_days_ago(self) -> None:
        posted_at = datetime.now(UTC) - timedelta(days=3)
        assert _format_posted_age(posted_at) == "3 days ago"

    def test_future_timestamp_does_not_go_negative(self) -> None:
        posted_at = datetime.now(UTC) + timedelta(minutes=5)
        assert _format_posted_age(posted_at) == "0 minutes ago"


class TestFormatClientHistory:
    def test_zero_hires_is_brand_new(self) -> None:
        assert _format_client_history(0) == "brand new client, 0 previous hires"

    def test_one_hire_is_singular(self) -> None:
        assert _format_client_history(1) == "1 previous hire"

    def test_multiple_hires_is_plural(self) -> None:
        assert _format_client_history(5) == "5 previous hires"


class TestFormatSkillOverlap:
    def test_no_skills_listed(self) -> None:
        assert _format_skill_overlap([], ["python"]) == "no specific skills listed"

    def test_no_overlap(self) -> None:
        result = _format_skill_overlap(["cobol"], ["python", "fastapi"])
        assert result == "none of the required skills match: cobol"

    def test_partial_overlap_lists_matched_skills(self) -> None:
        result = _format_skill_overlap(
            ["python", "cobol", "web scraping"], ["python", "web scraping"]
        )
        assert result == "2 of 3 required skills match: python, web scraping"

    def test_matching_is_case_and_whitespace_insensitive(self) -> None:
        result = _format_skill_overlap([" Python "], ["python"])
        assert result == "1 of 1 required skills match: python"


class TestNarratorAgentNarrate:
    def _mock_llm(self, blurb: str) -> MagicMock:
        structured_output = MagicMock()
        structured_output.blurb = blurb

        structured_llm = AsyncMock()
        structured_llm.ainvoke = AsyncMock(return_value=structured_output)

        llm = MagicMock()
        llm.bind = MagicMock(return_value=llm)
        llm.with_structured_output = MagicMock(return_value=structured_llm)
        return llm

    async def test_returns_the_llm_generated_blurb(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        llm = self._mock_llm("  New client, fresh posting. I'd apply now.  ")
        score = score_job(fresh_job, profile)
        agent = NarratorAgent(llm=llm)

        blurb = await agent.narrate(fresh_job, score, profile)

        assert blurb == "New client, fresh posting. I'd apply now."

    async def test_prompt_includes_grounded_job_facts(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        llm = self._mock_llm("A blurb.")
        score = score_job(fresh_job, profile)
        agent = NarratorAgent(llm=llm)

        await agent.narrate(fresh_job, score, profile)

        structured_llm = llm.with_structured_output.return_value
        prompt = structured_llm.ainvoke.await_args.args[0]
        user_content = prompt[1]["content"]
        assert fresh_job.title in user_content
        assert score.recommendation.value in user_content
        assert f"{score.total_score:.0f}" in user_content

    async def test_no_red_flags_renders_as_none(self, fresh_job: JobPost, profile: Profile) -> None:
        llm = self._mock_llm("A blurb.")
        score = score_job(fresh_job, profile)
        assert score.red_flags == []
        agent = NarratorAgent(llm=llm)

        await agent.narrate(fresh_job, score, profile)

        structured_llm = llm.with_structured_output.return_value
        prompt = structured_llm.ainvoke.await_args.args[0]
        assert "Red flags: none" in prompt[1]["content"]
