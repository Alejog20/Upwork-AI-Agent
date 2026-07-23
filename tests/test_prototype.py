"""Tests for `ulysses.agents.prototype`: the Prototype Agent."""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest

from ulysses.agents.prototype import (
    PrototypeAgent,
    build_prototype_zip,
    classify_prototype_category,
)
from ulysses.agents.scorer import score_job
from ulysses.config.profile import Profile
from ulysses.models import GeneratedPrototype, JobPost

_REQUIREMENT_LINE_RE = re.compile(r"^[A-Za-z0-9_.-]+==[A-Za-z0-9.]+$")


def _job(fresh_job: JobPost, **overrides: object) -> JobPost:
    return fresh_job.model_copy(update=overrides)


class TestClassifyPrototypeCategory:
    def test_scraping_keywords_classify_as_scraper(self, fresh_job: JobPost) -> None:
        job = _job(
            fresh_job,
            title="Need a web scraper",
            description="Build a scraper using BeautifulSoup and Playwright.",
            skills_required=["web scraping"],
        )
        assert classify_prototype_category(job) == "scraper"

    def test_email_keywords_classify_as_email_bot(self, fresh_job: JobPost) -> None:
        job = _job(
            fresh_job,
            title="Gmail inbox automation",
            description="Watch a Gmail inbox and process unread email notifications.",
            skills_required=["email"],
        )
        assert classify_prototype_category(job) == "email_bot"

    def test_api_keywords_classify_as_api(self, fresh_job: JobPost) -> None:
        job = _job(
            fresh_job,
            title="REST API integration",
            description="Integrate our backend with a third-party API via webhook.",
            skills_required=["fastapi"],
        )
        assert classify_prototype_category(job) == "api"

    def test_ai_keywords_classify_as_ai_pipeline(self, fresh_job: JobPost) -> None:
        job = _job(
            fresh_job,
            title="Build an AI chatbot",
            description="Use OpenAI and LangChain to build an LLM-powered agent.",
            skills_required=["langchain"],
        )
        assert classify_prototype_category(job) == "ai_pipeline"

    def test_no_clear_match_defaults_to_file_automation(self, fresh_job: JobPost) -> None:
        job = _job(
            fresh_job,
            title="Help with a task",
            description="Need general help with something.",
            skills_required=[],
        )
        assert classify_prototype_category(job) == "file_automation"


class TestBuildPrototypeZip:
    def test_zip_contains_all_four_files_with_correct_content(self) -> None:
        prototype = GeneratedPrototype(
            job_id="job-1",
            category="scraper",
            demo_script="print('hello')",
            requirements_txt="requests==2.32.3\n",
            readme_md="# Demo\n",
            config_example_env="# none needed\n",
            zip_filename="ulysses_demo_job-1.zip",
        )

        zip_bytes = build_prototype_zip(prototype)

        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            names = set(zf.namelist())
            assert names == {
                "ulysses_demo_job-1/demo.py",
                "ulysses_demo_job-1/requirements.txt",
                "ulysses_demo_job-1/README.md",
                "ulysses_demo_job-1/config.example.env",
            }
            assert zf.read("ulysses_demo_job-1/demo.py").decode() == "print('hello')"
            assert zf.read("ulysses_demo_job-1/README.md").decode() == "# Demo\n"


class TestPrototypeAgentGenerate:
    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        structured_output = MagicMock()
        structured_output.demo_script = "import requests\n\nprint(requests.get('https://x').text)"
        structured_output.what_it_demonstrates = "It fetches and prints a page."
        structured_output.extension_bullet_1 = "Add pagination support."
        structured_output.extension_bullet_2 = "Add data validation."
        structured_output.extension_bullet_3 = "Add retries and logging for production use."

        structured_llm = AsyncMock()
        structured_llm.ainvoke = AsyncMock(return_value=structured_output)

        llm = MagicMock()
        llm.bind = MagicMock(return_value=llm)
        llm.with_structured_output = MagicMock(return_value=structured_llm)
        return llm

    async def test_generate_returns_populated_prototype(
        self, fresh_job: JobPost, profile: Profile, mock_llm: MagicMock
    ) -> None:
        job = _job(
            fresh_job,
            title="Need a web scraper",
            description="Scrape a page for listings.",
            skills_required=["web scraping"],
        )
        score = score_job(job, profile)
        agent = PrototypeAgent(llm=mock_llm)

        result = await agent.generate(job, score, profile)

        assert result.job_id == job.id
        assert result.category == "scraper"
        assert "requests" in result.demo_script
        assert result.zip_filename == f"ulysses_demo_{job.id}.zip"

    async def test_requirements_txt_is_deterministic_per_category(
        self, fresh_job: JobPost, profile: Profile, mock_llm: MagicMock
    ) -> None:
        job = _job(fresh_job, title="Need a web scraper", skills_required=["web scraping"])
        score = score_job(job, profile)
        agent = PrototypeAgent(llm=mock_llm)

        result = await agent.generate(job, score, profile)

        lines = [line for line in result.requirements_txt.splitlines() if line.strip()]
        assert lines == ["requests==2.32.3", "beautifulsoup4==4.12.3"]
        for line in lines:
            assert _REQUIREMENT_LINE_RE.match(line), f"invalid pip requirement line: {line!r}"

    async def test_requirements_txt_for_stdlib_only_category_is_a_comment(
        self, fresh_job: JobPost, profile: Profile, mock_llm: MagicMock
    ) -> None:
        job = _job(
            fresh_job,
            title="Organize files in a folder",
            description="Rename and move files into dated subfolders.",
            skills_required=[],
        )
        score = score_job(job, profile)
        agent = PrototypeAgent(llm=mock_llm)

        result = await agent.generate(job, score, profile)

        assert result.category == "file_automation"
        assert result.requirements_txt.strip().startswith("#")

    async def test_readme_has_all_required_sections(
        self, fresh_job: JobPost, profile: Profile, mock_llm: MagicMock
    ) -> None:
        score = score_job(fresh_job, profile)
        agent = PrototypeAgent(llm=mock_llm)

        result = await agent.generate(fresh_job, score, profile)

        for section in (
            f"# Demo: {fresh_job.title}",
            "## What this demonstrates",
            "## How to run",
            "pip install -r requirements.txt",
            "python demo.py",
            "## What a full implementation would add",
            "## About me",
        ):
            assert section in result.readme_md

        assert profile.freelancer.github in result.readme_md

    async def test_config_example_env_matches_category(
        self, fresh_job: JobPost, profile: Profile, mock_llm: MagicMock
    ) -> None:
        job = _job(
            fresh_job,
            title="Build an AI chatbot",
            description="Use OpenAI to build an assistant.",
            skills_required=["langchain"],
        )
        score = score_job(job, profile)
        agent = PrototypeAgent(llm=mock_llm)

        result = await agent.generate(job, score, profile)

        assert result.category == "ai_pipeline"
        assert "OPENAI_API_KEY" in result.config_example_env

    async def test_invalid_syntax_from_llm_does_not_raise(
        self, fresh_job: JobPost, profile: Profile, mock_llm: MagicMock
    ) -> None:
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=MagicMock(
                demo_script="def broken(:\n    pass",
                what_it_demonstrates="broken",
                extension_bullet_1="a",
                extension_bullet_2="b",
                extension_bullet_3="c",
            )
        )
        score = score_job(fresh_job, profile)
        agent = PrototypeAgent(llm=mock_llm)

        result = await agent.generate(fresh_job, score, profile)

        assert result.demo_script == "def broken(:\n    pass"
