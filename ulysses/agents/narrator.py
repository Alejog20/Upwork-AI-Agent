"""Narrator Agent — a short, human-voiced explanation of a job's scoring verdict.

Alejandro already sees the raw score breakdown in a table; this explains the
*why* in Ulysses' own voice (see AGENTS.md), grounded strictly in the actual
computed score facts -- never inventing a reason that isn't in the data. One
cheap, fast LLM call (small `max_tokens`) per job, run for every
recommendation (APPLY_NOW/REVIEW/SKIP alike) -- explaining a call is what
makes it feel like judgment instead of a lookup table, and at this size it's
a small fraction of the cost a full proposal/prototype generation would be.
"""

from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from ulysses.config.profile import Profile
from ulysses.models import JobPost, JobScore
from ulysses.tools.llm import ainvoke_with_retry, get_llm

__all__ = ["NarratorAgent"]

_MAX_OUTPUT_TOKENS = 100
_TEMPERATURE = 0.8

_SYSTEM_PROMPT = """You are Ulysses, narrating your own scoring verdict on an Upwork job to \
Alejandro, the freelancer you work for. He can already see the raw numbers in a table above this \
message -- your job is to explain the *why* in one or two sentences, in plain human language, so \
the verdict feels like a colleague's quick take, not a dashboard readout.

Voice: intellectually sharp, direct, warm. Confident, not hedging. Dry wit used sparingly, never \
forced. No filler, no preamble ("Looking at this job..."), no restating the obvious ("This job \
scored X"). Get straight to what actually matters.

Ground everything in the specific facts you're given -- posting age, proposal count, client \
history, skill overlap, budget fit, red flags. Never invent a detail that isn't in the data. \
Reference at most 2 of the most decisive factors, not all of them -- pick what actually drove the \
verdict, not everything that happened to be true.

End with your actual call, stated plainly in your own voice -- never a bare label like \
"Recommendation: SKIP". Match this register:
- "New client, posted 9 minutes ago, and your scraper repo is a direct match. I'd apply now."
- "Stale posting with 20+ proposals already in -- you'd be shouting into a crowd. I'd skip this."
- "Budget's thin for the scope, but skills line up well. Worth a look, your call."

One to two sentences total. No more.
"""

_USER_PROMPT_TEMPLATE = """Job: {title}
Posted: {posted_age}
Proposals so far: {proposals_count}
Client history: {client_history}
Payment verified: {payment_verified}
Budget: {budget}
Skill match: {skill_overlap}
Red flags: {red_flags}
Total score: {total_score}/100
Category: {category}
Recommendation: {recommendation}
"""


class _NarrationOutput(BaseModel):
    """Structured output the LLM must produce: the verdict explanation."""

    blurb: str = Field(
        description="1-2 sentence explanation of the scoring verdict, in Ulysses' voice."
    )


class NarratorAgent:
    """Generates a short, human-voiced explanation of a job's score/recommendation."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        """Create a Narrator Agent.

        Args:
            llm: Chat model to use. Defaults to the shared client from `get_llm()`.
        """
        self._llm = llm or get_llm()

    async def narrate(self, job: JobPost, score: JobScore, profile: Profile) -> str:
        """Generate a short explanation of why a job scored the way it did."""
        structured_llm = self._llm.bind(
            max_tokens=_MAX_OUTPUT_TOKENS, temperature=_TEMPERATURE
        ).with_structured_output(_NarrationOutput)
        prompt = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_PROMPT_TEMPLATE.format(
                    title=job.title,
                    posted_age=_format_posted_age(job.posted_at),
                    proposals_count=(
                        job.proposals_count if job.proposals_count is not None else "not shown"
                    ),
                    client_history=_format_client_history(job.client_hires),
                    payment_verified="yes" if job.payment_verified else "not verified",
                    budget=str(job.budget),
                    skill_overlap=_format_skill_overlap(job.skills_required, profile.skills.all),
                    red_flags=", ".join(score.red_flags) if score.red_flags else "none",
                    total_score=f"{score.total_score:.0f}",
                    category=score.gig_category.value,
                    recommendation=score.recommendation.value,
                ),
            },
        ]
        output: _NarrationOutput = await ainvoke_with_retry(structured_llm, prompt)
        return output.blurb.strip()


def _format_posted_age(posted_at: datetime) -> str:
    """Render how long ago a job was posted as a short natural-language phrase."""
    age_minutes = (datetime.now(UTC) - posted_at).total_seconds() / 60
    if age_minutes < 60:
        return f"{max(0, round(age_minutes))} minutes ago"
    age_hours = age_minutes / 60
    if age_hours < 24:
        return f"{round(age_hours)} hours ago"
    age_days = age_hours / 24
    return f"{round(age_days)} days ago"


def _format_client_history(client_hires: int) -> str:
    """Render the client's hire count as a short natural-language phrase."""
    if client_hires == 0:
        return "brand new client, 0 previous hires"
    return f"{client_hires} previous hire{'s' if client_hires != 1 else ''}"


def _format_skill_overlap(skills_required: list[str], profile_skills: list[str]) -> str:
    """Render which of the job's required skills actually match the freelancer's profile."""
    if not skills_required:
        return "no specific skills listed"
    required = {skill.strip().lower() for skill in skills_required}
    known = {skill.strip().lower() for skill in profile_skills}
    matched = required & known
    if not matched:
        return f"none of the required skills match: {', '.join(sorted(required))}"
    return f"{len(matched)} of {len(required)} required skills match: {', '.join(sorted(matched))}"
