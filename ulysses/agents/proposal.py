"""Proposal Agent — drafts a human-sounding, professional Upwork cover letter.

Template category selection and the timeline/bid estimate are deterministic
(no LLM involved), so they stay fast, free, and fully unit-testable. Only the
hook and plan bullets — the parts that need to read the job description and
respond specifically to it — go through the LLM, via structured output so the
template can be filled reliably.

The full draft is hard-capped at 800 characters. That cap is enforced by
truncating the LLM-generated hook/bullets to fit *before* the template is
filled, so the static shell (and the timeline/bid line at the end) is never
the part that gets cut off.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger
from pydantic import BaseModel, Field

from ulysses.config.profile import Profile
from ulysses.config.settings import get_settings
from ulysses.models import BudgetType, GeneratedProposal, JobPost, JobScore
from ulysses.tools.llm import ainvoke_with_retry, get_llm

__all__ = [
    "ProposalAgent",
    "classify_category",
    "estimate_timeline_and_bid",
    "validate_proposal_text",
]

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates" / "proposals"

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "scraping": (
        "scrape",
        "scraper",
        "scraping",
        "crawl",
        "beautifulsoup",
        "playwright",
        "selenium",
        "web scraping",
    ),
    "api_dev": (
        "api",
        "rest api",
        "endpoint",
        "webhook",
        "fastapi",
        "flask",
        "third-party api",
        "integration",
    ),
    "data_pipeline": (
        "pipeline",
        "etl",
        "pandas",
        "transform",
        "csv",
        "excel",
        "spreadsheet",
        "reporting",
        "data pipeline",
    ),
    "ai_integration": (
        "openai",
        "llm",
        "gpt",
        "langchain",
        "ai agent",
        "chatbot",
        "machine learning",
        "nlp",
    ),
}
_DEFAULT_CATEGORY = "automation"

_FORBIDDEN_PHRASES: tuple[str, ...] = (
    "hello",
    "dear",
    "i am interested",
    "dear hiring manager",
    "in today's",
    "leveraging my",
    "i understand you need",
    "i understand that you",
    "i would be delighted",
    "furthermore",
    "moreover",
)
_MAX_CHARS = 800
_MAX_EMOJI = 2
_MAX_OUTPUT_TOKENS = 300
_DESCRIPTION_INPUT_CHAR_LIMIT = 500
_CHAR_BUDGET_SAFETY_MARGIN = 10
_EMOJI_RE = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]")

_SYSTEM_PROMPT = """You write Upwork proposal openers and plans for Alejandro Garcia, a Python \
freelancer. You are ghostwriting as him -- a real, specific, skilled person -- not an AI \
assistant, and it must not read like one.

Tone: polite, warm, and professional, but brief. Confident without being arrogant, human \
without being casual. Every sentence should earn its place -- this is a paid API call, so be \
economical with words without losing substance or warmth.

Hard rules:
- Never start with a greeting ("Hello", "Dear...", "Hi there").
- Never say "I am interested in your project" or any variation of it.
- Do NOT sound like an AI wrote this. Ban these tells specifically: "I understand you need/that \
you...", "In today's...", "Leveraging my expertise...", "I would be delighted...", using \
"Furthermore/Moreover/Additionally" as transitions, excessive em-dashes, and generic hedging \
("it seems", "I believe I could potentially"). Write the way a sharp, busy professional would \
actually text a promising client, not the way a chatbot writes a cover letter.
- The hook must be a pattern interrupt: reference the client's specific pain point from the \
job description in the first sentence, not a generic statement.
- Each plan bullet is one concrete, specific step -- not vague filler like "I'll analyze your \
needs".
- Naturally imply Alejandro is new to Upwork but not new to the field, without saying so \
directly -- let the specificity of the plan and proof of work carry that.
- Be concise: aim for a short hook (one sentence, two at most) and short bullets (under 15 \
words each). The full proposal has a hard 800-character budget, so verbosity gets truncated.
- At most one or two emoji total, only if genuinely fitting and professional (no faces, no \
generic sparkle/rocket spam) -- omit entirely if unsure. A single well-placed emoji reads better \
than two.
- Do not mention pricing or timeline; that's handled separately.
"""

_USER_PROMPT_TEMPLATE = """Job title: {title}
Job description: {description}
Skills required: {skills}
Best matching proof-of-work repo: {proof_repo} -- {proof_repo_url}
"""


class _ProposalLLMOutput(BaseModel):
    """Structured output the LLM must produce: the creative parts of the proposal."""

    hook: str = Field(
        description="A 1-2 sentence pattern-interrupt opener referencing the job's specific "
        "pain point. No greeting."
    )
    plan_bullet_1: str = Field(description="First concrete step of the solution for this job.")
    plan_bullet_2: str = Field(description="Second concrete step of the solution for this job.")
    plan_bullet_3: str = Field(description="Third concrete step of the solution for this job.")


def classify_category(job: JobPost) -> str:
    """Classify a job into one of the proposal template categories.

    Deterministic keyword matching over the job title, description, and
    required skills. Falls back to "automation" when nothing matches clearly.
    """
    haystack = " ".join([job.title, job.description, *job.skills_required]).lower()
    best_category = _DEFAULT_CATEGORY
    best_hits = 0
    for category, keywords in _CATEGORY_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in haystack)
        if hits > best_hits:
            best_hits = hits
            best_category = category
    return best_category


def estimate_timeline_and_bid(job: JobPost, profile: Profile) -> tuple[str, float]:
    """Estimate a timeline and bid from the job's budget, falling back to the freelancer's rate.

    Fixed-price jobs: bid the listed midpoint, scale timeline off it (roughly
    one day per $75, clamped to 1-14 days). Hourly jobs: bid the listed rate,
    "Ongoing". Unlisted budget: a generic 3-day estimate at the freelancer's rate.
    """
    midpoint = job.budget.midpoint
    if midpoint is None:
        default_days = 3
        bid = profile.freelancer.rate_usd_hr * default_days * 4
        return f"{default_days} days", round(bid, 0)

    if job.budget.type is BudgetType.HOURLY:
        return "Ongoing (hourly)", round(midpoint, 0)

    days = max(1, min(14, round(midpoint / 75)))
    return f"{days} day{'s' if days != 1 else ''}", round(midpoint, 0)


def validate_proposal_text(text: str) -> list[str]:
    """Return every proposal style rule violation found in a draft, if any."""
    violations: list[str] = []
    lowered = text.lower()
    for phrase in _FORBIDDEN_PHRASES:
        if phrase in lowered:
            violations.append(f'contains forbidden phrase "{phrase}"')

    if len(text) > _MAX_CHARS:
        violations.append(f"exceeds {_MAX_CHARS} characters ({len(text)})")

    emoji_count = len(_EMOJI_RE.findall(text))
    if emoji_count > _MAX_EMOJI:
        violations.append(f"contains {emoji_count} emoji, max is {_MAX_EMOJI}")

    return violations


class ProposalAgent:
    """Generates a human-sounding, professional Upwork proposal draft for a scored job."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        """Create a Proposal Agent.

        Args:
            llm: Chat model to use. Defaults to the shared client from `get_llm()`.
        """
        self._llm = llm or get_llm()

    async def generate(self, job: JobPost, score: JobScore, profile: Profile) -> GeneratedProposal:
        """Generate a complete, template-filled proposal draft for a scored job.

        Keeps token usage bounded on both sides of the call: the job
        description is truncated before it's sent to the LLM, and the
        completion itself is capped via `max_tokens`. The 800-character
        output budget is enforced by truncating the hook/bullets to fit
        *before* filling the template, so the static shell — including the
        timeline/bid line at the end — always survives intact.
        """
        category = classify_category(job)
        proof_repo, proof_repo_url = _select_proof_repo(score, profile)
        timeline, bid_usd = estimate_timeline_and_bid(job, profile)

        structured_llm = self._llm.bind(max_tokens=_MAX_OUTPUT_TOKENS).with_structured_output(
            _ProposalLLMOutput
        )
        prompt = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_PROMPT_TEMPLATE.format(
                    title=job.title,
                    description=_truncate_at_word_boundary(
                        job.description, _DESCRIPTION_INPUT_CHAR_LIMIT
                    ),
                    skills=", ".join(job.skills_required) or "not specified",
                    proof_repo=proof_repo,
                    proof_repo_url=proof_repo_url,
                ),
            },
        ]

        start = time.monotonic()
        llm_output: _ProposalLLMOutput = await ainvoke_with_retry(structured_llm, prompt)
        elapsed = time.monotonic() - start
        logger.bind(job_id=job.id, agent="proposal").info(
            "LLM call complete: model={} latency={:.2f}s", get_settings().llm_model, elapsed
        )

        hook, plan_bullets = _fit_content_to_budget(
            hook=llm_output.hook.strip(),
            plan_bullets=[
                llm_output.plan_bullet_1.strip(),
                llm_output.plan_bullet_2.strip(),
                llm_output.plan_bullet_3.strip(),
            ],
            category=category,
            proof_repo=proof_repo,
            proof_repo_url=proof_repo_url,
            timeline=timeline,
            bid_usd=bid_usd,
        )
        full_text = _fill_template(
            category=category,
            hook=hook,
            proof_repo=proof_repo,
            proof_repo_url=proof_repo_url,
            plan_bullets=plan_bullets,
            timeline=timeline,
            bid_usd=bid_usd,
        )

        violations = validate_proposal_text(full_text)
        if violations:
            logger.bind(job_id=job.id, agent="proposal").warning(
                "Proposal draft violations: {}", violations
            )

        return GeneratedProposal(
            job_id=job.id,
            category=category,
            hook=hook,
            plan_bullets=plan_bullets,
            proof_repo=proof_repo,
            proof_repo_url=proof_repo_url,
            timeline=timeline,
            bid_usd=bid_usd,
            full_text=full_text,
        )


def _select_proof_repo(score: JobScore, profile: Profile) -> tuple[str, str]:
    if score.matched_repos:
        top = score.matched_repos[0]
        return top.repo_name, top.url
    return "my portfolio", profile.freelancer.github


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Truncate `text` to at most `max_chars`, cutting at the last whole word."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip()


def _fit_content_to_budget(
    *,
    hook: str,
    plan_bullets: list[str],
    category: str,
    proof_repo: str,
    proof_repo_url: str,
    timeline: str,
    bid_usd: float,
) -> tuple[str, list[str]]:
    """Truncate the hook/bullets so the filled template fits the character budget.

    Measures the static shell (everything except the LLM-generated hook and
    bullets) with real values already in place, then splits the remaining
    budget evenly across the hook and three bullets. This keeps the timeline
    and bid — the last thing in the template — from ever being the part that
    gets cut off.
    """
    shell_length = len(
        _fill_template(
            category=category,
            hook="",
            proof_repo=proof_repo,
            proof_repo_url=proof_repo_url,
            plan_bullets=["", "", ""],
            timeline=timeline,
            bid_usd=bid_usd,
        )
    )
    content_budget = max(0, _MAX_CHARS - shell_length - _CHAR_BUDGET_SAFETY_MARGIN)
    per_field_budget = content_budget // 4

    fitted_hook = _truncate_at_word_boundary(hook, per_field_budget)
    fitted_bullets = [
        _truncate_at_word_boundary(bullet, per_field_budget) for bullet in plan_bullets
    ]
    return fitted_hook, fitted_bullets


def _fill_template(
    *,
    category: str,
    hook: str,
    proof_repo: str,
    proof_repo_url: str,
    plan_bullets: list[str],
    timeline: str,
    bid_usd: float,
) -> str:
    template = (_TEMPLATES_DIR / f"{category}.txt").read_text(encoding="utf-8")
    return template.format(
        hook=hook.strip(),
        proof_repo=proof_repo,
        proof_repo_url=proof_repo_url,
        plan_bullet_1=plan_bullets[0].strip(),
        plan_bullet_2=plan_bullets[1].strip(),
        plan_bullet_3=plan_bullets[2].strip(),
        timeline=timeline,
        bid=f"{bid_usd:.0f}",
    ).strip()
