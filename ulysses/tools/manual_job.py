"""Extracts a structured `JobPost` from a raw, manually pasted Upwork job listing.

Unlike `ulysses.tools.job_parser.parse_job_email` (which requires an HTML
`<a href="/jobs/...">` anchor to key off of), plain copy-pasted text has no
reliable structural signal to parse heuristically -- so this module hands the
whole pasted block to the LLM via structured output, following the same
dependency-injection pattern used by `ProposalAgent`/`PrototypeAgent`.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from ulysses.models import BudgetRange, BudgetType, JobPost
from ulysses.tools.llm import ainvoke_with_retry, get_llm

__all__ = ["ManualJobParseError", "extract_job_from_text"]

_MAX_INPUT_CHARS = 12_000
_MAX_OUTPUT_TOKENS = 1000
_MIN_TITLE_CHARS = 3
_MIN_DESCRIPTION_CHARS = 20
_MAX_POSTED_MINUTES_AGO = 60 * 24 * 365 * 5  # 5 years -- a sanity clamp, not a real limit

_SYSTEM_PROMPT = """You extract structured data from a raw, manually copy-pasted Upwork job \
listing. The text was pasted directly from the Upwork website by a freelancer, so it may \
include extra page chrome (navigation text, "Apply Now" buttons, unrelated boilerplate) mixed \
in with the actual listing -- ignore anything that isn't part of the job posting itself.

Extract only what is actually present in the text. Never invent a client rating, hire count, \
budget, or posting time that isn't shown -- leave those fields null/default instead of guessing.

For description, copy the full scope-of-work text as written, without summarizing, shortening, \
or rephrasing it.

For posted_minutes_ago, convert whatever relative time phrase appears (e.g. "posted 3 hours \
ago", "5 minutes ago", "2 days ago") into a plain integer number of minutes elapsed. If no \
posting time appears anywhere in the text, leave it null.
"""

_USER_PROMPT_TEMPLATE = """Pasted Upwork job listing:

{raw_text}
"""


class ManualJobParseError(Exception):
    """Raised when a pasted block of text doesn't look like a real job listing."""


class _ManualJobExtraction(BaseModel):
    """Structured fields the LLM must extract from a raw pasted Upwork job listing."""

    title: str = Field(description="The job's title/headline, verbatim or near-verbatim.")
    description: str = Field(
        description="The full job description/scope of work, verbatim as pasted -- do not "
        "summarize or shorten it."
    )
    budget_type: BudgetType = Field(
        default=BudgetType.UNKNOWN,
        description="'fixed' for a fixed-price job, 'hourly' for an hourly job, 'unknown' if "
        "no budget is stated at all.",
    )
    budget_min: float | None = Field(
        default=None, description="Minimum budget/rate amount, in the listing's currency."
    )
    budget_max: float | None = Field(
        default=None,
        description="Maximum budget/rate amount. Same as budget_min if only one number is given.",
    )
    budget_currency: str = Field(default="USD", description="Three-letter currency code, e.g. USD.")
    skills_required: list[str] = Field(
        default_factory=list, description="Skills/tags listed for the job."
    )
    client_hires: int = Field(
        default=0, description="Number of previous hires by this client, 0 if not stated."
    )
    client_rating: float | None = Field(
        default=None, description="Client's star rating out of 5, if shown."
    )
    payment_verified: bool = Field(
        default=False,
        description="Whether the listing shows the client's payment method as verified.",
    )
    proposals_count: int | None = Field(
        default=None,
        description="Number of proposals already submitted, if shown (use the upper bound of "
        "a range like '10 to 15 proposals').",
    )
    posted_minutes_ago: int | None = Field(
        default=None,
        description="Minutes elapsed since posting, converted from relative time shown in the "
        "text (e.g. '3 hours ago' -> 180). Null if no posting time is shown.",
    )
    url: str | None = Field(
        default=None,
        description="The job's Upwork URL, only if one literally appears in the pasted text. "
        "Null if no URL appears anywhere.",
    )


async def extract_job_from_text(raw_text: str, *, llm: BaseChatModel | None = None) -> JobPost:
    """Extract a structured `JobPost` from a raw, manually pasted Upwork job listing.

    Args:
        raw_text: The raw text the user pasted (may include extra page chrome).
        llm: Chat model override, primarily for tests. Defaults to `get_llm()`.

    Returns:
        A fully-populated `JobPost`. `id`/`url` are synthesized (a `manual://`
        placeholder, unless the LLM found a real URL in the text) and
        `posted_at` is computed from the extracted relative time (or "now"
        if none was found).

    Raises:
        ManualJobParseError: If `raw_text` is blank/too short, or if the
            extracted title/description are too short to be a real listing.
    """
    stripped = raw_text.strip()
    if len(stripped) < _MIN_DESCRIPTION_CHARS:
        raise ManualJobParseError(
            "That doesn't look like a job listing -- paste the full posting text and try again."
        )

    llm = llm or get_llm()
    structured_llm = llm.bind(max_tokens=_MAX_OUTPUT_TOKENS).with_structured_output(
        _ManualJobExtraction
    )
    prompt = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _USER_PROMPT_TEMPLATE.format(
                raw_text=_truncate_at_word_boundary(stripped, _MAX_INPUT_CHARS)
            ),
        },
    ]
    extraction: _ManualJobExtraction = await ainvoke_with_retry(structured_llm, prompt)

    title = extraction.title.strip()
    description = extraction.description.strip()
    if len(title) < _MIN_TITLE_CHARS or len(description) < _MIN_DESCRIPTION_CHARS:
        raise ManualJobParseError(
            "Couldn't find a job title/description in that text -- check what you pasted and "
            "try again."
        )

    job_id, url = _synthesize_id_and_url(extraction.url)
    return JobPost(
        id=job_id,
        title=title,
        description=description,
        budget=BudgetRange(
            type=extraction.budget_type,
            min_amount=extraction.budget_min,
            max_amount=extraction.budget_max,
            currency=extraction.budget_currency,
        ),
        skills_required=extraction.skills_required,
        client_hires=max(0, extraction.client_hires),
        client_rating=extraction.client_rating,
        payment_verified=extraction.payment_verified,
        proposals_count=extraction.proposals_count,
        posted_at=_posted_at_from_minutes_ago(extraction.posted_minutes_ago),
        url=url,
    )


def _synthesize_id_and_url(found_url: str | None) -> tuple[str, str]:
    """Derive a stable `(id, url)` pair, synthesizing a placeholder URL if none was found.

    A URL literally present in the pasted text is trusted as-is and hashed
    for the id -- the same `sha256(url)[:16]` idea as
    `job_parser._extract_job_id` -- so re-pasting the same listing updates
    the same DB row instead of creating a duplicate. Otherwise a random
    `manual://<uuid4>` stands in for the unique, indexed `Job.url` column.
    """
    url = found_url.strip() if found_url and found_url.strip() else f"manual://{uuid4()}"
    job_id = hashlib.sha256(url.encode()).hexdigest()[:16]
    return job_id, url


def _posted_at_from_minutes_ago(minutes_ago: int | None) -> datetime:
    """Compute an absolute `posted_at` from an LLM-extracted relative offset.

    Clamped to a generous [0, 5 years] range so a hallucinated or malformed
    value can't produce a nonsensical result -- a safety clamp, not an
    expected real-world case.
    """
    now = datetime.now(UTC)
    if minutes_ago is None:
        return now
    clamped = max(0, min(minutes_ago, _MAX_POSTED_MINUTES_AGO))
    return now - timedelta(minutes=clamped)


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Truncate `text` to at most `max_chars`, cutting at the last whole word."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return (truncated[:last_space] if last_space > 0 else truncated).rstrip()
