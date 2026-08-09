"""Proposal Agent — drafts a human-sounding, professional Upwork cover letter.

Template category selection, the timeline/bid estimate, and the milestone
count/split are all deterministic (no LLM involved), so they stay fast, free,
and fully unit-testable. Only the hook, plan bullets, close, and milestone
descriptions — the parts that need to read the job description and respond
specifically to it — go through the LLM, via structured output so the template
can be filled reliably.

The full draft (hook/proof/plan/close/pricing) is hard-capped at 1200
characters. That cap is enforced by truncating the LLM-generated hook/bullets/
close to fit *before* the template is filled, so the static shell (and the
pricing line at the end) is never the part that gets cut off. The milestone
breakdown, when there is one, is rendered separately via
`render_milestones_block` and is NOT part of that 1200-character budget --
Upwork's actual milestone breakdown is a structured feature set up separately
from the cover-letter text, so it's informational rather than paste-ready.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger
from pydantic import BaseModel, Field

from ulysses.config.profile import Profile
from ulysses.config.settings import get_settings
from ulysses.models import BudgetType, GeneratedProposal, JobPost, JobScore, Milestone
from ulysses.tools.example_retrieval import (
    ExampleProposal,
    find_best_matching_example,
    load_example_proposals,
)
from ulysses.tools.llm import ainvoke_with_retry, get_llm

__all__ = [
    "ProposalAgent",
    "classify_category",
    "estimate_timeline_and_bid",
    "milestone_count_for_days",
    "render_milestones_block",
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
_MAX_CHARS = 1200  # raised from 800: hook+3 bullets+close no longer fit 800 without truncation
_MAX_EMOJI = 2
_MAX_OUTPUT_TOKENS = 500  # bumped from 300: "close" + up to 4 milestone descriptions need room
_LLM_TEMPERATURE = 0.9  # higher than the client default -- reduces generic/repetitive phrasing
_DESCRIPTION_INPUT_CHAR_LIMIT = 500
_CHAR_BUDGET_SAFETY_MARGIN = 10
_EMOJI_RE = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]")
_MILESTONE_DAY_THRESHOLDS: tuple[int, ...] = (2, 5, 9)  # <=2d:1, <=5d:2, <=9d:3, else:4

_SYSTEM_PROMPT = """You are ghostwriting Upwork proposals as Alejandro Garcia, a Python \
freelancer -- a real, specific, skilled person, not an AI assistant. The proposal must read like \
Alejandro wrote it himself in two minutes, not like a template filled in by software.

Voice, in order of priority: professional and straightforward first, warm and human second, \
persuasive third. Confident, not arrogant. Persuasive, but only a little -- never oversell, \
never sound like you're closing a deal. Write the way a sharp, busy freelancer would actually \
text a client he's genuinely glad to work with.

Two examples of the same insight, one weak (AI-sounding) and one strong (how Alejandro actually \
writes) -- match the register of the strong ones, not the weak ones:

Weak: "Standard Excel API scripts often overwrite manual pricing or drop leading zeroes from \
UPCs, which can cause significant data integrity issues."
Strong: "Losing your manual pricing overrides every time you refresh from the API would drive \
anyone crazy -- that one's an easy fix."

Weak: "I would be delighted to assist you with this project and look forward to hearing from \
you soon regarding any questions you may have."
Strong: "Happy to answer anything before we start -- just ask."

Connect with this specific client:
- Reuse 1-2 exact words or phrases from their job description (their tool names, their term for \
the problem) in the hook or plan -- proof you actually read their posting, not a canned reply.
- Frame the hook around what the current situation is costing them -- time, money, or risk of a \
mistake -- rather than just naming the technical mechanism. Never invent a stake the posting \
doesn't reasonably support, and never manufacture urgency or scarcity ("only taking 2 clients \
this month", "spots filling fast") -- that reads as fake from a new account and is a known red \
flag on Upwork.
- Be maximally concrete: real tool/library names, real numbers, a specific mechanism. Vague \
reassurance ("I'll make sure it's reliable") is both generic and an AI tell.
- The proof-of-work repo and the plan's concrete steps ARE the risk-reduction for a 0-review \
account -- never say "trust me" or "I guarantee quality," let specificity carry it instead.
- Naturally imply Alejandro is new to Upwork but not new to the field, without saying so \
directly.

Structure:
- Hook: 1-2 sentences, a pattern interrupt naming their specific pain in plain human language. \
No greeting, ever ("Hello", "Dear...", "Hi there"). Never "I am interested in your project" or \
any variation of it.
- Plan: three concrete, specific steps -- not vague filler like "I'll analyze your needs".
- Milestones (only when asked for more than zero): one short deliverable description per \
milestone, under 12 words each, in delivery order, each building toward full completion -- not \
restatements of the plan bullets, but what gets handed over at that payment point.
- Close: one sentence, warm and confident, ending in a genuine, low-key call to action -- a \
specific question about the job, or an open invitation to ask questions. Never generic \
boilerplate that could be pasted into any other proposal unchanged. Never "looking forward to \
hearing from you" or other stock sign-offs -- that's an AI tell too. Never pushy or salesy \
("Let's hop on a call!").

Hard bans (these read as AI-generated immediately if they appear anywhere): "I understand you \
need/that you...", "In today's...", "Leveraging my expertise...", "I would be delighted...", \
using "Furthermore/Moreover/Additionally" as transitions, excessive em-dashes, generic hedging \
("it seems", "I believe I could potentially").

Length and formatting: aim for a short hook (one sentence, two at most) and short bullets \
(under 15 words each) -- the full proposal has a hard 1200-character budget, so verbosity gets \
truncated. At most one or two emoji total, only if genuinely fitting (no faces, no generic \
sparkle/rocket spam) -- omit entirely if unsure. Do not mention pricing or timeline; that's \
handled separately.
"""

_USER_PROMPT_TEMPLATE = """Job title: {title}
Job description: {description}
Skills required: {skills}
Best matching proof-of-work repo: {proof_repo} -- {proof_repo_url}
Milestones to propose: {milestone_count} (write exactly this many short milestone descriptions; \
write an empty list if this is 0)
"""


class _ProposalLLMOutput(BaseModel):
    """Structured output the LLM must produce: the creative parts of the proposal."""

    hook: str = Field(
        description="A 1-2 sentence pattern-interrupt opener referencing the job's specific "
        "pain point, in warm, human language. No greeting."
    )
    plan_bullet_1: str = Field(description="First concrete step of the solution for this job.")
    plan_bullet_2: str = Field(description="Second concrete step of the solution for this job.")
    plan_bullet_3: str = Field(description="Third concrete step of the solution for this job.")
    close: str = Field(
        description="One warm, confident closing sentence plus a soft call to action "
        "(e.g. inviting a quick question or a short chat). Specific to this job where "
        "possible -- never generic boilerplate repeated verbatim across proposals."
    )
    milestones: list[str] = Field(
        default_factory=list,
        description="Exactly as many short, specific milestone deliverable descriptions as "
        "requested by the user prompt's 'Milestones to propose' count, in delivery order. "
        "Empty list if that count is 0.",
    )


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


def estimate_timeline_and_bid(job: JobPost, profile: Profile) -> tuple[str, float, int | None]:
    """Estimate a timeline and bid from the job's budget, falling back to the freelancer's rate.

    Fixed-price jobs: bid the listed midpoint, scale timeline off it (roughly
    one day per $75, clamped to 1-14 days). Hourly jobs: bid the listed rate,
    "Ongoing". Unlisted budget: a generic 3-day estimate at the freelancer's rate.

    Returns:
        A `(timeline, bid_usd, days)` tuple. `days` is the raw day estimate used for
        milestone-count tiering (see `milestone_count_for_days`), or `None` for
        hourly contracts, which don't have milestones on Upwork.
    """
    midpoint = job.budget.midpoint
    if midpoint is None:
        default_days = 3
        bid = profile.freelancer.rate_usd_hr * default_days * 4
        return f"{default_days} days", round(bid, 0), default_days

    if job.budget.type is BudgetType.HOURLY:
        return "Ongoing (hourly)", round(midpoint, 0), None

    days = max(1, min(14, round(midpoint / 75)))
    return f"{days} day{'s' if days != 1 else ''}", round(midpoint, 0), days


def _render_pricing_line(budget_type: BudgetType, timeline: str, bid_usd: float) -> str:
    """Render the timeline/bid as one natural sentence, phrased for fixed vs. hourly work.

    Fixed-price and unknown-budget jobs get a project-style sentence ("I can have this
    done in 3 days for $200"); hourly jobs get a rate-style sentence instead, since
    "done in Ongoing (hourly) for $40" doesn't parse as English.
    """
    if budget_type is BudgetType.HOURLY:
        return f"I'd suggest ${bid_usd:.0f}/hr, ongoing as needed."
    return f"I can have this done in {timeline} for ${bid_usd:.0f}."


def milestone_count_for_days(days: int | None) -> int:
    """Return how many delivery milestones a fixed-price job should get, based on its day estimate.

    Ranges from 1 (quick, single-deliverable work) to 4 (the longest jobs the
    estimator produces), matching real Upwork fixed-price conventions where a short
    job gets a single payment and a longer one gets a staged breakdown. Hourly
    contracts (`days is None`) don't have milestones on Upwork at all -- callers
    should skip milestone generation entirely in that case.
    """
    if days is None:
        return 0
    for count, threshold in enumerate(_MILESTONE_DAY_THRESHOLDS, start=1):
        if days <= threshold:
            return count
    return len(_MILESTONE_DAY_THRESHOLDS) + 1


def _split_amount_evenly(total: float, parts: int) -> list[float]:
    """Split a dollar total into `parts` whole-dollar shares that sum exactly to `total`."""
    share = round(total / parts)
    amounts = [float(share)] * (parts - 1)
    amounts.append(round(total - share * (parts - 1), 2))
    return amounts


def _split_days_evenly(total_days: int, parts: int) -> list[int]:
    """Split a day total into `parts` whole-day shares that sum exactly to `total_days`."""
    base, remainder = divmod(total_days, parts)
    return [base + 1 if i < remainder else base for i in range(parts)]


def _normalize_milestone_descriptions(descriptions: list[str], count: int) -> list[str]:
    """Force the LLM's milestone descriptions to exactly `count` items.

    Structured-output schemas can't enforce a dynamic list length, so the LLM is
    asked for an exact count via the prompt but may still return the wrong number --
    pad with a generic fallback (extremely unlikely to be needed in practice) or
    truncate extras.
    """
    cleaned = [description.strip() for description in descriptions if description.strip()]
    cleaned = cleaned[:count]
    while len(cleaned) < count:
        cleaned.append(f"Milestone {len(cleaned) + 1} deliverable")
    return cleaned


def _build_milestones(
    descriptions: list[str], count: int, total_days: int | None, bid_usd: float
) -> list[Milestone]:
    """Assemble the final milestone breakdown: LLM descriptions, deterministic $/day splits.

    Returns an empty list when `count` is 0 -- either because the job is hourly
    (no `total_days`) or because a single-milestone job doesn't warrant a separate
    breakdown section (that case is just the standard one-payment pricing line).
    """
    if count == 0 or total_days is None:
        return []
    amounts = _split_amount_evenly(bid_usd, count)
    days = _split_days_evenly(total_days, count)
    normalized = _normalize_milestone_descriptions(descriptions, count)
    return [
        Milestone(description=description, amount_usd=amount, days=day)
        for description, amount, day in zip(normalized, amounts, days, strict=True)
    ]


def render_milestones_block(milestones: list[Milestone]) -> str:
    """Render a milestone breakdown as a separate reference block, outside the 1200-char cap.

    Not part of `full_text` -- Upwork's actual milestone breakdown is a structured
    feature set up separately from the cover-letter text, so this is informational,
    for Alejandro to copy into that UI when he sets up the contract, not meant to be
    pasted verbatim into the proposal box itself. Returns "" when there are none
    (hourly jobs, or jobs too small to warrant a multi-payment breakdown).
    """
    if not milestones:
        return ""
    lines = [
        f"{index}. {milestone.description} — ${milestone.amount_usd:.0f} "
        f"(~{milestone.days} day{'s' if milestone.days != 1 else ''})"
        for index, milestone in enumerate(milestones, start=1)
    ]
    return "\n\nSuggested milestones:\n" + "\n".join(lines)


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

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        examples: list[ExampleProposal] | None = None,
    ) -> None:
        """Create a Proposal Agent.

        Args:
            llm: Chat model to use. Defaults to the shared client from `get_llm()`.
            examples: Curated example proposals for few-shot retrieval. Defaults
                to `load_example_proposals()` (the bundled
                `config/example_proposals.yaml`).
        """
        self._llm = llm or get_llm()
        self._examples = examples if examples is not None else load_example_proposals()

    async def generate(self, job: JobPost, score: JobScore, profile: Profile) -> GeneratedProposal:
        """Generate a complete, template-filled proposal draft for a scored job.

        Keeps token usage bounded on both sides of the call: the job
        description is truncated before it's sent to the LLM, and the
        completion itself is capped via `max_tokens`. The 1200-character
        output budget is enforced by truncating the hook/bullets/close to fit
        *before* filling the template, so the static shell — including the
        pricing line at the end — always survives intact. Milestones (when
        there are any) are assembled separately and are not part of that
        budget at all.
        """
        category = classify_category(job)
        proof_repo, proof_repo_url = _select_proof_repo(score, profile)
        timeline, bid_usd, days = estimate_timeline_and_bid(job, profile)
        pricing_line = _render_pricing_line(job.budget.type, timeline, bid_usd)

        raw_milestone_count = milestone_count_for_days(days)
        milestone_count = raw_milestone_count if raw_milestone_count >= 2 else 0

        structured_llm = self._llm.bind(
            max_tokens=_MAX_OUTPUT_TOKENS, temperature=_LLM_TEMPERATURE
        ).with_structured_output(_ProposalLLMOutput)

        try:
            best_example = await find_best_matching_example(job, self._examples)
        except httpx.HTTPError as exc:
            logger.bind(job_id=job.id, agent="proposal").warning(
                "Example-proposal retrieval failed, drafting without a few-shot example: {}", exc
            )
            best_example = None

        prompt: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        prompt.extend(_build_few_shot_turns(best_example))
        prompt.append(
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
                    milestone_count=milestone_count,
                ),
            }
        )

        start = time.monotonic()
        llm_output: _ProposalLLMOutput = await ainvoke_with_retry(structured_llm, prompt)
        elapsed = time.monotonic() - start
        logger.bind(job_id=job.id, agent="proposal").info(
            "LLM call complete: model={} latency={:.2f}s", get_settings().llm_model, elapsed
        )

        hook, plan_bullets, close = _fit_content_to_budget(
            hook=llm_output.hook.strip(),
            plan_bullets=[
                llm_output.plan_bullet_1.strip(),
                llm_output.plan_bullet_2.strip(),
                llm_output.plan_bullet_3.strip(),
            ],
            close=llm_output.close.strip(),
            category=category,
            proof_repo=proof_repo,
            proof_repo_url=proof_repo_url,
            pricing_line=pricing_line,
        )
        full_text = _fill_template(
            category=category,
            hook=hook,
            proof_repo=proof_repo,
            proof_repo_url=proof_repo_url,
            plan_bullets=plan_bullets,
            close=close,
            pricing_line=pricing_line,
        )
        milestones = _build_milestones(llm_output.milestones, milestone_count, days, bid_usd)

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
            close=close,
            proof_repo=proof_repo,
            proof_repo_url=proof_repo_url,
            timeline=timeline,
            bid_usd=bid_usd,
            full_text=full_text,
            milestones=milestones,
        )


def _select_proof_repo(score: JobScore, profile: Profile) -> tuple[str, str]:
    if score.matched_repos:
        top = score.matched_repos[0]
        return top.repo_name, top.url
    return "my portfolio", profile.freelancer.github


def _build_few_shot_turns(example: ExampleProposal | None) -> list[dict[str, str]]:
    """Build the user/assistant few-shot turn pair for a matched example, if any.

    The assistant turn is JSON matching `_ProposalLLMOutput`'s field names, not
    free-form prose -- confirmed against the real configured LLM to work well
    as a demonstration ahead of a `.with_structured_output(...)` call, since it
    mirrors the exact shape being asked for rather than mixing registers.
    Returns an empty list (no turns added) when there's no example to show.
    """
    if example is None:
        return []
    user_turn = {
        "role": "user",
        "content": _USER_PROMPT_TEMPLATE.format(
            title=example.job_title,
            description=example.job_description,
            skills=", ".join(example.skills) or "not specified",
            proof_repo=example.proof_repo,
            proof_repo_url=example.proof_repo_url,
            milestone_count=len(example.milestones),
        ),
    }
    assistant_turn = {
        "role": "assistant",
        "content": json.dumps(
            {
                "hook": example.hook,
                "plan_bullet_1": example.plan_bullet_1,
                "plan_bullet_2": example.plan_bullet_2,
                "plan_bullet_3": example.plan_bullet_3,
                "close": example.close,
                "milestones": example.milestones,
            }
        ),
    }
    return [user_turn, assistant_turn]


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
    close: str,
    category: str,
    proof_repo: str,
    proof_repo_url: str,
    pricing_line: str,
) -> tuple[str, list[str], str]:
    """Truncate the hook/bullets/close so the filled template fits the character budget.

    Measures the static shell (everything except LLM-generated content) with real
    values already in place, then splits the remaining budget evenly across the five
    LLM-generated fields (hook, three bullets, close). In practice a real close reads
    as a full sentence -- often a question -- much closer in length to the hook than
    to a single plan bullet, so an equal split (rather than giving close a smaller
    share) is what actually keeps both the hook and close from being cut off mid-word.
    `close` also gets a defensive sentence-ending touch-up: if it's truncated (or the
    LLM simply forgot terminal punctuation), it's given a period so it never runs
    straight into the pricing line as one garbled sentence. This keeps the pricing
    line -- the last thing in the template -- from ever being the part that gets cut
    off.
    """
    shell_length = len(
        _fill_template(
            category=category,
            hook="",
            proof_repo=proof_repo,
            proof_repo_url=proof_repo_url,
            plan_bullets=["", "", ""],
            close="",
            pricing_line=pricing_line,
        )
    )
    content_budget = max(0, _MAX_CHARS - shell_length - _CHAR_BUDGET_SAFETY_MARGIN)
    per_field_budget = content_budget // 5

    fitted_hook = _truncate_at_word_boundary(hook, per_field_budget)
    fitted_bullets = [
        _truncate_at_word_boundary(bullet, per_field_budget) for bullet in plan_bullets
    ]
    fitted_close = _ensure_sentence_ending(_truncate_at_word_boundary(close, per_field_budget))
    return fitted_hook, fitted_bullets, fitted_close


def _ensure_sentence_ending(text: str) -> str:
    """Ensure `text` ends with terminal punctuation so it can't run into the next sentence.

    `close` sits directly next to `pricing_line` in the template with only a space
    between them -- if truncation (or the LLM itself) leaves it without a period,
    question mark, or exclamation point, the two would otherwise read as one garbled
    run-on sentence.
    """
    stripped = text.rstrip()
    if not stripped or stripped[-1] in ".!?":
        return stripped
    return stripped + "."


def _fill_template(
    *,
    category: str,
    hook: str,
    proof_repo: str,
    proof_repo_url: str,
    plan_bullets: list[str],
    close: str,
    pricing_line: str,
) -> str:
    template = (_TEMPLATES_DIR / f"{category}.txt").read_text(encoding="utf-8")
    return template.format(
        hook=hook.strip(),
        proof_repo=proof_repo,
        proof_repo_url=proof_repo_url,
        plan_bullet_1=plan_bullets[0].strip(),
        plan_bullet_2=plan_bullets[1].strip(),
        plan_bullet_3=plan_bullets[2].strip(),
        close=close.strip(),
        pricing_line=pricing_line,
    ).strip()
