"""Chat Agent — open-ended, natural-language conversation about a job or the queue.

Unlike `NarratorAgent`/`ProposalAgent`/`PrototypeAgent`, which fill a fixed
template via structured output, this agent's job is free-form back-and-forth,
so it returns plain text. It is stateless: callers (the `chat`/`advise` CLI
commands) own the conversation history and the system prompt for a given
conversation, and pass both in on every call -- the same construction
convention as the other agents (`llm` optional, defaults to `get_llm()`), just
without anything to hold between turns.

Cost controls are deliberate and explicit here, since an interactive chat is
much easier to run up a bill on than the one-shot calls the other agents make:
- `max_tokens` is capped low -- replies should be short, not essays.
- `thinking` is explicitly disabled. Claude Sonnet 5 runs adaptive thinking by
  default whenever `thinking` is omitted entirely, which would silently spend
  reasoning tokens on every casual turn.
- `cache_control: {"type": "ephemeral"}` is bound on every call. For a direct
  Anthropic client (confirmed by reading `langchain_anthropic`'s
  `chat_models.py`) this is forwarded straight to the API's own top-level
  caching convenience param rather than hoisted client-side, which auto-caches
  the last eligible block of the request. Since the system prompt plus prior
  history is byte-identical between turns, later turns read that prefix from
  cache (~0.1x cost) instead of paying full price for the whole growing
  transcript on every single turn.

`build_job_context_message`/`build_queue_digest` are deliberately pure,
LLM-free functions -- they produce the *content* a caller appends to
`CHAT_SYSTEM_PROMPT` to build a conversation's system prompt, kept separate
from persona text so both halves are independently testable.
"""

from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.language_models.chat_models import BaseChatModel

from ulysses.config.profile import Profile
from ulysses.models import GeneratedProposal, GeneratedPrototype, JobPost, JobScore
from ulysses.tools.db import Job
from ulysses.tools.llm import ainvoke_with_retry, get_llm

__all__ = [
    "CHAT_SYSTEM_PROMPT",
    "ChatAgent",
    "build_advisor_profile_summary",
    "build_job_context_message",
    "build_queue_digest",
]

_MAX_OUTPUT_TOKENS = 700
_MAX_QUEUE_DIGEST_JOBS = 30

CHAT_SYSTEM_PROMPT = """You are Ulysses, Alejandro's freelance-business copilot -- the same \
voice that scores jobs and drafts proposals, now talking with him directly instead of filling \
a template. He can paste a job and get a scored, drafted proposal and demo automatically; this \
conversation is for everything beyond that: refining a draft, questioning a call, or talking \
through the queue and his strategy.

Voice: intellectually sharp, direct, warm. Confident, not hedging. No filler, no preamble \
("Great question!", "I'd be happy to help with that"). Get straight to the substance.

Ground every answer strictly in the facts you've actually been given -- the job data, score \
breakdown, proposal/prototype text, or queue digest in this conversation. Never invent a detail \
that isn't there; say so plainly if something wasn't provided instead of guessing.

Default to short, concrete replies -- a few sentences, not an essay -- unless Alejandro \
explicitly asks for more detail or for you to rewrite a longer piece of text (like a full \
proposal). When he asks for a rewrite, give him the complete revised text, not a diff or a \
description of the change.
"""


class ChatAgent:
    """Stateless conversational agent: one call in, one plain-text reply out."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        """Create a Chat Agent.

        Args:
            llm: Chat model to use. Defaults to the shared client from `get_llm()`.
        """
        self._llm = llm or get_llm()

    async def reply(
        self, history: list[dict[str, str]], user_message: str, *, system_prompt: str
    ) -> str:
        """Generate the next conversational reply.

        Args:
            history: Prior turns in this conversation, oldest first, as
                `{"role": "user" | "assistant", "content": str}` dicts. Not
                mutated -- the caller is responsible for appending both this
                call's user message and its returned reply.
            user_message: The newest message from Alejandro.
            system_prompt: The full system prompt for this conversation
                (persona plus whatever job/queue context applies) -- see
                `CHAT_SYSTEM_PROMPT`, `build_job_context_message`, and
                `build_queue_digest`.

        Returns:
            The assistant's reply text, stripped of surrounding whitespace.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_message},
        ]
        chat_llm = self._llm.bind(
            max_tokens=_MAX_OUTPUT_TOKENS,
            thinking={"type": "disabled"},
            cache_control={"type": "ephemeral"},
        )
        response = await ainvoke_with_retry(chat_llm, messages)
        return str(response.content).strip()


def build_job_context_message(
    job: JobPost,
    score: JobScore,
    proposal: GeneratedProposal | None,
    prototype: GeneratedPrototype | None,
) -> str:
    """Render a job's full detail as plain text, for a per-job conversation's system prompt.

    Deliberately omits the prototype's generated source files (demo script,
    requirements, etc.) -- those can run long, and discussing the demo's code
    itself isn't this agent's job. Only the prototype's category/README are
    included, as a pointer that one exists.
    """
    proposals_count = job.proposals_count if job.proposals_count is not None else "not shown"
    payment_verified = "yes" if job.payment_verified else "not verified"
    skills_required = ", ".join(job.skills_required) if job.skills_required else "none listed"
    lines = [
        f"Job: {job.title}",
        f"URL: {job.url}",
        f"Posted: {_format_age(job.posted_at)}",
        f"Proposals so far: {proposals_count}",
        f"Client hires: {job.client_hires}",
        f"Payment verified: {payment_verified}",
        f"Budget: {job.budget}",
        f"Skills required: {skills_required}",
        "",
        f"Description: {job.description}",
        "",
        f"Score: {score.total_score:.0f}/100 ({score.gig_category.value}) -- "
        f"freshness {score.freshness_score:.0f}, proposals {score.proposal_score:.0f}, "
        f"client {score.client_score:.0f}, skills {score.skill_score:.0f}, "
        f"budget {score.budget_score:.0f}",
        f"Recommendation: {score.recommendation.value}",
        f"Red flags: {', '.join(score.red_flags) if score.red_flags else 'none'}",
    ]
    if proposal is not None:
        lines += ["", f"Current proposal draft ({proposal.category}):", proposal.full_text]
    if prototype is not None:
        lines += [
            "",
            f"A demo prototype was also generated ({prototype.category}):",
            prototype.readme_md,
        ]
    return "\n".join(lines)


def build_queue_digest(jobs: list[Job], *, limit: int = _MAX_QUEUE_DIGEST_JOBS) -> str:
    """Render a compact, one-line-per-job summary of the queue for the advisor's system prompt.

    Capped to the `limit` most recently seen jobs (the order `list_jobs`
    already returns them in) so a large queue doesn't blow up the advisor
    session's static context.
    """
    if not jobs:
        return "The job queue is currently empty."
    shown = jobs[:limit]
    lines = [f"Job queue ({len(shown)} of {len(jobs)} shown, most recent first):"]
    lines += [
        f"- [{job.score:.0f}] {job.category} | {job.status.value} | {job.title} | {job.url}"
        for job in shown
    ]
    return "\n".join(lines)


def build_advisor_profile_summary(profile: Profile) -> str:
    """Render a short plain-text profile summary for the advisor's system prompt."""
    rate = profile.freelancer.rate_usd_hr
    budget_min = profile.scoring.target_budget_min
    budget_max = profile.scoring.target_budget_max
    return (
        f"Alejandro's profile: {profile.freelancer.title}, ${rate:.0f}/hr. "
        f"Primary skills: {', '.join(profile.skills.primary)}. "
        f"Target budget: ${budget_min:.0f}-${budget_max:.0f}."
    )


def _format_age(posted_at: datetime) -> str:
    """Render how long ago a job was posted as a short natural-language phrase."""
    age_minutes = (datetime.now(UTC) - posted_at).total_seconds() / 60
    if age_minutes < 60:
        return f"{max(0, round(age_minutes))} minutes ago"
    age_hours = age_minutes / 60
    if age_hours < 24:
        return f"{round(age_hours)} hours ago"
    return f"{round(age_hours / 24)} days ago"
