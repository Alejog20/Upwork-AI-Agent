"""Domain models shared by every layer of Ulysses.

These Pydantic models represent the business objects that flow through the
LangGraph pipeline (`JobPost`, `JobScore`). They are intentionally decoupled
from the SQLModel persistence tables in `ulysses.tools.db`, which store a
flattened subset of this data for querying.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = [
    "BudgetRange",
    "BudgetType",
    "GeneratedProposal",
    "GigCategory",
    "JobPost",
    "JobScore",
    "Recommendation",
    "RepoMatch",
]


class BudgetType(StrEnum):
    """Whether a job's budget is a fixed price or an hourly rate."""

    FIXED = "fixed"
    HOURLY = "hourly"
    UNKNOWN = "unknown"


class BudgetRange(BaseModel):
    """A job's advertised budget, normalized to a min/max range."""

    type: BudgetType = BudgetType.UNKNOWN
    min_amount: float | None = None
    max_amount: float | None = None
    currency: str = "USD"

    @property
    def midpoint(self) -> float | None:
        """Return the midpoint of the range, or the single known bound."""
        if self.min_amount is not None and self.max_amount is not None:
            return (self.min_amount + self.max_amount) / 2
        return self.min_amount if self.min_amount is not None else self.max_amount

    def __str__(self) -> str:
        if self.type is BudgetType.UNKNOWN or self.midpoint is None:
            return "Budget not listed"
        if self.type is BudgetType.HOURLY:
            return f"${self.midpoint:.0f}/hr"
        if (
            self.min_amount is not None
            and self.max_amount is not None
            and self.min_amount != self.max_amount
        ):
            return f"${self.min_amount:.0f}-${self.max_amount:.0f} fixed"
        return f"${self.midpoint:.0f} fixed"


class JobPost(BaseModel):
    """A structured Upwork job posting extracted from an email or RSS entry."""

    id: str
    title: str
    description: str
    budget: BudgetRange = Field(default_factory=BudgetRange)
    skills_required: list[str] = Field(default_factory=list)
    client_hires: int = 0
    client_rating: float | None = None
    payment_verified: bool = False
    proposals_count: int | None = None
    posted_at: datetime
    url: str


class GigCategory(StrEnum):
    """Tiering bucket assigned to a job by the Scorer Agent."""

    TIER_1 = "tier1"
    TIER_2 = "tier2"
    TIER_3 = "tier3"


class Recommendation(StrEnum):
    """Scorer Agent's recommended action for a job."""

    APPLY_NOW = "apply_now"
    REVIEW = "review"
    SKIP = "skip"


class RepoMatch(BaseModel):
    """A GitHub repo ranked by relevance to a job's required skills."""

    repo_name: str
    url: str
    relevance_score: float


class JobScore(BaseModel):
    """Full scoring breakdown produced by the Scorer Agent."""

    total_score: float
    freshness_score: float
    proposal_score: float
    client_score: float
    skill_score: float
    budget_score: float
    matched_repos: list[RepoMatch] = Field(default_factory=list)
    gig_category: GigCategory
    red_flags: list[str] = Field(default_factory=list)
    recommendation: Recommendation


class GeneratedProposal(BaseModel):
    """A generated Upwork proposal draft, ready to send or copy."""

    job_id: str
    category: str
    hook: str
    plan_bullets: list[str]
    proof_repo: str
    proof_repo_url: str
    timeline: str
    bid_usd: float
    full_text: str
