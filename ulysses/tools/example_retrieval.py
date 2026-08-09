"""Semantic retrieval of curated "gold standard" example proposals.

A lightweight, no-fine-tuning way to anchor the Proposal Agent's style: a
handful of hand-written example proposals live in `config/example_proposals.yaml`;
`find_best_matching_example` picks the single closest one (by embedding cosine
similarity, via `tools.llm.aembed_texts`) to the real job being drafted for, so
`ProposalAgent.generate` can show the LLM a genuine example of the desired
output shape before asking it to write the real one.

No vector store or numpy: the example corpus is expected to stay small (a
handful of hand-curated entries), so a linear scan with pure-Python cosine
similarity is simpler and just as fast as any index would be at this scale.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ulysses.models import JobPost
from ulysses.tools.llm import aembed_texts

__all__ = ["ExampleProposal", "find_best_matching_example", "load_example_proposals"]

DEFAULT_EXAMPLE_PROPOSALS_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "example_proposals.yaml"
)


class ExampleProposal(BaseModel):
    """A curated example: a stand-in job and its ideal proposal, field-by-field.

    Decomposed into the same fields `_ProposalLLMOutput` requires (rather than
    one flat proposal string) so the few-shot "assistant" turn built from this
    example mirrors the exact structured-output shape the LLM is being asked
    to produce.
    """

    job_title: str
    job_description: str
    skills: list[str] = Field(default_factory=list)
    category: str
    proof_repo: str
    proof_repo_url: str
    hook: str
    plan_bullet_1: str
    plan_bullet_2: str
    plan_bullet_3: str
    close: str
    milestones: list[str] = Field(default_factory=list)


def load_example_proposals(
    path: Path = DEFAULT_EXAMPLE_PROPOSALS_PATH,
) -> list[ExampleProposal]:
    """Load curated example proposals from `example_proposals.yaml`.

    Returns an empty list if the file doesn't exist -- examples are optional;
    `ProposalAgent.generate` falls back to drafting with no few-shot example
    when none are available.
    """
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [ExampleProposal.model_validate(entry) for entry in raw]


async def find_best_matching_example(
    job: JobPost, examples: list[ExampleProposal]
) -> ExampleProposal | None:
    """Return the example whose stand-in job is semantically closest to `job`.

    Embeds the real job's title/description/skills and every example's
    title/description in a single batched call, then picks the highest
    cosine-similarity match. Returns `None` for an empty corpus -- callers
    should just skip the few-shot example in that case, not treat it as an
    error.
    """
    if not examples:
        return None

    query_text = f"{job.title}\n{job.description}\nSkills: {', '.join(job.skills_required)}"
    example_texts = [f"{example.job_title}\n{example.job_description}" for example in examples]

    vectors = await aembed_texts([query_text, *example_texts])
    query_vector, example_vectors = vectors[0], vectors[1:]

    similarities = [_cosine_similarity(query_vector, vector) for vector in example_vectors]
    best_index = max(range(len(examples)), key=lambda index: similarities[index])
    return examples[best_index]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two vectors, or 0.0 if either is zero."""
    dot_product = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)
