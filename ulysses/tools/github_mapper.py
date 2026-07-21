"""Maps a job's required skills against the freelancer's GitHub repos."""

from __future__ import annotations

from ulysses.config.profile import RepoConfig
from ulysses.models import RepoMatch

__all__ = ["rank_matching_repos"]


def rank_matching_repos(
    skills_required: list[str],
    repos: list[RepoConfig],
    *,
    top_n: int = 3,
) -> list[RepoMatch]:
    """Rank repos by tag overlap with a job's required skills.

    Uses Jaccard similarity (intersection over union) between the job's
    lowercased skill list and each repo's lowercased tag list.

    Args:
        skills_required: Skills extracted from the job posting.
        repos: The freelancer's repo configs, from `profile.yaml`.
        top_n: Maximum number of matches to return.

    Returns:
        Up to `top_n` `RepoMatch` entries, sorted by descending relevance.
        `relevance_score` is a 0-1 Jaccard similarity; repos with no overlap
        at all are still included (with a score of 0.0) so callers can see
        the full ranking.
    """
    normalized_skills = {skill.strip().lower() for skill in skills_required}

    matches: list[RepoMatch] = []
    for repo in repos:
        repo_tags = {tag.strip().lower() for tag in repo.tags}
        union = repo_tags | normalized_skills
        relevance = len(repo_tags & normalized_skills) / len(union) if union else 0.0
        matches.append(
            RepoMatch(repo_name=repo.name, url=repo.url, relevance_score=round(relevance, 4))
        )

    matches.sort(key=lambda match: match.relevance_score, reverse=True)
    return matches[:top_n]
