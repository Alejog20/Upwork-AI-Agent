"""Typed loader for `profile.yaml` — the freelancer's skills, repos, and scoring preferences."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

__all__ = [
    "AlertsConfig",
    "FreelancerProfile",
    "Profile",
    "RepoConfig",
    "ScoringConfig",
    "SkillsConfig",
    "load_profile",
]

DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent / "profile.yaml"


class FreelancerProfile(BaseModel):
    """Identity and headline info used in proposals and prototype READMEs."""

    name: str
    title: str
    github: str
    rate_usd_hr: float


class SkillsConfig(BaseModel):
    """The freelancer's declared skill tags, used for job/skill matching."""

    primary: list[str] = Field(default_factory=list)
    secondary: list[str] = Field(default_factory=list)

    @property
    def all(self) -> list[str]:
        """Return primary and secondary skills combined, in priority order."""
        return [*self.primary, *self.secondary]


class RepoConfig(BaseModel):
    """A GitHub repo the freelancer can point to as proof of work."""

    name: str
    url: str
    tags: list[str] = Field(default_factory=list)


class ScoringConfig(BaseModel):
    """Tunable thresholds that drive the Scorer Agent and alerting behavior."""

    target_budget_min: float
    target_budget_max: float
    min_score_to_notify: float
    instant_alert_threshold: float
    skip_if_proposals_above: int
    skip_if_posted_hours_ago: float


class AlertsConfig(BaseModel):
    """Telegram alerting configuration."""

    telegram_chat_id: str
    batch_interval_minutes: int


class Profile(BaseModel):
    """The freelancer's full profile: identity, skills, repos, and scoring preferences."""

    freelancer: FreelancerProfile
    skills: SkillsConfig
    repos: list[RepoConfig] = Field(default_factory=list)
    scoring: ScoringConfig
    alerts: AlertsConfig


def load_profile(path: Path = DEFAULT_PROFILE_PATH) -> Profile:
    """Load and validate `profile.yaml` from disk.

    Args:
        path: Path to the profile YAML file. Defaults to the bundled
            `ulysses/config/profile.yaml`.

    Returns:
        The parsed and validated `Profile`.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Profile.model_validate(raw)
