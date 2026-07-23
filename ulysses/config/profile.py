"""Typed loader for `profile.yaml` — the freelancer's skills, repos, and scoring preferences."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

__all__ = [
    "AlertsConfig",
    "FreelancerProfile",
    "Profile",
    "ProfileKeyError",
    "RepoConfig",
    "ScoringConfig",
    "SkillsConfig",
    "load_profile",
    "save_profile",
    "set_profile_value",
]

DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent / "profile.yaml"
_PROFILE_HEADER = (
    "# Your Upwork profile configuration — Ulysses uses this for scoring and proposals.\n"
)


class ProfileKeyError(Exception):
    """Raised when a `config set` key path doesn't exist or isn't a settable field."""


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


def save_profile(profile: Profile, path: Path = DEFAULT_PROFILE_PATH) -> None:
    """Write a `Profile` back to a YAML file, keeping the header comment.

    This is a full rewrite, not an in-place patch, so any manual comments or
    formatting elsewhere in an existing `profile.yaml` won't survive a save.
    """
    body = yaml.safe_dump(
        profile.model_dump(mode="json"),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    path.write_text(_PROFILE_HEADER + "\n" + body, encoding="utf-8")


def set_profile_value(profile: Profile, dotted_key: str, raw_value: str) -> Profile:
    """Return a copy of `profile` with one dotted-path field updated.

    Scalar fields (str/int/float/bool) are type-coerced from `raw_value`.
    String-list fields (e.g. `skills.primary`) are set from a comma-separated
    string. Nested list-of-model fields (like `repos`) aren't settable this
    way -- edit `profile.yaml` directly for those.

    Args:
        profile: The profile to update. Not mutated.
        dotted_key: A dotted path like `"freelancer.rate_usd_hr"` or
            `"scoring.min_score_to_notify"`.
        raw_value: The new value, as a raw string from the CLI.

    Returns:
        A new, validated `Profile` with the field updated.

    Raises:
        ProfileKeyError: If the key path doesn't exist or isn't settable.
    """
    parts = dotted_key.split(".")
    data = profile.model_dump(mode="python")
    cursor = data
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            raise ProfileKeyError(f"Unknown config section: {dotted_key!r}")
        cursor = cursor[part]

    leaf_key = parts[-1]
    if leaf_key not in cursor:
        raise ProfileKeyError(f"Unknown config key: {dotted_key!r}")

    cursor[leaf_key] = _coerce_value(cursor[leaf_key], raw_value, dotted_key)
    return Profile.model_validate(data)


def _coerce_value(current_value: object, raw_value: str, dotted_key: str) -> object:
    if isinstance(current_value, bool):
        return raw_value.strip().lower() in ("true", "1", "yes")
    if isinstance(current_value, int):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if isinstance(current_value, str):
        return raw_value
    if isinstance(current_value, list) and all(isinstance(item, str) for item in current_value):
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    raise ProfileKeyError(f"{dotted_key!r} isn't a settable scalar or string-list field")
