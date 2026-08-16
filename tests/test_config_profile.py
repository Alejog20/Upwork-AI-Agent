"""Tests for `ulysses.config.profile`'s save/set helpers used by `ulysses config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from ulysses.config.profile import (
    Profile,
    ProfileKeyError,
    load_profile,
    save_profile,
    set_profile_value,
)


class TestSetProfileValue:
    def test_sets_a_top_level_scalar_float_field(self, profile: Profile) -> None:
        updated = set_profile_value(profile, "freelancer.rate_usd_hr", "40")
        assert updated.freelancer.rate_usd_hr == 40.0
        assert profile.freelancer.rate_usd_hr != 40.0  # original untouched

    def test_sets_a_string_field(self, profile: Profile) -> None:
        updated = set_profile_value(profile, "freelancer.name", "New Name")
        assert updated.freelancer.name == "New Name"

    def test_sets_a_nested_scoring_field(self, profile: Profile) -> None:
        updated = set_profile_value(profile, "scoring.min_score_to_notify", "60")
        assert updated.scoring.min_score_to_notify == 60.0

    def test_sets_an_int_field(self, profile: Profile) -> None:
        updated = set_profile_value(profile, "scoring.skip_if_proposals_above", "10")
        assert updated.scoring.skip_if_proposals_above == 10
        assert isinstance(updated.scoring.skip_if_proposals_above, int)

    def test_sets_a_string_list_field_from_comma_separated_value(self, profile: Profile) -> None:
        updated = set_profile_value(profile, "skills.primary", "python, rust,  go")
        assert updated.skills.primary == ["python", "rust", "go"]

    def test_unknown_top_level_section_raises(self, profile: Profile) -> None:
        with pytest.raises(ProfileKeyError):
            set_profile_value(profile, "nonexistent.field", "x")

    def test_unknown_leaf_key_raises(self, profile: Profile) -> None:
        with pytest.raises(ProfileKeyError):
            set_profile_value(profile, "freelancer.nonexistent", "x")

    def test_non_scalar_non_stringlist_field_raises(self, profile: Profile) -> None:
        with pytest.raises(ProfileKeyError):
            set_profile_value(profile, "repos", "x")


class TestScoringWeights:
    def test_defaults_match_the_architecture_doc_weighting(self, profile: Profile) -> None:
        weights = profile.scoring.weights
        assert weights.freshness_under_15_min == 30.0
        assert weights.proposals_under_5 == 25.0
        assert weights.client_no_hires == 20.0
        assert weights.skill_match_max_points == 15.0
        assert weights.budget_match_max_points == 10.0

    def test_settable_via_three_level_dotted_key(self, profile: Profile) -> None:
        updated = set_profile_value(profile, "scoring.weights.freshness_under_15_min", "40")
        assert updated.scoring.weights.freshness_under_15_min == 40.0
        assert profile.scoring.weights.freshness_under_15_min == 30.0  # original untouched

    def test_bundled_profile_yaml_loads_with_default_weights(self) -> None:
        # The shipped profile.yaml doesn't declare a `weights:` block -- confirms
        # ScoringWeights' Pydantic defaults populate it without a config change.
        loaded = load_profile()
        assert loaded.scoring.weights.freshness_under_15_min == 30.0


class TestSaveAndReloadProfile:
    def test_round_trips_through_disk(self, profile: Profile, tmp_path: Path) -> None:
        path = tmp_path / "profile.yaml"
        updated = set_profile_value(profile, "freelancer.rate_usd_hr", "55")

        save_profile(updated, path)
        reloaded = load_profile(path)

        assert reloaded.freelancer.rate_usd_hr == 55.0
        assert reloaded.freelancer.name == profile.freelancer.name

    def test_keeps_non_ascii_characters_readable_on_disk(
        self, profile: Profile, tmp_path: Path
    ) -> None:
        """Regression test: yaml.safe_dump defaults to escaping non-ASCII as \\uXXXX,
        which round-trips fine through load_profile but makes the file unreadable/
        unmergeable for a human. save_profile must pass allow_unicode=True."""
        updated = set_profile_value(profile, "freelancer.name", "Alejandro García")
        path = tmp_path / "profile.yaml"

        save_profile(updated, path)

        raw_text = path.read_text(encoding="utf-8")
        assert "\\u" not in raw_text
        assert "García" in raw_text

    def test_keeps_the_header_comment(self, profile: Profile, tmp_path: Path) -> None:
        path = tmp_path / "profile.yaml"
        save_profile(profile, path)
        assert path.read_text(encoding="utf-8").startswith("# Your Upwork profile configuration")
