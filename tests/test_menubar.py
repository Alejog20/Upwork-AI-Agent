"""Tests for `ulysses.app.menubar`'s testable logic.

The real `rumps.App` (`UlyssesMenuBarApp`) is never instantiated here -- its
`__init__` starts a real background thread that connects to real IMAP/
Telegram via `get_settings()`, and `rumps.App` itself needs a real Cocoa/
PyObjC runtime context that doesn't behave predictably under pytest.
Instead, these tests call the class's methods directly against lightweight
stand-in objects exposing only the attributes each method reads, and test
the extracted pure `format_instant_alert` function on its own. Manual
verification on a real Mac is still needed for the actual menu bar UI
behavior -- see the module docstring in `ulysses/app/menubar.py`.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

# `rumps` (and its pyobjc dependencies) only installs on macOS -- see the
# `sys_platform == 'darwin'` marker in pyproject.toml. Skip this whole module
# rather than crash on import when running on CI's Linux runner.
pytest.importorskip("rumps")

from ulysses.agents.scorer import score_job
from ulysses.app.menubar import _ICON_FRAMES, UlyssesMenuBarApp, format_instant_alert
from ulysses.config.profile import Profile
from ulysses.models import JobPost


class TestFormatInstantAlert:
    def test_title_is_job_title(self, fresh_job: JobPost, profile: Profile) -> None:
        score = score_job(fresh_job, profile)
        payload = format_instant_alert(fresh_job, score)
        assert payload["title"] == fresh_job.title

    def test_subtitle_includes_score_and_budget(self, fresh_job: JobPost, profile: Profile) -> None:
        score = score_job(fresh_job, profile)
        payload = format_instant_alert(fresh_job, score)
        assert f"{score.total_score:.0f}/100" in payload["subtitle"]
        assert str(fresh_job.budget) in payload["subtitle"]

    def test_message_is_best_matched_repo(self, fresh_job: JobPost, profile: Profile) -> None:
        score = score_job(fresh_job, profile)
        payload = format_instant_alert(fresh_job, score)
        assert payload["message"] == score.matched_repos[0].repo_name

    def test_message_falls_back_when_no_repo_matched(
        self, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        score_no_match = score.model_copy(update={"matched_repos": []})
        payload = format_instant_alert(fresh_job, score_no_match)
        assert payload["message"] == "no repo match"


class _FakeApp:
    """Minimal stand-in exposing only what `UlyssesMenuBarApp` methods touch."""

    def __init__(self) -> None:
        self._paused_event = threading.Event()
        self._stop_event = threading.Event()
        self._crashed_event = threading.Event()
        self._status_item = MagicMock(title="Status: Running")
        self._jobs_today_item = MagicMock()
        self._proposals_item = MagicMock()
        self._prototypes_item = MagicMock()
        self._frame_index = 0
        self.icon = None


class TestTogglePause:
    def test_pausing_sets_event_and_updates_titles(self) -> None:
        fake = _FakeApp()
        sender = MagicMock(title="Pause")

        UlyssesMenuBarApp.toggle_pause(fake, sender)

        assert fake._paused_event.is_set()
        assert sender.title == "Resume"
        assert fake._status_item.title == "Status: Paused"

    def test_resuming_clears_event_and_updates_titles(self) -> None:
        fake = _FakeApp()
        fake._paused_event.set()
        sender = MagicMock(title="Resume")

        UlyssesMenuBarApp.toggle_pause(fake, sender)

        assert not fake._paused_event.is_set()
        assert sender.title == "Pause"
        assert fake._status_item.title == "Status: Running"


class TestQuitUlysses:
    def test_sets_stop_event_and_quits(self, mocker: MockerFixture) -> None:
        fake = _FakeApp()
        quit_mock = mocker.patch("ulysses.app.menubar.rumps.quit_application")

        UlyssesMenuBarApp.quit_ulysses(fake, MagicMock())

        assert fake._stop_event.is_set()
        quit_mock.assert_called_once()


class TestOpenActions:
    def test_open_job_queue_calls_osascript(self, mocker: MockerFixture) -> None:
        run_mock = mocker.patch("ulysses.app.menubar.subprocess.run")
        UlyssesMenuBarApp.open_job_queue(_FakeApp(), MagicMock())
        run_mock.assert_called_once()
        assert run_mock.call_args.args[0][0] == "osascript"

    def test_open_preferences_opens_env_file(self, mocker: MockerFixture) -> None:
        run_mock = mocker.patch("ulysses.app.menubar.subprocess.run")
        UlyssesMenuBarApp.open_preferences(_FakeApp(), MagicMock())
        run_mock.assert_called_once()
        args = run_mock.call_args.args[0]
        assert args[0] == "open"
        assert ".env" in args[1]


class TestPostInstantAlert:
    def test_calls_rumps_notification_with_formatted_payload(
        self, fresh_job: JobPost, profile: Profile, mocker: MockerFixture
    ) -> None:
        score = score_job(fresh_job, profile)
        notif_mock = mocker.patch("ulysses.app.menubar.rumps.notification")

        UlyssesMenuBarApp._post_instant_alert(_FakeApp(), fresh_job, score)

        notif_mock.assert_called_once_with(**format_instant_alert(fresh_job, score))

    def test_swallows_notification_failures(
        self, fresh_job: JobPost, profile: Profile, mocker: MockerFixture
    ) -> None:
        score = score_job(fresh_job, profile)
        mocker.patch("ulysses.app.menubar.rumps.notification", side_effect=RuntimeError("boom"))

        UlyssesMenuBarApp._post_instant_alert(_FakeApp(), fresh_job, score)  # must not raise


class TestAnimateIcon:
    def test_advances_frame_index_and_sets_icon(self) -> None:
        fake = _FakeApp()

        UlyssesMenuBarApp._animate_icon(fake, MagicMock())

        assert fake._frame_index == 1
        assert fake.icon == str(_ICON_FRAMES[1])

    def test_wraps_around_after_last_frame(self) -> None:
        fake = _FakeApp()
        fake._frame_index = len(_ICON_FRAMES) - 1

        UlyssesMenuBarApp._animate_icon(fake, MagicMock())

        assert fake._frame_index == 0
        assert fake.icon == str(_ICON_FRAMES[0])


class TestRefresh:
    def test_sets_crashed_status_when_crashed(self) -> None:
        fake = _FakeApp()
        fake._crashed_event.set()

        UlyssesMenuBarApp.refresh(fake, MagicMock())

        assert fake._status_item.title == "Status: Crashed"

    def test_updates_stat_items_from_db(self, mocker: MockerFixture) -> None:
        fake = _FakeApp()
        mocker.patch(
            "ulysses.app.menubar.sync_read_menubar_stats",
            return_value={"jobs_today": 3, "proposals_drafted": 2, "prototypes_built": 1},
        )
        mocker.patch("ulysses.app.menubar.get_settings")

        UlyssesMenuBarApp.refresh(fake, MagicMock())

        assert fake._jobs_today_item.title == "Jobs Today: 3"
        assert fake._proposals_item.title == "Proposals Drafted: 2"
        assert fake._prototypes_item.title == "Prototypes Built: 1"
