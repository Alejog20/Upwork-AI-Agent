"""macOS menu bar app for Ulysses, built with `rumps`.

Runs the exact same agent loop as `ulysses start` (via `ulysses.cli.main.run_forever`)
in a background thread, since rumps needs the main thread for its native Cocoa
event loop. Both surfaces share the same SQLite database for state.

Pause/Resume and Quit communicate with that background loop via plain
`threading.Event`s rather than `asyncio.Event`s or direct method calls --
see `ScoutAgent.run_forever`'s docstring for why (rumps callbacks run on a
different thread than the agent loop's event loop).

Menu-bar UI mutations (status title, item labels) only ever happen inside
`@rumps.timer`/menu-item callbacks, which rumps guarantees run on the main
thread -- the background thread only ever signals via `threading.Event`s or
`rumps.notification()`, never by touching menu/window state directly.
"""

from __future__ import annotations

import asyncio
import subprocess
import threading
from pathlib import Path

import rumps
from loguru import logger

from ulysses.cli.main import run_forever
from ulysses.config.profile import load_profile
from ulysses.config.settings import get_settings
from ulysses.models import JobPost, JobScore
from ulysses.tools.db import sync_read_menubar_stats

__all__ = ["UlyssesMenuBarApp", "format_instant_alert", "main"]

_STATS_REFRESH_SECONDS = 15


def format_instant_alert(job: JobPost, score: JobScore) -> dict[str, str]:
    """Build the title/subtitle/message for a macOS instant-alert notification."""
    best_repo = score.matched_repos[0].repo_name if score.matched_repos else "no repo match"
    return {
        "title": job.title,
        "subtitle": f"Score: {score.total_score:.0f}/100 | {job.budget}",
        "message": best_repo,
    }


class UlyssesMenuBarApp(rumps.App):
    """The Ulysses menu bar app: status, counts, pause/resume, queue, preferences, quit."""

    def __init__(self) -> None:
        """Build the menu and start the background agent loop."""
        super().__init__("Ulysses 🟢", quit_button=None)

        self._stop_event = threading.Event()
        self._paused_event = threading.Event()
        self._crashed_event = threading.Event()

        self._jobs_today_item = rumps.MenuItem("Jobs Today: 0")
        self._proposals_item = rumps.MenuItem("Proposals Drafted: 0")
        self._prototypes_item = rumps.MenuItem("Prototypes Built: 0")
        self._pause_resume_item = rumps.MenuItem("Pause", callback=self.toggle_pause)

        self.menu = [
            self._jobs_today_item,
            self._proposals_item,
            self._prototypes_item,
            None,
            rumps.MenuItem("Open Job Queue", callback=self.open_job_queue),
            self._pause_resume_item,
            None,
            rumps.MenuItem("Preferences...", callback=self.open_preferences),
            rumps.MenuItem("Quit Ulysses", callback=self.quit_ulysses),
        ]

        self._agent_thread = threading.Thread(target=self._run_agent_loop, daemon=True)
        self._agent_thread.start()

    def _run_agent_loop(self) -> None:
        settings = get_settings()
        profile = load_profile(settings.profile_path)
        try:
            asyncio.run(
                run_forever(
                    settings,
                    profile,
                    stop_event=self._stop_event,
                    paused_event=self._paused_event,
                    on_instant_alert=self._post_instant_alert,
                )
            )
        except Exception:
            logger.exception("Menu bar agent loop crashed")
            self._crashed_event.set()

    def _post_instant_alert(self, job: JobPost, score: JobScore) -> None:
        """Post a native macOS notification for an instant-alert-worthy job.

        Called from the background agent thread. `rumps.notification()`
        posts to `NSUserNotificationCenter`, which is safe to call off the
        main thread in practice (unlike mutating menu/window state) -- if
        that ever proves flaky on a given macOS version, marshal this call
        onto the main thread instead.
        """
        try:
            rumps.notification(**format_instant_alert(job, score))
        except Exception:
            logger.exception("Failed to post macOS notification")

    @rumps.timer(_STATS_REFRESH_SECONDS)
    def refresh(self, _sender: rumps.Timer) -> None:
        """Refresh job/proposal/prototype counts and the crashed/running indicator."""
        if self._crashed_event.is_set():
            self.title = "Ulysses 🔴"
            return

        settings = get_settings()
        stats = sync_read_menubar_stats(settings.db_path)
        self._jobs_today_item.title = f"Jobs Today: {stats['jobs_today']}"
        self._proposals_item.title = f"Proposals Drafted: {stats['proposals_drafted']}"
        self._prototypes_item.title = f"Prototypes Built: {stats['prototypes_built']}"

    def toggle_pause(self, sender: rumps.MenuItem) -> None:
        """Pause or resume the scout polling loop (Telegram/DB stay live either way)."""
        if self._paused_event.is_set():
            self._paused_event.clear()
            sender.title = "Pause"
            self.title = "Ulysses 🟢"
        else:
            self._paused_event.set()
            sender.title = "Resume"
            self.title = "Ulysses 🟡"

    def open_job_queue(self, _sender: rumps.MenuItem) -> None:
        """Open a new Terminal window running `ulysses queue`."""
        script = 'tell application "Terminal" to do script "uv run ulysses queue"'
        subprocess.run(["osascript", "-e", script], check=False)

    def open_preferences(self, _sender: rumps.MenuItem) -> None:
        """Open `.env` in the default editor."""
        subprocess.run(["open", str(Path(".env"))], check=False)

    def quit_ulysses(self, _sender: rumps.MenuItem) -> None:
        """Signal the agent loop to stop and quit the app."""
        self._stop_event.set()
        rumps.quit_application()


def main() -> None:
    """Entry point: `uv run python -m ulysses.app.menubar`."""
    UlyssesMenuBarApp().run()


if __name__ == "__main__":
    main()
