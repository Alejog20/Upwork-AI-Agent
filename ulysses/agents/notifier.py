"""Notifier Agent — formats scored jobs and sends them to Telegram with action buttons.

Alerting is threshold-based:
  - score >= instant_alert_threshold: sent immediately.
  - min_score_to_notify <= score < instant_alert_threshold: queued and sent
    in a batch every `batch_interval_minutes`.
  - score < min_score_to_notify: not sent at all (the job is still persisted
    by the Scout Agent for later reference).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from loguru import logger
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import NetworkError, RetryAfter
from telegram.ext import CallbackQueryHandler, ContextTypes
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ulysses.config.profile import ScoringConfig
from ulysses.models import GeneratedPrototype, JobPost, JobScore, Recommendation
from ulysses.tools.db import JobStatus, UlyssesDB

__all__ = ["InstantAlertHook", "NotifierAgent", "format_job_message"]

_telegram_send_retry = retry(
    retry=retry_if_exception_type((NetworkError, RetryAfter)),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)

_ACTION_LABELS: dict[str, str] = {
    "draft": "📝 Draft Proposal",
    "build": "🛠 Build Demo",
    "skip": "⏭ Skip",
    "archive": "📁 Archive",
}

_ACTION_STATUS: dict[str, JobStatus] = {
    "skip": JobStatus.SKIPPED,
    "archive": JobStatus.ARCHIVED,
}

DraftHandler = Callable[[str], Awaitable[None]]
BuildHandler = Callable[[str], Awaitable[None]]
InstantAlertHook = Callable[[JobPost, JobScore], None]


class NotifierAgent:
    """Sends scored jobs to a single Telegram chat and handles button presses."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        db: UlyssesDB,
        on_draft_requested: DraftHandler | None = None,
        on_build_requested: BuildHandler | None = None,
        on_instant_alert: InstantAlertHook | None = None,
    ) -> None:
        """Create a Notifier Agent bound to one Telegram chat.

        Args:
            bot_token: The Telegram bot token from BotFather.
            chat_id: The only chat ID this bot will ever send to or accept
                callbacks from — validated in `handle_callback`.
            db: Persistence layer, used to record notification and user-action state.
            on_draft_requested: Async callback invoked with a job id when the
                Draft or Regenerate button is pressed. Can also be set later
                via `set_draft_handler` to avoid constructor ordering issues.
            on_build_requested: Async callback invoked with a job id when the
                Build Demo button is pressed. Can also be set later via
                `set_build_handler`.
            on_instant_alert: Sync callback invoked with `(job, score)` for
                jobs that clear the instant-alert threshold, right after the
                Telegram message is sent -- used by the menu bar app to also
                post a native macOS notification. Sync (not async) because
                `rumps.notification()` is itself a plain sync call. Never
                fires for batched or silently-archived jobs.
        """
        self._bot = Bot(token=bot_token)
        self._chat_id = str(chat_id)
        self._db = db
        self._batch_queue: list[tuple[JobPost, JobScore]] = []
        self._on_draft_requested = on_draft_requested
        self._on_build_requested = on_build_requested
        self._on_instant_alert = on_instant_alert

    def set_draft_handler(self, handler: DraftHandler) -> None:
        """Register the callback invoked when the Draft or Regenerate button is pressed."""
        self._on_draft_requested = handler

    def set_build_handler(self, handler: BuildHandler) -> None:
        """Register the callback invoked when the Build Demo button is pressed."""
        self._on_build_requested = handler

    def set_instant_alert_hook(self, hook: InstantAlertHook) -> None:
        """Register the callback invoked when a job clears the instant-alert threshold."""
        self._on_instant_alert = hook

    @_telegram_send_retry
    async def _send_message(self, **kwargs: Any) -> None:
        """Send a Telegram message, retrying on transient network errors."""
        await self._bot.send_message(**kwargs)

    @_telegram_send_retry
    async def _send_document(self, **kwargs: Any) -> None:
        """Send a Telegram document, retrying on transient network errors."""
        await self._bot.send_document(**kwargs)

    @property
    def callback_handler(self) -> CallbackQueryHandler:
        """A `python-telegram-bot` handler for the inline action buttons."""
        return CallbackQueryHandler(self.handle_callback)

    async def handle_scored_job(
        self, job: JobPost, score: JobScore, thresholds: ScoringConfig
    ) -> None:
        """Route a scored job to immediate send, batch queue, or silent archive."""
        if score.total_score >= thresholds.instant_alert_threshold:
            await self._send_job_alert(job, score)
            if self._on_instant_alert is not None:
                try:
                    self._on_instant_alert(job, score)
                except Exception:
                    logger.bind(job_id=job.id, agent="notifier").exception(
                        "Instant-alert hook failed (e.g. macOS notification)"
                    )
        elif score.total_score >= thresholds.min_score_to_notify:
            self._batch_queue.append((job, score))
            logger.bind(job_id=job.id, agent="notifier").info("Queued for batched alert")
        else:
            logger.bind(job_id=job.id, agent="notifier").info(
                "Below notify threshold; silent archive only"
            )

    async def flush_batch(self) -> None:
        """Send every job currently queued for batched delivery."""
        if not self._batch_queue:
            return
        pending, self._batch_queue = self._batch_queue, []
        for job, score in pending:
            await self._send_job_alert(job, score)

    async def run_batch_loop(
        self, batch_interval_minutes: int, *, stop_event: threading.Event | None = None
    ) -> None:
        """Flush the batch queue on a fixed interval, until `stop_event` is set.

        `stop_event` is a `threading.Event` for the same cross-thread reason
        as `ScoutAgent.run_forever`'s -- see that docstring.
        """
        interval = timedelta(minutes=batch_interval_minutes).total_seconds()
        while stop_event is None or not stop_event.is_set():
            await asyncio.sleep(interval)
            await self.flush_batch()

    async def _send_job_alert(self, job: JobPost, score: JobScore) -> None:
        text = format_job_message(job, score)
        keyboard = _build_action_keyboard(job.id)
        await self._send_message(
            chat_id=self._chat_id,
            text=text,
            reply_markup=keyboard,
            parse_mode=ParseMode.HTML,
        )
        await self._db.update_status(job.id, JobStatus.NOTIFIED)
        logger.bind(job_id=job.id, agent="notifier").info("Sent job alert to Telegram")

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle an inline button press, validating it came from the configured chat."""
        query = update.callback_query
        if query is None or query.data is None:
            return

        chat_id = str(query.message.chat_id) if query.message else None
        if chat_id != self._chat_id:
            logger.warning("Ignoring callback from unauthorized chat_id={}", chat_id)
            await query.answer("Unauthorized", show_alert=True)
            return

        action, _, job_id = query.data.partition(":")
        await query.answer()

        status = _ACTION_STATUS.get(action)
        if status is not None:
            await self._db.update_status(job_id, status)
            logger.bind(job_id=job_id, agent="notifier").info("User action: {}", action)
        elif action in ("draft", "regenerate"):
            await self._request_draft(job_id, action)
        elif action == "build":
            await self._request_build(job_id)
        elif action == "copy":
            await self._send_plain_copy(job_id)
        else:
            logger.bind(job_id=job_id, agent="notifier").info(
                "User requested: {} (not yet implemented)", action
            )

    async def _request_draft(self, job_id: str, action: str) -> None:
        if self._on_draft_requested is None:
            logger.bind(job_id=job_id, agent="notifier").warning(
                "{} requested but no draft handler is wired", action
            )
            return
        try:
            await self._on_draft_requested(job_id)
        except Exception as exc:
            logger.bind(job_id=job_id, agent="notifier").exception("Draft generation failed")
            try:
                await self.send_error_message(f"Failed to draft a proposal for this job: {exc}")
            except Exception:
                logger.bind(job_id=job_id, agent="notifier").exception(
                    "Also failed to notify about the draft failure"
                )

    async def _request_build(self, job_id: str) -> None:
        if self._on_build_requested is None:
            logger.bind(job_id=job_id, agent="notifier").warning(
                "build requested but no build handler is wired"
            )
            return
        try:
            await self._on_build_requested(job_id)
        except Exception as exc:
            logger.bind(job_id=job_id, agent="notifier").exception("Prototype generation failed")
            try:
                await self.send_error_message(f"Failed to build a demo for this job: {exc}")
            except Exception:
                logger.bind(job_id=job_id, agent="notifier").exception(
                    "Also failed to notify about the build failure"
                )

    async def _send_plain_copy(self, job_id: str) -> None:
        drafts = await self._db.get_proposal_drafts(job_id)
        if not drafts:
            logger.bind(job_id=job_id, agent="notifier").warning(
                "Copy requested but no draft exists"
            )
            return
        await self._send_message(chat_id=self._chat_id, text=drafts[-1].content)

    async def send_proposal_draft(self, job_id: str, text: str) -> None:
        """Send a generated proposal draft to Telegram with copy/regenerate buttons."""
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text="✅ Copy to Clipboard", callback_data=f"copy:{job_id}"
                    ),
                    InlineKeyboardButton(
                        text="🔄 Regenerate", callback_data=f"regenerate:{job_id}"
                    ),
                ]
            ]
        )
        await self._send_message(
            chat_id=self._chat_id,
            text=f"--- PROPOSAL DRAFT ---\n\n{text}\n\n--- END DRAFT ---",
            reply_markup=keyboard,
        )
        await self._db.update_status(job_id, JobStatus.DRAFTED)
        logger.bind(job_id=job_id, agent="notifier").info("Sent proposal draft to Telegram")

    async def send_prototype_zip(
        self, job_id: str, prototype: GeneratedPrototype, zip_bytes: bytes
    ) -> None:
        """Send a generated prototype as a Telegram document, plus a README preview."""
        await self._send_document(
            chat_id=self._chat_id,
            document=zip_bytes,
            filename=prototype.zip_filename,
            caption=f"Demo prototype ({prototype.category})",
        )
        await self._send_message(chat_id=self._chat_id, text=prototype.readme_md)
        await self._db.update_status(job_id, JobStatus.BUILT)
        logger.bind(job_id=job_id, agent="notifier").info("Sent prototype zip to Telegram")

    async def send_error_message(self, text: str) -> None:
        """Surface an agent failure to the user as a plain Telegram message."""
        await self._send_message(chat_id=self._chat_id, text=f"⚠️ {text}")


def format_job_message(job: JobPost, score: JobScore) -> str:
    """Render a scored job into the exact Telegram alert format Ulysses uses."""
    recommendation_label = {
        Recommendation.APPLY_NOW: "APPLY NOW",
        Recommendation.REVIEW: "REVIEW",
        Recommendation.SKIP: "SKIP",
    }[score.recommendation]

    skills_line = ", ".join(job.skills_required) if job.skills_required else "none listed"
    best_repo = score.matched_repos[0].repo_name if score.matched_repos else "none matched"
    red_flags_line = ", ".join(score.red_flags) if score.red_flags else "none"
    proposals_line = f"~{job.proposals_count}" if job.proposals_count is not None else "unknown"
    posted_line = _format_posted_ago(job.posted_at)

    header = f"{score.gig_category.value.upper()} | {recommendation_label}"
    return (
        f"🎯 Score: {score.total_score:.0f}/100 | {header}\n\n"
        f"📌 {job.title}\n"
        f"💰 {job.budget} | ⏱ {posted_line}\n"
        f"👤 Client: {job.client_hires} hires | "
        f"{'✅ Payment verified' if job.payment_verified else '⚠️ Payment not verified'}\n"
        f"📊 Proposals: {proposals_line}\n\n"
        f"🔗 Skills matched: {skills_line}\n"
        f"📁 Best repo match: {best_repo}\n\n"
        f"⚠️ Red flags: {red_flags_line}"
    )


def _format_posted_ago(posted_at: datetime) -> str:
    minutes = max(0, int((datetime.now(UTC) - posted_at).total_seconds() // 60))
    if minutes < 60:
        return f"Posted {minutes} min ago"
    if minutes < 24 * 60:
        return f"Posted {minutes // 60}h ago"
    return f"Posted {minutes // (24 * 60)}d ago"


def _build_action_keyboard(job_id: str) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(text=label, callback_data=f"{action}:{job_id}")
        for action, label in _ACTION_LABELS.items()
    ]
    return InlineKeyboardMarkup([buttons])
