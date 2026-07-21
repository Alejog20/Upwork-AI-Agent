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
from datetime import UTC, datetime, timedelta

from loguru import logger
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import CallbackQueryHandler, ContextTypes

from ulysses.config.profile import ScoringConfig
from ulysses.models import JobPost, JobScore, Recommendation
from ulysses.tools.db import JobStatus, UlyssesDB

__all__ = ["NotifierAgent", "format_job_message"]

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


class NotifierAgent:
    """Sends scored jobs to a single Telegram chat and handles button presses."""

    def __init__(self, bot_token: str, chat_id: str, db: UlyssesDB) -> None:
        """Create a Notifier Agent bound to one Telegram chat.

        Args:
            bot_token: The Telegram bot token from BotFather.
            chat_id: The only chat ID this bot will ever send to or accept
                callbacks from — validated in `handle_callback`.
            db: Persistence layer, used to record notification and user-action state.
        """
        self._bot = Bot(token=bot_token)
        self._chat_id = str(chat_id)
        self._db = db
        self._batch_queue: list[tuple[JobPost, JobScore]] = []

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

    async def run_batch_loop(self, batch_interval_minutes: int) -> None:
        """Flush the batch queue on a fixed interval, forever."""
        interval = timedelta(minutes=batch_interval_minutes).total_seconds()
        while True:
            await asyncio.sleep(interval)
            await self.flush_batch()

    async def _send_job_alert(self, job: JobPost, score: JobScore) -> None:
        text = format_job_message(job, score)
        keyboard = _build_action_keyboard(job.id)
        await self._bot.send_message(
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
        else:
            # "draft" / "build" are wired up by the LangGraph orchestrator in
            # later phases; Phase 1 just records the intent was seen.
            logger.bind(job_id=job_id, agent="notifier").info(
                "User requested: {} (handled by graph)", action
            )


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
