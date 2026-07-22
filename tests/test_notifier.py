"""Tests for `ulysses.agents.notifier`: message formatting and agent behavior."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture
from telegram.error import NetworkError, TimedOut

from ulysses.agents.notifier import NotifierAgent, format_job_message
from ulysses.agents.scorer import score_job
from ulysses.config.profile import Profile
from ulysses.models import JobPost, JobScore
from ulysses.tools.db import JobStatus


@pytest.fixture
def fresh_score(fresh_job: JobPost, profile: Profile, now: datetime) -> JobScore:
    return score_job(fresh_job, profile, now=now)


class TestFormatJobMessage:
    def test_includes_score_title_and_recommendation(
        self, fresh_job: JobPost, fresh_score: JobScore
    ) -> None:
        message = format_job_message(fresh_job, fresh_score)
        assert fresh_job.title in message
        assert f"{fresh_score.total_score:.0f}/100" in message
        assert "APPLY NOW" in message

    def test_includes_matched_skills_and_repo(
        self, fresh_job: JobPost, fresh_score: JobScore
    ) -> None:
        message = format_job_message(fresh_job, fresh_score)
        assert "python" in message.lower()
        assert fresh_score.matched_repos[0].repo_name in message

    def test_shows_no_red_flags_when_none_detected(
        self, fresh_job: JobPost, fresh_score: JobScore
    ) -> None:
        message = format_job_message(fresh_job, fresh_score)
        assert "Red flags: none" in message

    def test_shows_red_flags_when_present(
        self, fresh_job: JobPost, profile: Profile, now: datetime
    ) -> None:
        job = fresh_job.model_copy(update={"description": "Simple task, prove yourself first."})
        score = score_job(job, profile, now=now)
        message = format_job_message(job, score)
        assert "simple task" in message
        assert "prove yourself" in message


class TestNotifierAgentThresholdRouting:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        db = MagicMock()
        db.update_status = AsyncMock()
        return NotifierAgent(bot_token="fake-token", chat_id="123456", db=db)

    async def test_instant_alert_sends_immediately(
        self, notifier: NotifierAgent, fresh_job: JobPost, fresh_score: JobScore, profile: Profile
    ) -> None:
        notifier._bot.send_message = AsyncMock()
        await notifier.handle_scored_job(fresh_job, fresh_score, profile.scoring)
        notifier._bot.send_message.assert_awaited_once()
        notifier._db.update_status.assert_awaited_once_with(fresh_job.id, JobStatus.NOTIFIED)

    async def test_mid_score_is_queued_not_sent(
        self, notifier: NotifierAgent, fresh_job: JobPost, fresh_score: JobScore, profile: Profile
    ) -> None:
        notifier._bot.send_message = AsyncMock()
        mid_score = fresh_score.model_copy(update={"total_score": 60.0})
        await notifier.handle_scored_job(fresh_job, mid_score, profile.scoring)
        notifier._bot.send_message.assert_not_awaited()
        assert len(notifier._batch_queue) == 1

    async def test_low_score_is_silently_dropped(
        self, notifier: NotifierAgent, fresh_job: JobPost, fresh_score: JobScore, profile: Profile
    ) -> None:
        notifier._bot.send_message = AsyncMock()
        low_score = fresh_score.model_copy(update={"total_score": 10.0})
        await notifier.handle_scored_job(fresh_job, low_score, profile.scoring)
        notifier._bot.send_message.assert_not_awaited()
        assert notifier._batch_queue == []

    async def test_flush_batch_sends_all_queued_jobs(
        self, notifier: NotifierAgent, fresh_job: JobPost, fresh_score: JobScore, profile: Profile
    ) -> None:
        notifier._bot.send_message = AsyncMock()
        mid_score = fresh_score.model_copy(update={"total_score": 60.0})
        await notifier.handle_scored_job(fresh_job, mid_score, profile.scoring)
        await notifier.flush_batch()
        notifier._bot.send_message.assert_awaited_once()
        assert notifier._batch_queue == []

    async def test_flush_batch_is_a_no_op_when_empty(self, notifier: NotifierAgent) -> None:
        notifier._bot.send_message = AsyncMock()
        await notifier.flush_batch()
        notifier._bot.send_message.assert_not_awaited()


class TestNotifierAgentCallbackHandling:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        db = MagicMock()
        db.update_status = AsyncMock()
        return NotifierAgent(bot_token="fake-token", chat_id="123456", db=db)

    def _make_update(self, chat_id: str, data: str) -> MagicMock:
        update = MagicMock()
        update.callback_query.data = data
        update.callback_query.message.chat_id = chat_id
        update.callback_query.answer = AsyncMock()
        return update

    async def test_skip_action_updates_status(self, notifier: NotifierAgent) -> None:
        update = self._make_update("123456", "skip:job-1")
        await notifier.handle_callback(update, MagicMock())
        notifier._db.update_status.assert_awaited_once_with("job-1", JobStatus.SKIPPED)

    async def test_archive_action_updates_status(self, notifier: NotifierAgent) -> None:
        update = self._make_update("123456", "archive:job-2")
        await notifier.handle_callback(update, MagicMock())
        notifier._db.update_status.assert_awaited_once_with("job-2", JobStatus.ARCHIVED)

    async def test_draft_action_does_not_touch_db_status_directly(
        self, notifier: NotifierAgent
    ) -> None:
        notifier.set_draft_handler(AsyncMock())
        update = self._make_update("123456", "draft:job-3")
        await notifier.handle_callback(update, MagicMock())
        notifier._db.update_status.assert_not_awaited()

    async def test_draft_action_invokes_draft_handler(self, notifier: NotifierAgent) -> None:
        handler = AsyncMock()
        notifier.set_draft_handler(handler)
        update = self._make_update("123456", "draft:job-3")
        await notifier.handle_callback(update, MagicMock())
        handler.assert_awaited_once_with("job-3")

    async def test_regenerate_action_invokes_draft_handler(self, notifier: NotifierAgent) -> None:
        handler = AsyncMock()
        notifier.set_draft_handler(handler)
        update = self._make_update("123456", "regenerate:job-3")
        await notifier.handle_callback(update, MagicMock())
        handler.assert_awaited_once_with("job-3")

    async def test_draft_with_no_handler_wired_does_not_raise(
        self, notifier: NotifierAgent
    ) -> None:
        update = self._make_update("123456", "draft:job-3")
        await notifier.handle_callback(update, MagicMock())  # should not raise

    async def test_draft_handler_failure_sends_error_message(self, notifier: NotifierAgent) -> None:
        notifier.set_draft_handler(AsyncMock(side_effect=RuntimeError("boom")))
        notifier._bot.send_message = AsyncMock()
        update = self._make_update("123456", "draft:job-3")
        await notifier.handle_callback(update, MagicMock())
        notifier._bot.send_message.assert_awaited_once()
        call_kwargs = notifier._bot.send_message.call_args.kwargs
        assert "Failed to draft" in call_kwargs["text"]

    async def test_copy_action_sends_latest_draft_content(self, notifier: NotifierAgent) -> None:
        draft = MagicMock(content="the draft text")
        notifier._db.get_proposal_drafts = AsyncMock(return_value=[draft])
        notifier._bot.send_message = AsyncMock()
        update = self._make_update("123456", "copy:job-3")
        await notifier.handle_callback(update, MagicMock())
        notifier._bot.send_message.assert_awaited_once_with(chat_id="123456", text="the draft text")

    async def test_copy_action_with_no_drafts_sends_nothing(self, notifier: NotifierAgent) -> None:
        notifier._db.get_proposal_drafts = AsyncMock(return_value=[])
        notifier._bot.send_message = AsyncMock()
        update = self._make_update("123456", "copy:job-3")
        await notifier.handle_callback(update, MagicMock())
        notifier._bot.send_message.assert_not_awaited()

    async def test_rejects_callback_from_unauthorized_chat(self, notifier: NotifierAgent) -> None:
        update = self._make_update("999999", "skip:job-4")
        await notifier.handle_callback(update, MagicMock())
        notifier._db.update_status.assert_not_awaited()
        update.callback_query.answer.assert_awaited_once_with("Unauthorized", show_alert=True)


class TestSendProposalDraft:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        db = MagicMock()
        db.update_status = AsyncMock()
        return NotifierAgent(bot_token="fake-token", chat_id="123456", db=db)

    async def test_sends_message_with_copy_and_regenerate_buttons(
        self, notifier: NotifierAgent
    ) -> None:
        notifier._bot.send_message = AsyncMock()
        await notifier.send_proposal_draft("job-9", "draft body text")

        notifier._bot.send_message.assert_awaited_once()
        call_kwargs = notifier._bot.send_message.call_args.kwargs
        assert "draft body text" in call_kwargs["text"]
        buttons = call_kwargs["reply_markup"].inline_keyboard[0]
        assert buttons[0].callback_data == "copy:job-9"
        assert buttons[1].callback_data == "regenerate:job-9"
        notifier._db.update_status.assert_awaited_once_with("job-9", JobStatus.DRAFTED)


class TestSendErrorMessage:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        return NotifierAgent(bot_token="fake-token", chat_id="123456", db=MagicMock())

    async def test_sends_plain_error_text(self, notifier: NotifierAgent) -> None:
        notifier._bot.send_message = AsyncMock()
        await notifier.send_error_message("something broke")
        notifier._bot.send_message.assert_awaited_once_with(
            chat_id="123456", text="⚠️ something broke"
        )


class TestSendMessageRetry:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        return NotifierAgent(bot_token="fake-token", chat_id="123456", db=MagicMock())

    async def test_retries_on_network_error_then_succeeds(
        self, notifier: NotifierAgent, mocker: MockerFixture
    ) -> None:
        mocker.patch("asyncio.sleep", AsyncMock())
        notifier._bot.send_message = AsyncMock(side_effect=[TimedOut(), None])
        await notifier.send_error_message("retry me")
        assert notifier._bot.send_message.await_count == 2

    async def test_reraises_after_exhausting_retries(
        self, notifier: NotifierAgent, mocker: MockerFixture
    ) -> None:
        mocker.patch("asyncio.sleep", AsyncMock())
        notifier._bot.send_message = AsyncMock(side_effect=NetworkError("down"))
        with pytest.raises(NetworkError):
            await notifier.send_error_message("will fail")
        assert notifier._bot.send_message.await_count == 3

    async def test_draft_handler_failure_and_error_notification_both_failing_does_not_raise(
        self, notifier: NotifierAgent, mocker: MockerFixture
    ) -> None:
        mocker.patch("asyncio.sleep", AsyncMock())
        notifier.set_draft_handler(AsyncMock(side_effect=RuntimeError("boom")))
        notifier._bot.send_message = AsyncMock(side_effect=NetworkError("also down"))
        update = MagicMock()
        update.callback_query.data = "draft:job-1"
        update.callback_query.message.chat_id = "123456"
        update.callback_query.answer = AsyncMock()

        await notifier.handle_callback(update, MagicMock())  # must not raise
