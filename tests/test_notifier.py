"""Tests for `ulysses.agents.notifier`: message formatting and agent behavior."""

from __future__ import annotations

import threading
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture
from telegram.error import NetworkError, TimedOut

from ulysses.agents.notifier import NotifierAgent, _looks_like_job_paste, format_job_message
from ulysses.agents.scorer import score_job
from ulysses.config.profile import Profile
from ulysses.models import GeneratedProposal, GeneratedPrototype, JobPost, JobScore
from ulysses.tools.db import JobStatus
from ulysses.tools.manual_job import ManualJobParseError


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
    def notifier(self, mocker: MockerFixture, profile: Profile) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        db = MagicMock()
        db.update_status = AsyncMock()
        return NotifierAgent(bot_token="fake-token", chat_id="123456", db=db, profile=profile)

    async def test_instant_alert_sends_immediately(
        self, notifier: NotifierAgent, fresh_job: JobPost, fresh_score: JobScore, profile: Profile
    ) -> None:
        notifier._bot.send_message = AsyncMock()
        await notifier.handle_scored_job(fresh_job, fresh_score, profile.scoring)
        notifier._bot.send_message.assert_awaited_once()
        notifier._db.update_status.assert_awaited_once_with(fresh_job.id, JobStatus.NOTIFIED)

    async def test_instant_alert_invokes_instant_alert_hook(
        self, notifier: NotifierAgent, fresh_job: JobPost, fresh_score: JobScore, profile: Profile
    ) -> None:
        notifier._bot.send_message = AsyncMock()
        hook = MagicMock()
        notifier.set_instant_alert_hook(hook)

        await notifier.handle_scored_job(fresh_job, fresh_score, profile.scoring)

        hook.assert_called_once_with(fresh_job, fresh_score)

    async def test_instant_alert_hook_failure_does_not_break_telegram_send(
        self, notifier: NotifierAgent, fresh_job: JobPost, fresh_score: JobScore, profile: Profile
    ) -> None:
        notifier._bot.send_message = AsyncMock()
        notifier.set_instant_alert_hook(MagicMock(side_effect=RuntimeError("boom")))

        await notifier.handle_scored_job(fresh_job, fresh_score, profile.scoring)  # must not raise

        notifier._bot.send_message.assert_awaited_once()

    async def test_hook_is_not_invoked_for_batched_or_silent_jobs(
        self, notifier: NotifierAgent, fresh_job: JobPost, fresh_score: JobScore, profile: Profile
    ) -> None:
        notifier._bot.send_message = AsyncMock()
        hook = MagicMock()
        notifier.set_instant_alert_hook(hook)

        mid_score = fresh_score.model_copy(update={"total_score": 60.0})
        await notifier.handle_scored_job(fresh_job, mid_score, profile.scoring)
        low_score = fresh_score.model_copy(update={"total_score": 10.0})
        await notifier.handle_scored_job(fresh_job, low_score, profile.scoring)

        hook.assert_not_called()

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


class TestRunBatchLoop:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture, profile: Profile) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        db = MagicMock()
        db.update_status = AsyncMock()
        return NotifierAgent(bot_token="fake-token", chat_id="123456", db=db, profile=profile)

    async def test_stop_event_ends_the_loop_immediately(self, notifier: NotifierAgent) -> None:
        stop_event = threading.Event()
        stop_event.set()

        # Should return without ever sleeping or flushing -- no mocking needed.
        await notifier.run_batch_loop(30, stop_event=stop_event)

    async def test_flushes_on_each_interval_until_stopped(
        self, notifier: NotifierAgent, mocker: MockerFixture
    ) -> None:
        flush_mock = mocker.patch.object(notifier, "flush_batch", AsyncMock())
        stop_event = threading.Event()
        call_count = 0

        async def fake_sleep(_seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                stop_event.set()

        mocker.patch("asyncio.sleep", side_effect=fake_sleep)

        await notifier.run_batch_loop(30, stop_event=stop_event)

        assert flush_mock.await_count == 2


class TestNotifierAgentCallbackHandling:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture, profile: Profile) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        db = MagicMock()
        db.update_status = AsyncMock()
        return NotifierAgent(bot_token="fake-token", chat_id="123456", db=db, profile=profile)

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

    async def test_build_action_invokes_build_handler(self, notifier: NotifierAgent) -> None:
        handler = AsyncMock()
        notifier.set_build_handler(handler)
        update = self._make_update("123456", "build:job-5")
        await notifier.handle_callback(update, MagicMock())
        handler.assert_awaited_once_with("job-5")

    async def test_build_with_no_handler_wired_does_not_raise(
        self, notifier: NotifierAgent
    ) -> None:
        update = self._make_update("123456", "build:job-5")
        await notifier.handle_callback(update, MagicMock())  # should not raise

    async def test_build_handler_failure_sends_error_message(self, notifier: NotifierAgent) -> None:
        notifier.set_build_handler(AsyncMock(side_effect=RuntimeError("boom")))
        notifier._bot.send_message = AsyncMock()
        update = self._make_update("123456", "build:job-5")
        await notifier.handle_callback(update, MagicMock())
        notifier._bot.send_message.assert_awaited_once()
        call_kwargs = notifier._bot.send_message.call_args.kwargs
        assert "Failed to build" in call_kwargs["text"]

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
    def notifier(self, mocker: MockerFixture, profile: Profile) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        db = MagicMock()
        db.update_status = AsyncMock()
        return NotifierAgent(bot_token="fake-token", chat_id="123456", db=db, profile=profile)

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


class TestSendPrototypeZip:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture, profile: Profile) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        db = MagicMock()
        db.update_status = AsyncMock()
        return NotifierAgent(bot_token="fake-token", chat_id="123456", db=db, profile=profile)

    async def test_sends_document_and_readme_preview(self, notifier: NotifierAgent) -> None:
        prototype = GeneratedPrototype(
            job_id="job-9",
            category="scraper",
            demo_script="print('hi')",
            requirements_txt="requests==2.32.3\n",
            readme_md="# Demo README content",
            config_example_env="# none needed\n",
            zip_filename="ulysses_demo_job-9.zip",
        )
        notifier._bot.send_document = AsyncMock()
        notifier._bot.send_message = AsyncMock()

        await notifier.send_prototype_zip("job-9", prototype, b"zip-bytes")

        notifier._bot.send_document.assert_awaited_once()
        doc_kwargs = notifier._bot.send_document.call_args.kwargs
        assert doc_kwargs["document"] == b"zip-bytes"
        assert doc_kwargs["filename"] == "ulysses_demo_job-9.zip"

        notifier._bot.send_message.assert_awaited_once_with(
            chat_id="123456", text="# Demo README content"
        )
        notifier._db.update_status.assert_awaited_once_with("job-9", JobStatus.BUILT)


class TestSendErrorMessage:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture, profile: Profile) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        return NotifierAgent(
            bot_token="fake-token", chat_id="123456", db=MagicMock(), profile=profile
        )

    async def test_sends_plain_error_text(self, notifier: NotifierAgent) -> None:
        notifier._bot.send_message = AsyncMock()
        await notifier.send_error_message("something broke")
        notifier._bot.send_message.assert_awaited_once_with(
            chat_id="123456", text="⚠️ something broke"
        )


class TestSendMessageRetry:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture, profile: Profile) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        return NotifierAgent(
            bot_token="fake-token", chat_id="123456", db=MagicMock(), profile=profile
        )

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


class TestLooksLikeJobPaste:
    def test_short_text_is_not_a_job_paste(self) -> None:
        assert _looks_like_job_paste("What's my highest scoring job?") is False

    def test_long_text_is_a_job_paste(self) -> None:
        assert _looks_like_job_paste("x" * 150) is True

    def test_boundary_length(self) -> None:
        assert _looks_like_job_paste("x" * 100) is True
        assert _looks_like_job_paste("x" * 99) is False


class TestHandleTextMessage:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture, profile: Profile) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        db = AsyncMock()
        db.list_jobs = AsyncMock(return_value=[])
        return NotifierAgent(bot_token="fake-token", chat_id="123456", db=db, profile=profile)

    def _make_message_update(self, chat_id: str, text: str) -> MagicMock:
        update = MagicMock()
        update.message.chat_id = chat_id
        update.message.text = text
        return update

    def _mock_chat_agent(self, mocker: MockerFixture, reply: str) -> MagicMock:
        return mocker.patch(
            "ulysses.agents.notifier.ChatAgent",
            return_value=MagicMock(reply=AsyncMock(return_value=reply)),
        )

    async def test_ignores_unauthorized_chat(self, notifier: NotifierAgent) -> None:
        handler = AsyncMock()
        notifier.set_job_text_handler(handler)
        update = self._make_message_update("999999", "x" * 150)
        await notifier.handle_text_message(update, MagicMock())
        handler.assert_not_awaited()

    async def test_update_with_no_message_is_ignored(self, notifier: NotifierAgent) -> None:
        update = MagicMock()
        update.message = None
        await notifier.handle_text_message(update, MagicMock())  # must not raise

    async def test_blank_message_is_ignored(self, notifier: NotifierAgent) -> None:
        notifier._bot.send_message = AsyncMock()
        update = self._make_message_update("123456", "   ")
        await notifier.handle_text_message(update, MagicMock())
        notifier._bot.send_message.assert_not_awaited()

    async def test_short_message_routes_to_chat_agent(
        self, notifier: NotifierAgent, mocker: MockerFixture
    ) -> None:
        chat_agent_mock = self._mock_chat_agent(mocker, "A reply.")
        notifier._bot.send_message = AsyncMock()
        update = self._make_message_update("123456", "What's my best job?")

        await notifier.handle_text_message(update, MagicMock())

        chat_agent_mock.return_value.reply.assert_awaited_once()
        _, message_arg = chat_agent_mock.return_value.reply.await_args.args
        assert message_arg == "What's my best job?"
        notifier._bot.send_message.assert_awaited_once_with(chat_id="123456", text="A reply.")

    async def test_long_message_that_parses_triggers_job_pipeline_not_chat(
        self, notifier: NotifierAgent, mocker: MockerFixture
    ) -> None:
        handler = AsyncMock()
        notifier.set_job_text_handler(handler)
        chat_agent_mock = mocker.patch("ulysses.agents.notifier.ChatAgent")
        text = "A " * 60
        update = self._make_message_update("123456", text)

        await notifier.handle_text_message(update, MagicMock())

        handler.assert_awaited_once_with(text.strip())
        chat_agent_mock.assert_not_called()

    async def test_parse_error_falls_back_to_chat_without_an_error_message(
        self, notifier: NotifierAgent, mocker: MockerFixture
    ) -> None:
        notifier.set_job_text_handler(AsyncMock(side_effect=ManualJobParseError("nope")))
        chat_agent_mock = self._mock_chat_agent(mocker, "A reply.")
        notifier._bot.send_message = AsyncMock()
        text = "A " * 60
        update = self._make_message_update("123456", text)

        await notifier.handle_text_message(update, MagicMock())

        chat_agent_mock.return_value.reply.assert_awaited_once()
        notifier._bot.send_message.assert_awaited_once_with(chat_id="123456", text="A reply.")

    async def test_other_exception_sends_error_message_not_chat(
        self, notifier: NotifierAgent, mocker: MockerFixture
    ) -> None:
        notifier.set_job_text_handler(AsyncMock(side_effect=RuntimeError("boom")))
        chat_agent_mock = mocker.patch("ulysses.agents.notifier.ChatAgent")
        notifier._bot.send_message = AsyncMock()
        text = "A " * 60
        update = self._make_message_update("123456", text)

        await notifier.handle_text_message(update, MagicMock())

        chat_agent_mock.assert_not_called()
        notifier._bot.send_message.assert_awaited_once()
        assert "Something went wrong" in notifier._bot.send_message.call_args.kwargs["text"]

    async def test_no_job_handler_wired_falls_back_to_chat_even_for_long_messages(
        self, notifier: NotifierAgent, mocker: MockerFixture
    ) -> None:
        chat_agent_mock = self._mock_chat_agent(mocker, "A reply.")
        notifier._bot.send_message = AsyncMock()
        text = "A " * 60
        update = self._make_message_update("123456", text)

        await notifier.handle_text_message(update, MagicMock())

        chat_agent_mock.return_value.reply.assert_awaited_once()

    async def test_chat_reply_failure_sends_error_message(
        self, notifier: NotifierAgent, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "ulysses.agents.notifier.ChatAgent",
            return_value=MagicMock(reply=AsyncMock(side_effect=RuntimeError("boom"))),
        )
        notifier._bot.send_message = AsyncMock()
        update = self._make_message_update("123456", "short question")

        await notifier.handle_text_message(update, MagicMock())

        notifier._bot.send_message.assert_awaited_once()
        assert "Something went wrong" in notifier._bot.send_message.call_args.kwargs["text"]

    async def test_second_turn_reuses_the_same_chat_agent_instance(
        self, notifier: NotifierAgent, mocker: MockerFixture
    ) -> None:
        chat_agent_mock = self._mock_chat_agent(mocker, "A reply.")
        notifier._bot.send_message = AsyncMock()

        first = self._make_message_update("123456", "first")
        second = self._make_message_update("123456", "second")
        await notifier.handle_text_message(first, MagicMock())
        await notifier.handle_text_message(second, MagicMock())

        chat_agent_mock.assert_called_once()
        assert chat_agent_mock.return_value.reply.await_count == 2


class TestHandleJobCommand:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture, profile: Profile) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        return NotifierAgent(
            bot_token="fake-token", chat_id="123456", db=AsyncMock(), profile=profile
        )

    def _make_command_update(self, chat_id: str) -> MagicMock:
        update = MagicMock()
        update.message.chat_id = chat_id
        return update

    async def test_no_identifier_shows_usage(self, notifier: NotifierAgent) -> None:
        notifier._bot.send_message = AsyncMock()
        update = self._make_command_update("123456")

        await notifier.handle_job_command(update, MagicMock(args=[]))

        notifier._bot.send_message.assert_awaited_once_with(
            chat_id="123456", text="Usage: /job <url-or-id>"
        )

    async def test_unknown_identifier_sends_not_found(
        self, notifier: NotifierAgent, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "ulysses.agents.notifier.build_job_lookup_message", new=AsyncMock(return_value=None)
        )
        notifier._bot.send_message = AsyncMock()
        update = self._make_command_update("123456")

        await notifier.handle_job_command(update, MagicMock(args=["bad-id"]))

        notifier._bot.send_message.assert_awaited_once_with(
            chat_id="123456", text="No job found matching: bad-id"
        )

    async def test_found_identifier_pushes_history_and_confirms(
        self, notifier: NotifierAgent, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "ulysses.agents.notifier.build_job_lookup_message",
            new=AsyncMock(return_value="job detail text"),
        )
        notifier._bot.send_message = AsyncMock()
        update = self._make_command_update("123456")

        await notifier.handle_job_command(update, MagicMock(args=["job-1"]))

        assert notifier._chat_history[0]["content"] == "job detail text"
        notifier._bot.send_message.assert_awaited_once_with(
            chat_id="123456", text="Loaded details for job-1."
        )

    async def test_unauthorized_chat_does_nothing(self, notifier: NotifierAgent) -> None:
        notifier._bot.send_message = AsyncMock()
        update = self._make_command_update("999999")

        await notifier.handle_job_command(update, MagicMock(args=["job-1"]))

        notifier._bot.send_message.assert_not_awaited()


class TestHandleRefreshCommand:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture, profile: Profile) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        db = AsyncMock()
        db.list_jobs = AsyncMock(return_value=[])
        return NotifierAgent(bot_token="fake-token", chat_id="123456", db=db, profile=profile)

    def _make_command_update(self, chat_id: str) -> MagicMock:
        update = MagicMock()
        update.message.chat_id = chat_id
        return update

    async def test_refreshes_queue_digest_and_confirms(self, notifier: NotifierAgent) -> None:
        notifier._bot.send_message = AsyncMock()
        update = self._make_command_update("123456")

        await notifier.handle_refresh_command(update, MagicMock())

        notifier._db.list_jobs.assert_awaited_once()
        assert len(notifier._chat_history) == 2
        notifier._bot.send_message.assert_awaited_once_with(
            chat_id="123456", text="Queue digest refreshed."
        )

    async def test_unauthorized_chat_does_nothing(self, notifier: NotifierAgent) -> None:
        notifier._bot.send_message = AsyncMock()
        update = self._make_command_update("999999")

        await notifier.handle_refresh_command(update, MagicMock())

        notifier._db.list_jobs.assert_not_awaited()
        notifier._bot.send_message.assert_not_awaited()


class TestHandleHelpCommand:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture, profile: Profile) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        return NotifierAgent(
            bot_token="fake-token", chat_id="123456", db=AsyncMock(), profile=profile
        )

    async def test_sends_capability_summary(self, notifier: NotifierAgent) -> None:
        notifier._bot.send_message = AsyncMock()
        update = MagicMock()
        update.message.chat_id = "123456"

        await notifier.handle_help_command(update, MagicMock())

        notifier._bot.send_message.assert_awaited_once()
        assert "/job" in notifier._bot.send_message.call_args.kwargs["text"]

    async def test_unauthorized_chat_does_nothing(self, notifier: NotifierAgent) -> None:
        notifier._bot.send_message = AsyncMock()
        update = MagicMock()
        update.message.chat_id = "999999"

        await notifier.handle_help_command(update, MagicMock())

        notifier._bot.send_message.assert_not_awaited()


class TestHandlerProperties:
    """The new inbound-message handler properties are real `python-telegram-bot` handlers."""

    @pytest.fixture
    def notifier(self, mocker: MockerFixture, profile: Profile) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        return NotifierAgent(
            bot_token="fake-token", chat_id="123456", db=AsyncMock(), profile=profile
        )

    def test_message_handler_is_a_message_handler(self, notifier: NotifierAgent) -> None:
        from telegram.ext import MessageHandler

        assert isinstance(notifier.message_handler, MessageHandler)

    def test_job_command_handler_is_a_command_handler(self, notifier: NotifierAgent) -> None:
        from telegram.ext import CommandHandler

        assert isinstance(notifier.job_command_handler, CommandHandler)

    def test_refresh_command_handler_is_a_command_handler(self, notifier: NotifierAgent) -> None:
        from telegram.ext import CommandHandler

        assert isinstance(notifier.refresh_command_handler, CommandHandler)

    def test_help_command_handler_is_a_command_handler(self, notifier: NotifierAgent) -> None:
        from telegram.ext import CommandHandler

        assert isinstance(notifier.help_command_handler, CommandHandler)


class TestSendJobProcessedSummary:
    @pytest.fixture
    def notifier(self, mocker: MockerFixture, profile: Profile) -> NotifierAgent:
        mocker.patch("ulysses.agents.notifier.Bot")
        return NotifierAgent(
            bot_token="fake-token", chat_id="123456", db=AsyncMock(), profile=profile
        )

    def _proposal(self) -> GeneratedProposal:
        return GeneratedProposal(
            job_id="job-1",
            category="scraping",
            hook="hook",
            plan_bullets=["step"],
            close="close",
            proof_repo="repo",
            proof_repo_url="https://example.com/repo",
            timeline="3 days",
            bid_usd=200.0,
            full_text="Generated proposal text.",
        )

    def _prototype(self) -> GeneratedPrototype:
        return GeneratedPrototype(
            job_id="job-1",
            category="scraper",
            demo_script="print('hi')",
            requirements_txt="requests\n",
            readme_md="# Demo\n",
            config_example_env="# none\n",
            zip_filename="demo.zip",
        )

    async def test_sends_score_card_proposal_and_prototype(
        self, notifier: NotifierAgent, fresh_job: JobPost, fresh_score: JobScore
    ) -> None:
        notifier.send_proposal_draft = AsyncMock()
        notifier.send_prototype_zip = AsyncMock()
        notifier._bot.send_message = AsyncMock()
        proposal, prototype = self._proposal(), self._prototype()

        await notifier.send_job_processed_summary(
            fresh_job, fresh_score, proposal, prototype, b"zip-bytes"
        )

        notifier._bot.send_message.assert_awaited_once()
        notifier.send_proposal_draft.assert_awaited_once_with(fresh_job.id, proposal.full_text)
        notifier.send_prototype_zip.assert_awaited_once_with(fresh_job.id, prototype, b"zip-bytes")
        assert len(notifier._chat_history) == 2
        assert fresh_job.title in notifier._chat_history[0]["content"]

    async def test_skipped_job_sends_plain_note_instead_of_draft(
        self, notifier: NotifierAgent, fresh_job: JobPost, fresh_score: JobScore
    ) -> None:
        notifier.send_proposal_draft = AsyncMock()
        notifier.send_prototype_zip = AsyncMock()
        notifier._bot.send_message = AsyncMock()

        await notifier.send_job_processed_summary(fresh_job, fresh_score, None, None, None)

        notifier.send_proposal_draft.assert_not_awaited()
        notifier.send_prototype_zip.assert_not_awaited()
        assert notifier._bot.send_message.await_count == 2
        skip_text = notifier._bot.send_message.await_args_list[1].kwargs["text"]
        assert "SKIP" in skip_text
