"""Tests for `ulysses.tools.llm`: the shared LLM client factory and retry wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_openai import ChatOpenAI
from openai import APITimeoutError
from pytest_mock import MockerFixture

from ulysses.tools.llm import ainvoke_with_retry, get_llm

_FAKE_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(request=_FAKE_REQUEST)


class TestGetLlm:
    def test_returns_a_chat_openai_client_configured_from_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ULYSSES_IMAP_USER", "me@gmail.com")
        monkeypatch.setenv("ULYSSES_IMAP_APP_PASSWORD", "secret")
        monkeypatch.setenv("ULYSSES_TELEGRAM_BOT_TOKEN", "token")
        monkeypatch.setenv("ULYSSES_TELEGRAM_CHAT_ID", "123456")
        monkeypatch.setenv("ULYSSES_LLM_MODEL", "gpt-4o-mini")
        get_llm.cache_clear()
        try:
            llm = get_llm()
            assert isinstance(llm, ChatOpenAI)
            assert llm.model_name == "gpt-4o-mini"
        finally:
            get_llm.cache_clear()

    def test_returns_the_same_cached_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ULYSSES_IMAP_USER", "me@gmail.com")
        monkeypatch.setenv("ULYSSES_IMAP_APP_PASSWORD", "secret")
        monkeypatch.setenv("ULYSSES_TELEGRAM_BOT_TOKEN", "token")
        monkeypatch.setenv("ULYSSES_TELEGRAM_CHAT_ID", "123456")
        get_llm.cache_clear()
        try:
            assert get_llm() is get_llm()
        finally:
            get_llm.cache_clear()


class TestAinvokeWithRetry:
    async def test_returns_result_on_success(self) -> None:
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(return_value="ok")
        assert await ainvoke_with_retry(runnable, {"x": 1}) == "ok"
        runnable.ainvoke.assert_awaited_once_with({"x": 1})

    async def test_retries_on_transient_timeout_then_succeeds(self, mocker: MockerFixture) -> None:
        mocker.patch("asyncio.sleep", AsyncMock())
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=[_timeout_error(), "ok"])
        assert await ainvoke_with_retry(runnable, {}) == "ok"
        assert runnable.ainvoke.await_count == 2

    async def test_reraises_after_exhausting_retries(self, mocker: MockerFixture) -> None:
        mocker.patch("asyncio.sleep", AsyncMock())
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=_timeout_error())
        with pytest.raises(APITimeoutError):
            await ainvoke_with_retry(runnable, {})
        assert runnable.ainvoke.await_count == 3
