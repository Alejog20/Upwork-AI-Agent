"""Tests for `ulysses.tools.llm`: the shared LLM client factory and retry wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from langchain_openai import ChatOpenAI
from openai import APITimeoutError
from pytest_mock import MockerFixture

from ulysses.config.settings import get_settings
from ulysses.tools.llm import aembed_texts, ainvoke_with_retry, get_llm

_FAKE_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULYSSES_IMAP_USER", "me@gmail.com")
    monkeypatch.setenv("ULYSSES_IMAP_APP_PASSWORD", "secret")
    monkeypatch.setenv("ULYSSES_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("ULYSSES_TELEGRAM_CHAT_ID", "123456")
    monkeypatch.setenv("ULYSSES_LLM_API_KEY", "test-key")
    get_settings.cache_clear()


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
        monkeypatch.setenv("ULYSSES_LLM_API_KEY", "test-key")
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
        monkeypatch.setenv("ULYSSES_LLM_API_KEY", "test-key")
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


def _mock_embed_client(mocker: MockerFixture, responses: dict[str, list[float]]) -> AsyncMock:
    """Patch `httpx.AsyncClient` so each POST returns the vector for its request text."""

    async def _fake_post(url: str, params: dict, json: dict) -> MagicMock:
        text = json["content"]["parts"][0]["text"]
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = MagicMock(return_value={"embedding": {"values": responses[text]}})
        return response

    client = AsyncMock()
    client.post = AsyncMock(side_effect=_fake_post)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    mocker.patch("ulysses.tools.llm.httpx.AsyncClient", return_value=client)
    return client


class TestAembedTexts:
    async def test_embeds_each_text_and_returns_vectors_in_order(
        self, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        _set_required_env(monkeypatch)
        _mock_embed_client(mocker, {"hello": [0.1, 0.2], "world": [0.3, 0.4]})

        try:
            vectors = await aembed_texts(["hello", "world"])
        finally:
            get_settings.cache_clear()

        assert vectors == [[0.1, 0.2], [0.3, 0.4]]

    async def test_retries_on_transient_http_error_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        _set_required_env(monkeypatch)
        mocker.patch("asyncio.sleep", AsyncMock())

        call_count = 0

        async def _flaky_post(url: str, params: dict, json: dict) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("timed out")
            response = MagicMock()
            response.raise_for_status = MagicMock()
            response.json = MagicMock(return_value={"embedding": {"values": [1.0]}})
            return response

        client = AsyncMock()
        client.post = AsyncMock(side_effect=_flaky_post)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        mocker.patch("ulysses.tools.llm.httpx.AsyncClient", return_value=client)

        try:
            vectors = await aembed_texts(["hello"])
        finally:
            get_settings.cache_clear()

        assert vectors == [[1.0]]
        assert call_count == 2

    async def test_reraises_after_exhausting_retries(
        self, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        _set_required_env(monkeypatch)
        mocker.patch("asyncio.sleep", AsyncMock())

        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        mocker.patch("ulysses.tools.llm.httpx.AsyncClient", return_value=client)

        try:
            with pytest.raises(httpx.TimeoutException):
                await aembed_texts(["hello"])
        finally:
            get_settings.cache_clear()

        assert client.post.await_count == 3
