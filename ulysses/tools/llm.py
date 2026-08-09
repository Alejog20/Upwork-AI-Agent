"""Factory for the shared chat LLM client — the only place a model is instantiated.

Every agent must obtain its LLM through `get_llm()` rather than constructing a
chat model directly, so swapping providers/models is a one-line change. Retry
logic wraps every call via `ainvoke_with_retry`; the 30s hard timeout and
retry count are configured on the client itself in `get_llm()`.

`aembed_texts` is a separate, narrower entry point for embeddings (used by
`tools.example_retrieval` for semantic example-proposal retrieval). It does
NOT go through `get_llm()`/`ChatOpenAI` -- Gemini's OpenAI-compatible endpoint
(what `llm_base_url` points `ChatOpenAI` at) returns HTTP 501 UNIMPLEMENTED for
embeddings, confirmed empirically. Gemini's *native* embedContent endpoint (a
different URL, same API key) works, so this calls it directly via `httpx`
(already a project dependency) rather than adding a whole new LangChain
provider package for one REST call.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

import httpx
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from openai import APIError, APITimeoutError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ulysses.config.settings import get_settings

__all__ = ["aembed_texts", "ainvoke_with_retry", "get_llm"]

_TIMEOUT_SECONDS = 30
_MAX_RETRIES = 3
_EMBED_CONTENT_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"

_llm_retry = retry(
    retry=retry_if_exception_type((APIError, APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(_MAX_RETRIES),
    reraise=True,
)

_embed_retry = retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(_MAX_RETRIES),
    reraise=True,
)


@lru_cache
def get_llm() -> ChatOpenAI:
    """Return the shared, process-wide chat model client, configured from `Settings`."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=_TIMEOUT_SECONDS,
        max_retries=_MAX_RETRIES,
    )


@_llm_retry
async def ainvoke_with_retry(runnable: Runnable, input_: Any) -> Any:
    """Invoke a LangChain `Runnable` with retry — use this for every agent LLM call."""
    return await runnable.ainvoke(input_)


async def aembed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via Gemini's native embedContent REST endpoint.

    Runs one request per text, in parallel via `asyncio.gather` (async-first,
    per project convention), each independently retried on failure.
    """
    settings = get_settings()
    url = _EMBED_CONTENT_URL.format(model=settings.llm_embedding_model)
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
        return await asyncio.gather(
            *(_embed_one(client, url, settings.llm_api_key, text) for text in texts)
        )


@_embed_retry
async def _embed_one(client: httpx.AsyncClient, url: str, api_key: str, text: str) -> list[float]:
    response = await client.post(
        url, params={"key": api_key}, json={"content": {"parts": [{"text": text}]}}
    )
    response.raise_for_status()
    return response.json()["embedding"]["values"]
