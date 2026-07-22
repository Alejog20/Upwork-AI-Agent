"""Factory for the shared chat LLM client — the only place a model is instantiated.

Every agent must obtain its LLM through `get_llm()` rather than constructing a
chat model directly, so swapping providers/models is a one-line change. Retry
logic wraps every call via `ainvoke_with_retry`; the 30s hard timeout and
retry count are configured on the client itself in `get_llm()`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from openai import APIError, APITimeoutError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ulysses.config.settings import get_settings

__all__ = ["ainvoke_with_retry", "get_llm"]

_TIMEOUT_SECONDS = 30
_MAX_RETRIES = 3

_llm_retry = retry(
    retry=retry_if_exception_type((APIError, APITimeoutError)),
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
