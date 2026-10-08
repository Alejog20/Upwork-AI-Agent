"""In-process pub/sub so the dashboard can reflect live state changes.

Pure `asyncio` -- no FastAPI/web dependency -- so `tools/` stays
framework-agnostic and agents can depend on it exactly like they already
depend on `tools/db.py`. `NotifierAgent` and `cli.main`'s action functions
publish events here; `ulysses.dashboard.api`'s `/ws/events` route is the
only subscriber that knows this bus exists.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

__all__ = ["DashboardEventBus"]


class DashboardEventBus:
    """Fans out broadcast events to every currently-subscribed consumer.

    Each subscriber gets its own unbounded `asyncio.Queue`; a slow or
    disconnected consumer can never block `broadcast` for the others, and
    unsubscribing (via the `subscribe` generator finishing or being closed)
    is automatic.
    """

    def __init__(self) -> None:
        """Start with no subscribers."""
        self._subscribers: set[asyncio.Queue[dict[str, object]]] = set()

    async def broadcast(self, event: dict[str, object]) -> None:
        """Push `event` onto every currently-subscribed consumer's queue."""
        for queue in list(self._subscribers):
            queue.put_nowait(event)

    async def subscribe(self) -> AsyncIterator[dict[str, object]]:
        """Yield events as they're broadcast, until the consumer stops iterating."""
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    async def aclose(self) -> None:
        """Drop all subscribers, e.g. during process shutdown."""
        self._subscribers.clear()
