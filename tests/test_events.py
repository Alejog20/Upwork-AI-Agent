"""Tests for the in-process pub/sub bus in `ulysses.tools.events`."""

from __future__ import annotations

import asyncio
import contextlib

from ulysses.tools.events import DashboardEventBus


class TestDashboardEventBus:
    async def test_a_single_subscriber_receives_a_broadcast_event(self) -> None:
        bus = DashboardEventBus()
        subscriber = bus.subscribe()
        task = asyncio.ensure_future(anext(subscriber))
        await asyncio.sleep(0)  # let the generator run until its first `await`

        await bus.broadcast({"type": "job_scored", "job_id": "job-1"})
        event = await asyncio.wait_for(task, timeout=1.0)

        assert event == {"type": "job_scored", "job_id": "job-1"}

    async def test_multiple_subscribers_each_receive_the_same_event(self) -> None:
        bus = DashboardEventBus()
        first_task = asyncio.ensure_future(anext(bus.subscribe()))
        second_task = asyncio.ensure_future(anext(bus.subscribe()))
        await asyncio.sleep(0)

        await bus.broadcast({"type": "job_updated", "job_id": "job-2"})

        first_event, second_event = await asyncio.wait_for(
            asyncio.gather(first_task, second_task), timeout=1.0
        )
        assert first_event == {"type": "job_updated", "job_id": "job-2"}
        assert second_event == {"type": "job_updated", "job_id": "job-2"}

    async def test_broadcast_with_no_subscribers_does_not_raise(self) -> None:
        bus = DashboardEventBus()
        await bus.broadcast({"type": "job_scored", "job_id": "job-3"})

    async def test_a_closed_subscriber_is_removed_and_does_not_affect_others(self) -> None:
        bus = DashboardEventBus()
        leaving = bus.subscribe()
        staying = bus.subscribe()
        leaving_task = asyncio.ensure_future(anext(leaving))
        staying_task = asyncio.ensure_future(anext(staying))
        await asyncio.sleep(0)
        assert len(bus._subscribers) == 2

        leaving_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await leaving_task
        await leaving.aclose()
        assert len(bus._subscribers) == 1

        await bus.broadcast({"type": "job_updated", "job_id": "job-4"})
        event = await asyncio.wait_for(staying_task, timeout=1.0)

        assert event == {"type": "job_updated", "job_id": "job-4"}

    async def test_aclose_drops_all_subscribers(self) -> None:
        bus = DashboardEventBus()
        subscriber_task = asyncio.ensure_future(anext(bus.subscribe()))
        await asyncio.sleep(0)
        assert len(bus._subscribers) == 1

        await bus.aclose()

        assert bus._subscribers == set()
        subscriber_task.cancel()
