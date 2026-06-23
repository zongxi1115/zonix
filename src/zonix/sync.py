from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import TypeVar

T = TypeVar("T")

_STREAM_DONE = object()


def run_sync(awaitable_factory: Callable[[], Awaitable[T]]) -> T:
    """Run an async Zonix operation from synchronous Python code."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable_factory())

    result: dict[str, T] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable_factory())
        except BaseException as exc:  # noqa: BLE001 - re-raise on caller thread
            error["value"] = exc

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if error:
        raise error["value"]
    return result["value"]


def iter_async_sync(async_iter_factory: Callable[[], AsyncIterator[T]]) -> Iterator[T]:
    """Expose an async iterator as a blocking synchronous iterator."""

    items: queue.Queue[object] = queue.Queue()

    async def consume() -> None:
        try:
            async for item in async_iter_factory():
                items.put(item)
        except BaseException as exc:  # noqa: BLE001 - re-raise on caller thread
            items.put(exc)
        finally:
            items.put(_STREAM_DONE)

    def runner() -> None:
        asyncio.run(consume())

    thread = threading.Thread(target=runner)
    thread.start()
    try:
        while True:
            item = items.get()
            if item is _STREAM_DONE:
                break
            if isinstance(item, BaseException):
                raise item
            yield item  # type: ignore[misc]
    finally:
        thread.join(timeout=0.1)
