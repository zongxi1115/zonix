from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

from .events import ErrorEvent, Finish, NodeEnd, NodeStart
from .exceptions import RunPaused
from .types import Message, RunResult, RunState, Span, Usage


class EventBus:
    def __init__(self, emit: Callable[[Any], Awaitable[None]] | None = None) -> None:
        self._emit = emit

    async def publish(self, event: Any) -> None:
        if self._emit is not None:
            await self._emit(event)


async def run_node(
    node: Any,
    task: Any,
    *,
    ctx: Any = None,
    session: Any = None,
    extra: str | None = None,
    approvals: dict[str, Any] | None = None,
    run_id: str | None = None,
    emit: Callable[[Any], Awaitable[None]] | None = None,
) -> RunResult:
    bus = EventBus(emit)
    trace = Span(name=node.name, path=(node.name,))
    state = RunState(
        ctx=ctx,
        usage=Usage(),
        messages=[],
        scratch={},
        trace=trace,
        bus=bus,
        session=session,
        approvals=approvals or {},
        extra=extra,
        run_id=run_id or trace.attributes.get("run_id") or "",
    )
    if not state.run_id:
        state.run_id = f"run_{id(state):x}"

    await bus.publish(NodeStart(state.path, node.name))
    status = "done"
    output: Any = None
    pending = None
    error = None

    async def resume(approve: bool, approved_input: dict[str, Any] | None) -> RunResult:
        if pending is None:
            raise RuntimeError("No pending approval to resume.")
        next_approvals = dict(approvals or {})
        if approve:
            next_approvals[pending.call_id] = approved_input or True
            next_approvals[pending.approval_key] = approved_input or True
        else:
            next_approvals[pending.call_id] = False
            next_approvals[pending.approval_key] = False
        return await run_node(
            node,
            task,
            ctx=ctx,
            session=session,
            extra=extra,
            approvals=next_approvals,
            run_id=state.run_id,
            emit=emit,
        )

    try:
        output = await node.invoke(task, state)
        trace.finish("ok")
        await bus.publish(NodeEnd(state.path, node.name, "ok"))
    except RunPaused as pause:
        status = "paused"
        pending = pause.pending
        trace.finish("paused")
        await bus.publish(NodeEnd(state.path, node.name, "paused"))
    except Exception as exc:
        status = "error"
        error = str(exc)
        trace.finish("error")
        await bus.publish(ErrorEvent(state.path, str(exc), type(exc).__name__))
        await bus.publish(NodeEnd(state.path, node.name, "error"))
        raise

    result = RunResult(
        run_id=state.run_id,
        output=output,
        usage=state.usage,
        trace=trace,
        messages=list(state.messages),
        scratch=dict(state.scratch),
        status=status,
        pending=pending,
        error=error,
        _resume=resume if pending is not None else None,
    )
    await bus.publish(Finish(state.path, result.output, result.usage))
    return result


async def stream_node(
    node: Any,
    task: Any,
    *,
    ctx: Any = None,
    session: Any = None,
    extra: str | None = None,
    approvals: dict[str, Any] | None = None,
) -> AsyncIterator[Any]:
    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def emit(event: Any) -> None:
        await queue.put(event)

    async def worker() -> None:
        try:
            await run_node(
                node,
                task,
                ctx=ctx,
                session=session,
                extra=extra,
                approvals=approvals,
                emit=emit,
            )
        except Exception:
            pass

    task_obj = asyncio.create_task(worker())
    try:
        while True:
            event = await queue.get()
            yield event
            if isinstance(event, Finish):
                break
            if isinstance(event, ErrorEvent) and task_obj.done():
                break
    finally:
        await task_obj


def append_message(messages: list[Message], role: str, content: str, **data: Any) -> None:
    messages.append(Message(role=role, content=content, **data))
