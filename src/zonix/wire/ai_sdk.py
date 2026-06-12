from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from zonix.events import (
    ApprovalRequired,
    ErrorEvent,
    Event,
    Finish,
    ReasoningDelta,
    TextDelta,
    TextEnd,
    TextStart,
    ToolInputAvailable,
    ToolInputDelta,
    ToolInputStart,
    ToolOutputAvailable,
)
from zonix.serialization import to_jsonable


def sse(part: dict[str, Any]) -> str:
    return f"data: {json.dumps(to_jsonable(part), ensure_ascii=False)}\n\n"


def done_marker() -> str:
    return "data: [DONE]\n\n"


def event_to_part(event: Event) -> dict[str, Any] | None:
    if isinstance(event, TextStart):
        return {"type": "text-start", "id": event.id}
    if isinstance(event, TextDelta):
        return {"type": "text-delta", "id": event.id, "delta": event.delta}
    if isinstance(event, TextEnd):
        return {"type": "text-end", "id": event.id}
    if isinstance(event, ReasoningDelta):
        return {"type": "reasoning-delta", "id": event.id, "delta": event.delta}
    if isinstance(event, ToolInputStart):
        return {
            "type": "tool-input-start",
            "toolCallId": event.call_id,
            "toolName": event.tool,
        }
    if isinstance(event, ToolInputDelta):
        return {
            "type": "tool-input-delta",
            "toolCallId": event.call_id,
            "inputTextDelta": event.delta,
        }
    if isinstance(event, ToolInputAvailable):
        return {
            "type": "tool-input-available",
            "toolCallId": event.call_id,
            "toolName": event.tool,
            "input": event.input,
        }
    if isinstance(event, ToolOutputAvailable):
        return {
            "type": "tool-output-available",
            "toolCallId": event.call_id,
            "output": event.output,
        }
    if isinstance(event, ApprovalRequired):
        return {
            "type": "data-approval-required",
            "id": event.call_id,
            "data": {"toolName": event.tool, "input": event.input},
        }
    if isinstance(event, ErrorEvent):
        return {"type": "error", "errorText": event.message}
    if isinstance(event, Finish):
        return {"type": "finish", "usage": event.usage}
    return None


async def to_ai_sdk(
    stream: AsyncIterator[Event],
    *,
    message_id: str | None = None,
) -> AsyncIterator[str]:
    yield sse({"type": "start", "messageId": message_id or f"msg_{uuid.uuid4().hex}"})
    async for event in stream:
        part = event_to_part(event)
        if part is not None:
            yield sse(part)
    yield done_marker()
