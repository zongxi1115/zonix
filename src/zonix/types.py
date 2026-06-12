from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")
Emit = Callable[[Any], Awaitable[None]]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0
    thinking_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    provider_details: dict[str, Any] = Field(default_factory=dict)

    def add(self, other: Usage | None) -> Usage:
        if other is None:
            return self
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.thinking_tokens += other.thinking_tokens
        self.model_calls += other.model_calls
        self.tool_calls += other.tool_calls
        self.provider_details = _merge_provider_details(
            self.provider_details,
            other.provider_details,
        )
        return self

    def __iadd__(self, other: Usage | None) -> Usage:
        return self.add(other)


def _merge_provider_details(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        current = merged.get(key)
        if isinstance(current, int | float) and isinstance(value, int | float):
            merged[key] = current + value
        elif isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_provider_details(current, value)
        elif key not in merged:
            merged[key] = value
        else:
            merged[key] = value
    return merged


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


MessageLike: TypeAlias = Message | dict[str, Any]


class ToolCall(BaseModel):
    call_id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex}")
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)


def system_message(content: str, **data: Any) -> Message:
    return Message(role="system", content=content, data=data)


def user_message(content: str, **data: Any) -> Message:
    return Message(role="user", content=content, data=data)


def assistant_message(content: str | None = None, **data: Any) -> Message:
    return Message(role="assistant", content=content, data=data)


def assistant_tool_call_message(
    *tool_calls: ToolCall | dict[str, Any],
    content: str | None = None,
) -> Message:
    calls = [
        call.model_dump(mode="json")
        if isinstance(call, ToolCall)
        else ToolCall.model_validate(call).model_dump(mode="json")
        for call in tool_calls
    ]
    return Message(role="assistant", content=content, data={"tool_calls": calls})


def tool_message(
    call_id: str,
    tool: str,
    output: Any,
    *,
    content: str | None = None,
) -> Message:
    from .serialization import to_jsonable

    return Message(
        role="tool",
        name=tool,
        tool_call_id=call_id,
        content=(
            content
            if content is not None
            else json.dumps(to_jsonable(output), ensure_ascii=False)
        ),
    )


def coerce_messages(messages: Sequence[MessageLike] | None) -> list[Message]:
    if messages is None:
        return []
    return [
        message if isinstance(message, Message) else Message.model_validate(message)
        for message in messages
    ]


class PendingApproval(BaseModel):
    call_id: str
    tool: str
    input: dict[str, Any]
    approval_key: str


class Route(BaseModel):
    next: str | None = None
    done: bool = False
    input: Any = None
    reason: str | None = None


class Span(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    path: tuple[str, ...] = Field(default_factory=tuple)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    status: Literal["running", "ok", "error", "paused"] = "running"
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    children: list[Span] = Field(default_factory=list)

    def child(self, name: str, **attributes: Any) -> Span:
        span = Span(name=name, path=(*self.path, name), attributes=attributes)
        self.children.append(span)
        return span

    def record(self, event: Any) -> None:
        from .serialization import to_jsonable

        self.events.append(to_jsonable(event))

    def finish(self, status: Literal["ok", "error", "paused"] = "ok") -> None:
        self.status = status
        self.ended_at = datetime.now(UTC)


@dataclass
class RunState:
    ctx: Any
    usage: Usage
    messages: list[Message]
    scratch: dict[str, Any]
    trace: Span
    bus: Any
    session: Any = None
    message_history: list[Message] = field(default_factory=list)
    model_calls: list[ModelCall] = field(default_factory=list)
    approvals: dict[str, Any] = field(default_factory=dict)
    extra: str | None = None
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex}")
    stop_requested: bool = False
    stop_output: Any = None

    @property
    def path(self) -> tuple[str, ...]:
        return self.trace.path

    def scoped(self, name: str, **attributes: Any) -> RunState:
        return replace(self, messages=[], trace=self.trace.child(name, **attributes))

    def request_stop(self, output: Any = None) -> None:
        self.stop_requested = True
        self.stop_output = output


class Node(Protocol):
    name: str
    in_type: type[Any]
    out_type: type[Any]

    async def invoke(self, x: Any, st: RunState) -> Any:
        ...


class ModelCall(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: str | None = None
    model: str | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    raw_request: Any = None
    raw_response: Any = None
    usage: Usage = Field(default_factory=Usage)
    response_id: str | None = None
    status: str | None = None
    finish_reason: str | None = None
    message_data: dict[str, Any] = Field(default_factory=dict)


@dataclass
class RunResult:
    run_id: str
    output: Any
    usage: Usage
    trace: Span
    messages: list[Message]
    scratch: dict[str, Any]
    model_calls: list[ModelCall] = field(default_factory=list)
    status: Literal["done", "paused", "error"] = "done"
    pending: PendingApproval | None = None
    error: str | None = None
    _resume: Callable[[bool, dict[str, Any] | None], Awaitable[RunResult]] | None = None

    @property
    def paused(self) -> bool:
        return self.status == "paused"

    @property
    def last_model_call(self) -> ModelCall | None:
        return self.model_calls[-1] if self.model_calls else None

    @property
    def last_response(self) -> Any:
        call = self.last_model_call
        return None if call is None else call.raw_response

    def dump(self) -> dict[str, Any]:
        from .serialization import to_jsonable

        return {
            "run_id": self.run_id,
            "status": self.status,
            "output": to_jsonable(self.output),
            "usage": to_jsonable(self.usage),
            "trace": to_jsonable(self.trace),
            "messages": to_jsonable(self.messages),
            "model_calls": to_jsonable(self.model_calls),
            "scratch": to_jsonable(self.scratch),
            "pending": to_jsonable(self.pending),
            "error": self.error,
        }

    async def resume(
        self,
        approve: bool = True,
        input: dict[str, Any] | None = None,
    ) -> RunResult:
        if not self.paused or self.pending is None:
            raise RuntimeError("Only paused runs can be resumed.")
        if self._resume is None:
            raise RuntimeError("This run was loaded from a dump and has no live runner.")
        return await self._resume(approve, input)
