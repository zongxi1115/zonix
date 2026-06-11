from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field


T = TypeVar("T")
Emit = Callable[[Any], Awaitable[None]]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0

    def add(self, other: Usage | None) -> Usage:
        if other is None:
            return self
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.model_calls += other.model_calls
        self.tool_calls += other.tool_calls
        return self

    def __iadd__(self, other: Usage | None) -> Usage:
        return self.add(other)


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    call_id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex}")
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)


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
    approvals: dict[str, Any] = field(default_factory=dict)
    extra: str | None = None
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex}")

    @property
    def path(self) -> tuple[str, ...]:
        return self.trace.path

    def scoped(self, name: str, **attributes: Any) -> RunState:
        return replace(self, messages=[], trace=self.trace.child(name, **attributes))


class Node(Protocol):
    name: str
    in_type: type[Any]
    out_type: type[Any]

    async def invoke(self, x: Any, st: RunState) -> Any:
        ...


@dataclass
class RunResult:
    run_id: str
    output: Any
    usage: Usage
    trace: Span
    messages: list[Message]
    scratch: dict[str, Any]
    status: Literal["done", "paused", "error"] = "done"
    pending: PendingApproval | None = None
    error: str | None = None
    _resume: Callable[[bool, dict[str, Any] | None], Awaitable[RunResult]] | None = None

    @property
    def paused(self) -> bool:
        return self.status == "paused"

    def dump(self) -> dict[str, Any]:
        from .serialization import to_jsonable

        return {
            "run_id": self.run_id,
            "status": self.status,
            "output": to_jsonable(self.output),
            "usage": to_jsonable(self.usage),
            "trace": to_jsonable(self.trace),
            "messages": to_jsonable(self.messages),
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
