from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .serialization import to_jsonable
from .types import Usage


@dataclass(frozen=True)
class Event:
    path: tuple[str, ...]

    @property
    def type(self) -> str:
        name = type(self).__name__
        chars: list[str] = []
        for i, char in enumerate(name):
            if char.isupper() and i:
                chars.append("-")
            chars.append(char.lower())
        return "".join(chars)

    def dump(self) -> dict[str, Any]:
        data = to_jsonable(self)
        if isinstance(data, dict):
            data["type"] = self.type
            return data
        return {"type": self.type, "value": data}


@dataclass(frozen=True)
class NodeStart(Event):
    name: str


@dataclass(frozen=True)
class NodeEnd(Event):
    name: str
    status: str = "ok"


@dataclass(frozen=True)
class TextStart(Event):
    id: str


@dataclass(frozen=True)
class TextDelta(Event):
    id: str
    delta: str


@dataclass(frozen=True)
class TextEnd(Event):
    id: str


@dataclass(frozen=True)
class ReasoningDelta(Event):
    id: str
    delta: str


@dataclass(frozen=True)
class ToolInputStart(Event):
    call_id: str
    tool: str


@dataclass(frozen=True)
class ToolInputDelta(Event):
    call_id: str
    delta: str


@dataclass(frozen=True)
class ToolInputAvailable(Event):
    call_id: str
    tool: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolOutputAvailable(Event):
    call_id: str
    output: Any


@dataclass(frozen=True)
class ApprovalRequired(Event):
    call_id: str
    tool: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ErrorEvent(Event):
    message: str
    error_type: str = "error"


@dataclass(frozen=True)
class Finish(Event):
    output: Any
    usage: Usage
