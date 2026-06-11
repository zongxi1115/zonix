from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from zonix.events import TextDelta, TextEnd, TextStart, ToolInputAvailable, ToolInputStart
from zonix.types import Message, ToolCall, Usage


class ModelRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[Message]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    output_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str | None = None
    output: Any = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    raw: Any = None


class SupportsEmit(Protocol):
    async def __call__(self, event: Any) -> None:
        ...


class BaseChatModel:
    name: str = ""
    settings: dict[str, Any]

    def __init__(self, name: str = "", **settings: Any) -> None:
        self.name = name
        self.settings = settings

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    async def stream_complete(
        self,
        request: ModelRequest,
        emit: SupportsEmit,
        path: tuple[str, ...],
    ) -> ModelResponse:
        response = await self.complete(request)
        text = response.text or ""
        if text:
            text_id = "text_0"
            await emit(TextStart(path, text_id))
            await emit(TextDelta(path, text_id, text))
            await emit(TextEnd(path, text_id))
        for call in response.tool_calls:
            await emit(ToolInputStart(path, call.call_id, call.tool))
            await emit(ToolInputAvailable(path, call.call_id, call.tool, call.input))
        return response
