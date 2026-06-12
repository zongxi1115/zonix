from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from zonix.events import TextDelta, TextEnd, TextStart, ToolInputAvailable, ToolInputStart
from zonix.types import Message, ToolCall, Usage

OUTPUT_TOOL_NAME = "zonix_final_output"


def output_tool_schema(schema: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": OUTPUT_TOOL_NAME,
            "description": f"Return the final structured output for {name or 'this run'}.",
            "parameters": schema,
        },
    }


class ModelRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[Message]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    output_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ctx: Any = None
    state: Any = None
    task: Any = None


class ModelResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str | None = None
    output: Any = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    message_data: dict[str, Any] = Field(default_factory=dict)
    provider: str | None = None
    model: str | None = None
    request_data: dict[str, Any] = Field(default_factory=dict)
    raw_request: Any = None
    raw: Any = None
    response_id: str | None = None
    status: str | None = None
    finish_reason: str | None = None


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
