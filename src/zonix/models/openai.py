from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from zonix.events import TextDelta, TextEnd, TextStart, ToolInputAvailable, ToolInputDelta, ToolInputStart
from zonix.exceptions import ModelError
from zonix.types import ToolCall, Usage

from .base import BaseChatModel, ModelRequest, ModelResponse, SupportsEmit


@dataclass
class OpenAI(BaseChatModel):
    model: str = "gpt-5.2"
    temperature: float | None = None
    api_key: str | None = None
    base_url: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"openai:{self.model}"

    def _messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        return [
            {
                "role": message.role,
                "content": message.content or "",
                **({"name": message.name} if message.name and message.role != "tool" else {}),
                **({"tool_call_id": message.tool_call_id} if message.tool_call_id else {}),
            }
            for message in request.messages
        ]

    def _kwargs(self, request: ModelRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(request),
            **self.settings,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    async def _client(self) -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ModelError("Install zonix[openai] to use the OpenAI adapter.") from exc
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        client = await self._client()
        response = await client.chat.completions.create(**self._kwargs(request))
        choice = response.choices[0].message
        calls: list[ToolCall] = []
        for tool_call in choice.tool_calls or []:
            raw_args = tool_call.function.arguments or "{}"
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed = {"_raw": raw_args}
            calls.append(
                ToolCall(
                    call_id=tool_call.id,
                    tool=tool_call.function.name,
                    input=parsed,
                )
            )

        usage = Usage(model_calls=1)
        if response.usage is not None:
            usage.input_tokens = response.usage.prompt_tokens or 0
            usage.output_tokens = response.usage.completion_tokens or 0
            usage.total_tokens = response.usage.total_tokens or 0
        return ModelResponse(
            text=choice.content or "",
            tool_calls=calls,
            usage=usage,
            raw=response.model_dump(mode="json") if hasattr(response, "model_dump") else response,
        )

    async def stream_complete(
        self,
        request: ModelRequest,
        emit: SupportsEmit,
        path: tuple[str, ...],
    ) -> ModelResponse:
        client = await self._client()
        stream = await client.chat.completions.create(**self._kwargs(request), stream=True)
        text_parts: list[str] = []
        text_started = False
        tool_parts: dict[int, dict[str, Any]] = {}

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                if not text_started:
                    text_started = True
                    await emit(TextStart(path, "text_0"))
                text_parts.append(delta.content)
                await emit(TextDelta(path, "text_0", delta.content))

            for delta_call in delta.tool_calls or []:
                index = delta_call.index
                current = tool_parts.setdefault(
                    index,
                    {
                        "id": delta_call.id or f"call_{uuid.uuid4().hex}",
                        "name": None,
                        "args": "",
                        "started": False,
                    },
                )
                if delta_call.id:
                    current["id"] = delta_call.id
                if delta_call.function and delta_call.function.name:
                    current["name"] = delta_call.function.name
                if current["name"] and not current["started"]:
                    current["started"] = True
                    await emit(ToolInputStart(path, current["id"], current["name"]))
                if delta_call.function and delta_call.function.arguments:
                    current["args"] += delta_call.function.arguments
                    await emit(ToolInputDelta(path, current["id"], delta_call.function.arguments))

        if text_started:
            await emit(TextEnd(path, "text_0"))

        calls: list[ToolCall] = []
        for current in tool_parts.values():
            raw_args = current["args"] or "{}"
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed = {"_raw": raw_args}
            name = current["name"] or "unknown"
            call = ToolCall(call_id=current["id"], tool=name, input=parsed)
            calls.append(call)
            await emit(ToolInputAvailable(path, call.call_id, call.tool, call.input))

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=calls,
            usage=Usage(model_calls=1),
        )
