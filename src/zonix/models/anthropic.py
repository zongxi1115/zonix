from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zonix.exceptions import ModelError
from zonix.types import ToolCall, Usage

from .base import OUTPUT_TOOL_NAME, BaseChatModel, ModelRequest, ModelResponse, SupportsEmit, output_tool_schema


@dataclass
class Anthropic(BaseChatModel):
    model: str = "claude-4.5-sonnet"
    temperature: float | None = None
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 4096
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"anthropic:{self.model}"

    async def _client(self) -> Any:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ModelError("Install zonix[anthropic] to use the Anthropic adapter.") from exc
        return AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)

    def _messages(self, request: ModelRequest) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []

        for message in request.messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue
            if message.role == "tool":
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id or message.name or "tool",
                                "content": message.content or "",
                            }
                        ],
                    }
                )
                continue
            if message.role == "assistant" and message.data.get("tool_calls"):
                content: list[dict[str, Any]] = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                for call in message.data["tool_calls"]:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call["call_id"],
                            "name": call["tool"],
                            "input": call.get("input", {}),
                        }
                    )
                messages.append({"role": "assistant", "content": content})
                continue
            role = "assistant" if message.role == "assistant" else "user"
            messages.append({"role": role, "content": message.content or ""})

        return "\n\n".join(system_parts) or None, messages

    def _tools(self, tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for tool in tool_schemas:
            function = tool.get("function", {})
            tools.append(
                {
                    "name": function.get("name"),
                    "description": function.get("description") or "",
                    "input_schema": function.get("parameters") or {"type": "object"},
                }
            )
        return tools

    def _kwargs(self, request: ModelRequest) -> dict[str, Any]:
        system, messages = self._messages(request)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            **self.settings,
        }
        if system:
            kwargs["system"] = system
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        tools = self._tools(request.tools)
        if request.output_schema is not None:
            tools.extend(self._tools([output_tool_schema(request.output_schema, request.output_name)]))
        if tools:
            kwargs["tools"] = tools
            if request.output_schema is not None and not request.tools:
                kwargs.setdefault("tool_choice", {"type": "tool", "name": OUTPUT_TOOL_NAME})
        return kwargs

    async def complete(self, request: ModelRequest) -> ModelResponse:
        client = await self._client()
        response = await client.messages.create(**self._kwargs(request))

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        output: Any = None
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                if getattr(block, "name") == OUTPUT_TOOL_NAME:
                    output = getattr(block, "input", {}) or {}
                    continue
                calls.append(
                    ToolCall(
                        call_id=getattr(block, "id"),
                        tool=getattr(block, "name"),
                        input=getattr(block, "input", {}) or {},
                    )
                )

        usage = Usage(model_calls=1)
        if getattr(response, "usage", None) is not None:
            usage.input_tokens = getattr(response.usage, "input_tokens", 0) or 0
            usage.output_tokens = getattr(response.usage, "output_tokens", 0) or 0
            usage.total_tokens = usage.input_tokens + usage.output_tokens

        return ModelResponse(
            text="".join(text_parts),
            output=output,
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
        # Anthropic-compatible providers differ in streaming details. The base
        # fallback still emits typed Zonix events while keeping complete()
        # as the single compatibility surface.
        return await super().stream_complete(request, emit, path)
