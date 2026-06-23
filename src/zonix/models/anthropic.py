from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from zonix.events import (
    ReasoningDelta,
    TextDelta,
    TextEnd,
    TextStart,
    ToolInputAvailable,
    ToolInputDelta,
    ToolInputStart,
)
from zonix.content import (
    content_blocks,
    content_text,
    image_media_type,
    image_source,
    split_data_url,
)
from zonix.exceptions import ModelError
from zonix.types import ToolCall, Usage

from .base import (
    OUTPUT_TOOL_NAME,
    BaseChatModel,
    ModelRequest,
    ModelResponse,
    SupportsEmit,
    output_tool_schema,
)


@dataclass
class Anthropic(BaseChatModel):
    model: str = "claude-sonnet-4-6"
    temperature: float | None = None
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 4096
    thinking_config: dict[str, Any] = field(default_factory=dict)
    effort_value: str | None = None
    service_tier: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    container: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"anthropic:{self.model}"

    def thinking(
        self,
        type: str = "adaptive",
        *,
        budget_tokens: int | None = None,
        display: str | None = None,
        **extra: Any,
    ) -> Anthropic:
        self.thinking_config["type"] = type
        if budget_tokens is not None:
            self.thinking_config["budget_tokens"] = budget_tokens
        if display is not None:
            self.thinking_config["display"] = display
        self.thinking_config.update(extra)
        return self

    def effort(self, value: str) -> Anthropic:
        self.effort_value = value
        return self

    def max_output(self, tokens: int) -> Anthropic:
        self.max_tokens = tokens
        return self

    def tier(self, service_tier: str) -> Anthropic:
        self.service_tier = service_tier
        return self

    def with_metadata(self, **metadata: Any) -> Anthropic:
        self.metadata.update(metadata)
        return self

    def with_settings(self, **settings: Any) -> Anthropic:
        self.settings.update(settings)
        return self

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
                text = content_text(message.content)
                if text:
                    system_parts.append(text)
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
            if message.role == "assistant" and message.data.get("anthropic_content"):
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.data["anthropic_content"],
                    }
                )
                continue
            if message.role == "assistant" and message.data.get("tool_calls"):
                content: list[dict[str, Any]] = []
                text = content_text(message.content)
                if text:
                    content.append({"type": "text", "text": text})
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
            messages.append({"role": role, "content": _anthropic_content(message.content)})

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
        if self.thinking_config and "thinking" not in kwargs:
            kwargs["thinking"] = dict(self.thinking_config)
        if self.effort_value is not None and "effort" not in kwargs:
            kwargs["effort"] = self.effort_value
        if self.service_tier is not None and "service_tier" not in kwargs:
            kwargs["service_tier"] = self.service_tier
        if self.metadata and "metadata" not in kwargs:
            kwargs["metadata"] = dict(self.metadata)
        if self.container is not None and "container" not in kwargs:
            kwargs["container"] = self.container
        tools = self._tools(request.tools)
        if request.output_schema is not None:
            tools.extend(
                self._tools([output_tool_schema(request.output_schema, request.output_name)])
            )
        if tools:
            kwargs["tools"] = tools
            if request.output_schema is not None and not request.tools:
                kwargs.setdefault("tool_choice", {"type": "tool", "name": OUTPUT_TOOL_NAME})
        return kwargs

    async def complete(self, request: ModelRequest) -> ModelResponse:
        client = await self._client()
        kwargs = self._kwargs(request)
        response = await client.messages.create(**kwargs)
        return self._response_from_message(response, kwargs)

    def _response_from_message(
        self,
        response: Any,
        request_kwargs: dict[str, Any],
        *,
        raw_override: Any = None,
        streamed_text: str | None = None,
        streamed_reasoning: str | None = None,
    ) -> ModelResponse:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: list[ToolCall] = []
        output: Any = None
        content_blocks: list[dict[str, Any]] = []
        redacted_thinking: list[dict[str, Any]] = []

        for block in response.content:
            block_type = getattr(block, "type", None)
            block_data = _dump(block)
            if isinstance(block_data, dict):
                content_blocks.append(block_data)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "thinking":
                thinking = getattr(block, "thinking", "") or ""
                if thinking:
                    reasoning_parts.append(thinking)
            elif block_type == "redacted_thinking":
                if isinstance(block_data, dict):
                    redacted_thinking.append(block_data)
            elif block_type == "tool_use":
                if block.name == OUTPUT_TOOL_NAME:
                    output = getattr(block, "input", {}) or {}
                    continue
                calls.append(
                    ToolCall(
                        call_id=block.id,
                        tool=block.name,
                        input=getattr(block, "input", {}) or {},
                    )
                )

        if streamed_text is not None:
            text_parts = [streamed_text]
        if streamed_reasoning is not None:
            reasoning_parts = [streamed_reasoning]

        message_data: dict[str, Any] = {
            "anthropic_content": content_blocks,
            "id": getattr(response, "id", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "stop_sequence": getattr(response, "stop_sequence", None),
            "service_tier": getattr(response, "service_tier", None),
        }
        if reasoning_parts:
            message_data["reasoning"] = [{"text": "".join(reasoning_parts)}]
        if redacted_thinking:
            message_data["redacted_thinking"] = redacted_thinking

        return ModelResponse(
            text="".join(text_parts),
            output=output,
            tool_calls=calls,
            usage=_usage_from_anthropic(getattr(response, "usage", None), model_calls=1),
            message_data=message_data,
            provider="anthropic",
            model=getattr(response, "model", None) or self.model,
            request_data=_dump(request_kwargs),
            raw_request=_dump(request_kwargs),
            raw=raw_override if raw_override is not None else _dump(response),
            response_id=getattr(response, "id", None),
            status="completed",
            finish_reason=getattr(response, "stop_reason", None),
        )

    async def stream_complete(
        self,
        request: ModelRequest,
        emit: SupportsEmit,
        path: tuple[str, ...],
    ) -> ModelResponse:
        client = await self._client()
        kwargs = self._kwargs(request)
        stream_factory = getattr(client.messages, "stream", None)
        if stream_factory is None:
            return await super().stream_complete(request, emit, path)

        raw_events: list[Any] = []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        text_started = False
        block_types: dict[int, str] = {}
        tool_blocks: dict[int, dict[str, Any]] = {}
        final_message: Any = None

        async with stream_factory(**kwargs) as stream:
            async for event in stream:
                raw_event = _dump(event)
                raw_events.append(raw_event)
                event_type = getattr(event, "type", "") or (
                    raw_event.get("type", "") if isinstance(raw_event, dict) else ""
                )
                index = getattr(event, "index", None)
                if event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    block_type = getattr(block, "type", None)
                    if index is not None and block_type:
                        block_types[index] = block_type
                    if block_type == "text" and not text_started:
                        text_started = True
                        await emit(TextStart(path, "text_0"))
                    elif block_type == "tool_use":
                        tool_blocks[index or 0] = {
                            "id": getattr(block, "id", None),
                            "name": getattr(block, "name", None),
                            "args": "",
                        }
                        if getattr(block, "id", None) and getattr(block, "name", None):
                            await emit(ToolInputStart(path, block.id, block.name))
                    continue
                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        text = getattr(delta, "text", "") or ""
                        if text:
                            if not text_started:
                                text_started = True
                                await emit(TextStart(path, "text_0"))
                            text_parts.append(text)
                            await emit(TextDelta(path, "text_0", text))
                    elif delta_type == "thinking_delta":
                        thinking = getattr(delta, "thinking", "") or ""
                        if thinking:
                            reasoning_parts.append(thinking)
                            await emit(ReasoningDelta(path, "reasoning_0", thinking))
                    elif delta_type == "input_json_delta":
                        partial = getattr(delta, "partial_json", "") or ""
                        current = tool_blocks.setdefault(
                            index or 0,
                            {"id": None, "name": None, "args": ""},
                        )
                        current["args"] += partial
                        if current.get("id"):
                            await emit(ToolInputDelta(path, current["id"], partial))
                    continue

            final_message = stream.get_final_message()
            if inspect.isawaitable(final_message):
                final_message = await final_message

        if text_started:
            await emit(TextEnd(path, "text_0"))

        parsed = self._response_from_message(
            final_message,
            kwargs,
            raw_override={"events": raw_events, "message": _dump(final_message)},
            streamed_text="".join(text_parts) if text_parts else None,
            streamed_reasoning="".join(reasoning_parts) if reasoning_parts else None,
        )
        for call in parsed.tool_calls:
            await emit(ToolInputAvailable(path, call.call_id, call.tool, call.input))
        return parsed


def _anthropic_content(content: Any) -> str | list[dict[str, Any]]:
    if content is None or isinstance(content, str):
        return content or ""

    blocks: list[dict[str, Any]] = []
    for block in content_blocks(content):
        block_type = str(block.get("type") or "").strip()
        if block_type in {"text", "input_text"}:
            text = block.get("text", block.get("content"))
            if isinstance(text, str) and text:
                blocks.append({"type": "text", "text": text})
            continue
        if block_type in {"image", "image_url", "input_image"}:
            image = _anthropic_image_block(block)
            if image:
                blocks.append(image)
    return blocks or content_text(content)


def _anthropic_image_block(block: dict[str, Any]) -> dict[str, Any] | None:
    source = image_source(block)
    if not source:
        return None

    media_type, data = split_data_url(source)
    if data is not None:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type(block) or media_type or "image/png",
                "data": data,
            },
        }

    if source.startswith(("http://", "https://")):
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": source,
            },
        }

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": image_media_type(block) or "image/png",
            "data": source,
        },
    }


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _dump(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_dump(child) for child in value]
    return value


def _usage_from_anthropic(usage_obj: Any, *, model_calls: int) -> Usage:
    usage = Usage(model_calls=model_calls)
    if usage_obj is None:
        return usage
    usage.input_tokens = getattr(usage_obj, "input_tokens", 0) or 0
    usage.output_tokens = getattr(usage_obj, "output_tokens", 0) or 0
    usage.cache_creation_input_tokens = getattr(usage_obj, "cache_creation_input_tokens", 0) or 0
    usage.cache_read_input_tokens = getattr(usage_obj, "cache_read_input_tokens", 0) or 0
    output_details = getattr(usage_obj, "output_tokens_details", None)
    if output_details is not None:
        usage.thinking_tokens = getattr(output_details, "thinking_tokens", 0) or 0
        usage.reasoning_tokens = usage.thinking_tokens
    usage.total_tokens = (
        usage.input_tokens
        + usage.output_tokens
        + usage.cache_creation_input_tokens
        + usage.cache_read_input_tokens
    )
    raw = _dump(usage_obj)
    usage.provider_details = raw if isinstance(raw, dict) else {}
    return usage
