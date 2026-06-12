from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from zonix.events import (
    ReasoningDelta,
    TextDelta,
    TextEnd,
    TextStart,
    ToolInputAvailable,
    ToolInputDelta,
    ToolInputStart,
)
from zonix.exceptions import ModelError
from zonix.types import ToolCall, Usage

from .base import BaseChatModel, ModelRequest, ModelResponse, SupportsEmit


@dataclass
class OpenAI(BaseChatModel):
    model: str = "gpt-5.5"
    temperature: float | None = None
    api_key: str | None = None
    base_url: str | None = None
    native_structured_output: bool = True
    fallback_to_prompt_output: bool = True
    api: Literal["chat", "responses"] = "chat"
    reasoning_config: dict[str, Any] = field(default_factory=dict)
    text_config: dict[str, Any] = field(default_factory=dict)
    max_output_tokens: int | None = None
    previous_response_id: str | None = None
    include_values: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"openai:{self.model}"

    def responses(self) -> OpenAI:
        self.api = "responses"
        return self

    def chat(self) -> OpenAI:
        self.api = "chat"
        return self

    def reasoning(
        self,
        effort: str | None = None,
        *,
        summary: str | None = None,
        **extra: Any,
    ) -> OpenAI:
        self.api = "responses"
        if effort is not None:
            self.reasoning_config["effort"] = effort
        if summary is not None:
            self.reasoning_config["summary"] = summary
        self.reasoning_config.update(extra)
        return self

    def verbosity(self, value: str) -> OpenAI:
        self.api = "responses"
        self.text_config["verbosity"] = value
        return self

    def max_output(self, tokens: int) -> OpenAI:
        self.api = "responses"
        self.max_output_tokens = tokens
        return self

    def max_tokens(self, tokens: int) -> OpenAI:
        if self.api == "responses":
            self.max_output_tokens = tokens
        else:
            self.settings["max_completion_tokens"] = tokens
        return self

    def previous_response(self, response_id: str) -> OpenAI:
        self.api = "responses"
        self.previous_response_id = response_id
        return self

    def include(self, *items: str) -> OpenAI:
        self.api = "responses"
        for item in items:
            if item not in self.include_values:
                self.include_values.append(item)
        return self

    def with_settings(self, **settings: Any) -> OpenAI:
        self.settings.update(settings)
        return self

    def _messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            item: dict[str, Any] = {
                "role": message.role,
                "content": message.content or "",
                **({"name": message.name} if message.name and message.role != "tool" else {}),
                **({"tool_call_id": message.tool_call_id} if message.tool_call_id else {}),
            }
            if message.role == "assistant" and message.data.get("tool_calls"):
                item["content"] = message.content
                item["tool_calls"] = [
                    {
                        "id": call["call_id"],
                        "type": "function",
                        "function": {
                            "name": call["tool"],
                            "arguments": json.dumps(call.get("input", {}), ensure_ascii=False),
                        },
                    }
                    for call in message.data["tool_calls"]
                ]
            messages.append(item)
        return messages

    def _chat_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(request),
            **self.settings,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if (
            self.native_structured_output
            and request.output_schema is not None
            and not request.tools
            and "response_format" not in kwargs
        ):
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(request.output_name),
                    "strict": True,
                    "schema": _openai_strict_schema(request.output_schema),
                },
            }
        if request.tools:
            kwargs["tools"] = request.tools
            kwargs.setdefault("tool_choice", "auto")
        return kwargs

    def _responses_input(self, request: ModelRequest) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id or message.name or "tool",
                        "output": message.content or "",
                    }
                )
                continue
            if message.role == "assistant" and message.data.get("tool_calls"):
                if message.content:
                    items.append({"role": "assistant", "content": message.content})
                for call in message.data["tool_calls"]:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call["call_id"],
                            "name": call["tool"],
                            "arguments": json.dumps(call.get("input", {}), ensure_ascii=False),
                        }
                    )
                continue
            items.append({"role": message.role, "content": message.content or ""})
        return items

    def _responses_tools(self, tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for tool in tool_schemas:
            function = tool.get("function", {})
            tools.append(
                {
                    "type": "function",
                    "name": function.get("name"),
                    "description": function.get("description") or "",
                    "parameters": function.get("parameters") or {"type": "object"},
                }
            )
        return tools

    def _responses_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": self._responses_input(request),
            **self.settings,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if request.tools:
            kwargs["tools"] = self._responses_tools(request.tools)
            kwargs.setdefault("tool_choice", "auto")
        if request.output_schema is not None and "text" not in kwargs:
            text = dict(self.text_config)
            text["format"] = {
                "type": "json_schema",
                "name": _schema_name(request.output_name),
                "strict": True,
                "schema": _openai_strict_schema(request.output_schema),
            }
            kwargs["text"] = text
        elif self.text_config and "text" not in kwargs:
            kwargs["text"] = dict(self.text_config)
        if self.reasoning_config and "reasoning" not in kwargs:
            kwargs["reasoning"] = dict(self.reasoning_config)
        if self.max_output_tokens is not None and "max_output_tokens" not in kwargs:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if self.previous_response_id is not None and "previous_response_id" not in kwargs:
            kwargs["previous_response_id"] = self.previous_response_id
        if self.include_values and "include" not in kwargs:
            kwargs["include"] = list(self.include_values)
        return kwargs

    async def _client(self) -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ModelError("Install zonix[openai] to use the OpenAI adapter.") from exc
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self.api == "responses":
            return await self._complete_responses(request)
        return await self._complete_chat(request)

    async def _complete_chat(self, request: ModelRequest) -> ModelResponse:
        client = await self._client()
        kwargs = self._chat_kwargs(request)
        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception:
            if not self.fallback_to_prompt_output or "response_format" not in kwargs:
                raise
            kwargs = dict(kwargs)
            kwargs.pop("response_format", None)
            response = await client.chat.completions.create(**kwargs)
        choice_item = response.choices[0]
        choice = choice_item.message
        calls = _chat_tool_calls(choice.tool_calls or [])

        message_data = _message_data_from_chat(choice)
        usage = _usage_from_openai_chat(getattr(response, "usage", None), model_calls=1)
        return ModelResponse(
            text=choice.content or "",
            tool_calls=calls,
            usage=usage,
            message_data=message_data,
            provider="openai",
            model=getattr(response, "model", None) or self.model,
            request_data=_dump(kwargs),
            raw_request=_dump(kwargs),
            raw=_dump(response),
            response_id=getattr(response, "id", None),
            status="completed",
            finish_reason=getattr(choice_item, "finish_reason", None),
        )

    async def _complete_responses(self, request: ModelRequest) -> ModelResponse:
        client = await self._client()
        kwargs = self._responses_kwargs(request)
        response = await client.responses.create(**kwargs)
        return self._model_response_from_responses(response, kwargs)

    def _model_response_from_responses(
        self,
        response: Any,
        request_kwargs: dict[str, Any],
        *,
        raw_override: Any = None,
        streamed_text: str | None = None,
        streamed_calls: list[ToolCall] | None = None,
        streamed_reasoning: list[dict[str, str]] | None = None,
    ) -> ModelResponse:
        raw = _dump(response)
        text = (
            streamed_text
            if streamed_text is not None
            else getattr(response, "output_text", None)
        )
        if text is None:
            text = _responses_text(raw)
        calls = streamed_calls if streamed_calls is not None else _responses_tool_calls(raw)
        reasoning = (
            streamed_reasoning
            if streamed_reasoning is not None
            else _responses_reasoning(raw)
        )
        usage = _usage_from_openai_responses(getattr(response, "usage", None), raw, model_calls=1)
        message_data: dict[str, Any] = {
            "output": raw.get("output", []) if isinstance(raw, dict) else [],
        }
        if reasoning:
            message_data["reasoning"] = reasoning
        incomplete = raw.get("incomplete_details") if isinstance(raw, dict) else None
        if incomplete:
            message_data["incomplete_details"] = incomplete
        return ModelResponse(
            text=text or "",
            tool_calls=calls,
            usage=usage,
            message_data=message_data,
            provider="openai",
            model=(raw.get("model") if isinstance(raw, dict) else None) or self.model,
            request_data=_dump(request_kwargs),
            raw_request=_dump(request_kwargs),
            raw=raw_override if raw_override is not None else raw,
            response_id=raw.get("id") if isinstance(raw, dict) else getattr(response, "id", None),
            status=(
                raw.get("status") if isinstance(raw, dict) else getattr(response, "status", None)
            ),
            finish_reason=_responses_finish_reason(raw),
        )

    async def stream_complete(
        self,
        request: ModelRequest,
        emit: SupportsEmit,
        path: tuple[str, ...],
    ) -> ModelResponse:
        if self.api == "responses":
            return await self._stream_responses(request, emit, path)
        return await self._stream_chat(request, emit, path)

    async def _stream_chat(
        self,
        request: ModelRequest,
        emit: SupportsEmit,
        path: tuple[str, ...],
    ) -> ModelResponse:
        client = await self._client()
        kwargs = self._chat_kwargs(request)
        try:
            stream = await client.chat.completions.create(**kwargs, stream=True)
        except Exception:
            if not self.fallback_to_prompt_output or "response_format" not in kwargs:
                raise
            kwargs = dict(kwargs)
            kwargs.pop("response_format", None)
            stream = await client.chat.completions.create(**kwargs, stream=True)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        text_started = False
        tool_parts: dict[int, dict[str, Any]] = {}
        raw_chunks: list[Any] = []
        usage = Usage(model_calls=1)
        model_name = self.model
        response_id: str | None = None
        finish_reason: str | None = None

        async for chunk in stream:
            raw_chunks.append(_dump(chunk))
            if getattr(chunk, "id", None):
                response_id = chunk.id
            if getattr(chunk, "model", None):
                model_name = chunk.model
            if getattr(chunk, "usage", None) is not None:
                usage = _usage_from_openai_chat(chunk.usage, model_calls=1)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", finish_reason) or finish_reason
            delta = choice.delta
            reasoning_delta = _delta_reasoning(delta)
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
                await emit(ReasoningDelta(path, "reasoning_0", reasoning_delta))
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
            call = ToolCall(
                call_id=current["id"],
                tool=current["name"] or "unknown",
                input=_json_args(current["args"] or "{}"),
            )
            calls.append(call)
            await emit(ToolInputAvailable(path, call.call_id, call.tool, call.input))

        message_data: dict[str, Any] = {}
        if reasoning_parts:
            message_data["reasoning"] = [{"text": "".join(reasoning_parts)}]
        return ModelResponse(
            text="".join(text_parts),
            tool_calls=calls,
            usage=usage,
            message_data=message_data,
            provider="openai",
            model=model_name,
            request_data=_dump(kwargs),
            raw_request=_dump(kwargs),
            raw={"chunks": raw_chunks},
            response_id=response_id,
            status="completed",
            finish_reason=finish_reason,
        )

    async def _stream_responses(
        self,
        request: ModelRequest,
        emit: SupportsEmit,
        path: tuple[str, ...],
    ) -> ModelResponse:
        client = await self._client()
        kwargs = self._responses_kwargs(request)
        stream = await client.responses.create(**kwargs, stream=True)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: list[ToolCall] = []
        raw_events: list[Any] = []
        final_response: Any = None
        text_started = False

        async for event in stream:
            raw_event = _dump(event)
            raw_events.append(raw_event)
            event_type = getattr(event, "type", "") or (
                raw_event.get("type", "") if isinstance(raw_event, dict) else ""
            )
            if event_type.endswith("output_text.delta"):
                delta = getattr(event, "delta", None) or (
                    raw_event.get("delta") if isinstance(raw_event, dict) else None
                )
                if delta:
                    if not text_started:
                        text_started = True
                        await emit(TextStart(path, "text_0"))
                    text_parts.append(delta)
                    await emit(TextDelta(path, "text_0", delta))
                continue
            if "reasoning" in event_type and event_type.endswith(".delta"):
                delta = _event_text_delta(event, raw_event)
                if delta:
                    reasoning_parts.append(delta)
                    await emit(ReasoningDelta(path, "reasoning_0", delta))
                continue
            if event_type == "response.output_item.done":
                item = getattr(event, "item", None) or (
                    raw_event.get("item") if isinstance(raw_event, dict) else None
                )
                call = _responses_tool_call_from_item(_dump(item))
                if call is not None:
                    calls.append(call)
                    await emit(ToolInputStart(path, call.call_id, call.tool))
                    await emit(ToolInputAvailable(path, call.call_id, call.tool, call.input))
                continue
            if event_type in {"response.completed", "response.incomplete", "response.failed"}:
                final_response = getattr(event, "response", None) or (
                    raw_event.get("response") if isinstance(raw_event, dict) else None
                )

        if text_started:
            await emit(TextEnd(path, "text_0"))

        if final_response is not None:
            parsed = self._model_response_from_responses(
                final_response,
                kwargs,
                raw_override={"events": raw_events, "response": _dump(final_response)},
                streamed_text="".join(text_parts) if text_parts else None,
                streamed_calls=calls or None,
                streamed_reasoning=(
                    [{"text": "".join(reasoning_parts)}] if reasoning_parts else None
                ),
            )
            return parsed
        return ModelResponse(
            text="".join(text_parts),
            tool_calls=calls,
            usage=Usage(model_calls=1),
            message_data=(
                {"reasoning": [{"text": "".join(reasoning_parts)}]} if reasoning_parts else {}
            ),
            provider="openai",
            model=self.model,
            request_data=_dump(kwargs),
            raw_request=_dump(kwargs),
            raw={"events": raw_events},
            status="completed",
        )


def _schema_name(name: str | None) -> str:
    raw = name or "zonix_output"
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in raw)
    return cleaned[:64] or "zonix_output"


def _openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(schema))
    _close_objects(copied)
    return copied


def _close_objects(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            value.setdefault("additionalProperties", False)
        for child in value.values():
            _close_objects(child)
    elif isinstance(value, list):
        for child in value:
            _close_objects(child)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _dump(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_dump(child) for child in value]
    return value


def _json_args(raw_args: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw_args}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _chat_tool_calls(tool_calls: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for tool_call in tool_calls:
        calls.append(
            ToolCall(
                call_id=tool_call.id,
                tool=tool_call.function.name,
                input=_json_args(tool_call.function.arguments or "{}"),
            )
        )
    return calls


def _message_data_from_chat(choice: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    reasoning = _message_reasoning(choice)
    if reasoning:
        data["reasoning"] = [{"text": reasoning}]
    refusal = getattr(choice, "refusal", None)
    if refusal:
        data["refusal"] = refusal
    return data


def _message_reasoning(message: Any) -> str | None:
    for name in ("reasoning_content", "reasoning", "thinking"):
        value = getattr(message, name, None)
        if isinstance(value, str) and value:
            return value
    return None


def _delta_reasoning(delta: Any) -> str | None:
    for name in ("reasoning_content", "reasoning", "thinking"):
        value = getattr(delta, name, None)
        if isinstance(value, str) and value:
            return value
    return None


def _usage_from_openai_chat(usage_obj: Any, *, model_calls: int) -> Usage:
    usage = Usage(model_calls=model_calls)
    if usage_obj is None:
        return usage
    usage.input_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
    usage.output_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
    usage.total_tokens = getattr(usage_obj, "total_tokens", 0) or 0
    prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
    completion_details = getattr(usage_obj, "completion_tokens_details", None)
    if prompt_details is not None:
        usage.cached_input_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
    if completion_details is not None:
        usage.reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0
    usage.provider_details = _dump(usage_obj) if isinstance(_dump(usage_obj), dict) else {}
    return usage


def _usage_from_openai_responses(usage_obj: Any, raw: Any, *, model_calls: int) -> Usage:
    usage = Usage(model_calls=model_calls)
    raw_usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    usage.input_tokens = (
        getattr(usage_obj, "input_tokens", None) or raw_usage.get("input_tokens", 0) or 0
    )
    usage.output_tokens = (
        getattr(usage_obj, "output_tokens", None) or raw_usage.get("output_tokens", 0) or 0
    )
    usage.total_tokens = (
        getattr(usage_obj, "total_tokens", None) or raw_usage.get("total_tokens", 0) or 0
    )
    input_details = raw_usage.get("input_tokens_details") or {}
    output_details = raw_usage.get("output_tokens_details") or {}
    usage.cached_input_tokens = input_details.get("cached_tokens", 0) or 0
    usage.reasoning_tokens = output_details.get("reasoning_tokens", 0) or 0
    usage.provider_details = raw_usage if isinstance(raw_usage, dict) else {}
    return usage


def _responses_text(raw: Any) -> str:
    if not isinstance(raw, dict):
        return ""
    parts: list[str] = []
    for item in raw.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"}:
                parts.append(content.get("text", ""))
    return "".join(parts)


def _responses_tool_calls(raw: Any) -> list[ToolCall]:
    if not isinstance(raw, dict):
        return []
    calls: list[ToolCall] = []
    for item in raw.get("output", []) or []:
        call = _responses_tool_call_from_item(item)
        if call is not None:
            calls.append(call)
    return calls


def _responses_tool_call_from_item(item: Any) -> ToolCall | None:
    if not isinstance(item, dict) or item.get("type") != "function_call":
        return None
    return ToolCall(
        call_id=item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}",
        tool=item.get("name") or "unknown",
        input=_json_args(item.get("arguments") or "{}"),
    )


def _responses_reasoning(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, dict):
        return []
    items: list[dict[str, str]] = []
    for item in raw.get("output", []) or []:
        if item.get("type") != "reasoning":
            continue
        for summary in item.get("summary", []) or []:
            text = summary.get("text")
            if text:
                items.append({"text": text})
    return items


def _responses_finish_reason(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    incomplete = raw.get("incomplete_details")
    if isinstance(incomplete, dict):
        return incomplete.get("reason")
    return raw.get("status")


def _event_text_delta(event: Any, raw_event: Any) -> str | None:
    for name in ("delta", "text", "summary_text"):
        value = getattr(event, name, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(raw_event, dict):
        for name in ("delta", "text", "summary_text"):
            value = raw_event.get(name)
            if isinstance(value, str) and value:
                return value
    return None
