from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from zonix.events import ReasoningDelta, TextDelta, TextEnd, TextStart
from zonix.exceptions import ModelError
from zonix.types import ToolCall, Usage

from .base import BaseChatModel, ModelRequest, ModelResponse, SupportsEmit


@dataclass
class Gemini(BaseChatModel):
    model: str = "gemini-3-pro"
    api_key: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    thinking_config: dict[str, Any] = field(default_factory=dict)
    response_mime_type: str | None = None
    safety_settings: list[dict[str, Any]] = field(default_factory=list)
    tool_config: dict[str, Any] = field(default_factory=dict)
    cached_content: str | None = None
    system_instruction: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"gemini:{self.model}"

    def thinking_budget(self, tokens: int) -> Gemini:
        self.thinking_config["thinking_budget"] = tokens
        return self

    def include_thoughts(self, enabled: bool = True) -> Gemini:
        self.thinking_config["include_thoughts"] = enabled
        return self

    def max_output(self, tokens: int) -> Gemini:
        self.max_output_tokens = tokens
        return self

    def json_mode(self) -> Gemini:
        self.response_mime_type = "application/json"
        return self

    def safety(self, *settings: dict[str, Any]) -> Gemini:
        self.safety_settings.extend(settings)
        return self

    def with_tool_config(self, **config: Any) -> Gemini:
        self.tool_config.update(config)
        return self

    def with_settings(self, **settings: Any) -> Gemini:
        self.settings.update(settings)
        return self

    async def _client(self) -> Any:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ModelError("Install zonix[gemini] to use the Gemini adapter.") from exc
        return genai.Client(api_key=self.api_key)

    def _contents(self, request: ModelRequest) -> tuple[list[dict[str, Any]], str | None]:
        contents: list[dict[str, Any]] = []
        system_parts: list[str] = []
        for message in request.messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue
            role = "model" if message.role == "assistant" else "user"
            if message.role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": message.name or "tool",
                                    "response": {
                                        "call_id": message.tool_call_id,
                                        "content": message.content or "",
                                    },
                                }
                            }
                        ],
                    }
                )
                continue
            contents.append({"role": role, "parts": [{"text": message.content or ""}]})
        return contents, "\n\n".join(system_parts) or None

    def _tools(self, tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declarations: list[dict[str, Any]] = []
        for tool in tool_schemas:
            function = tool.get("function", {})
            declarations.append(
                {
                    "name": function.get("name"),
                    "description": function.get("description") or "",
                    "parameters": function.get("parameters") or {"type": "object"},
                }
            )
        return [{"function_declarations": declarations}] if declarations else []

    def _config_data(self, request: ModelRequest, request_system: str | None) -> dict[str, Any]:
        config = dict(self.settings)
        if self.temperature is not None and "temperature" not in config:
            config["temperature"] = self.temperature
        if self.max_output_tokens is not None and "max_output_tokens" not in config:
            config["max_output_tokens"] = self.max_output_tokens
        if self.thinking_config and "thinking_config" not in config:
            config["thinking_config"] = dict(self.thinking_config)
        if self.response_mime_type is not None and "response_mime_type" not in config:
            config["response_mime_type"] = self.response_mime_type
        if request.output_schema is not None:
            config.setdefault("response_mime_type", "application/json")
            config.setdefault("response_schema", request.output_schema)
        if self.safety_settings and "safety_settings" not in config:
            config["safety_settings"] = list(self.safety_settings)
        if self.tool_config and "tool_config" not in config:
            config["tool_config"] = dict(self.tool_config)
        if self.cached_content is not None and "cached_content" not in config:
            config["cached_content"] = self.cached_content
        system_instruction = self.system_instruction or request_system
        if system_instruction is not None and "system_instruction" not in config:
            config["system_instruction"] = system_instruction
        tools = self._tools(request.tools)
        if tools and "tools" not in config:
            config["tools"] = tools
        return config

    async def complete(self, request: ModelRequest) -> ModelResponse:
        client = await self._client()
        contents, request_system = self._contents(request)
        config = self._config_data(request, request_system)
        response = client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        if inspect.isawaitable(response):
            response = await response
        return self._response_from_gemini(response, contents, config)

    async def stream_complete(
        self,
        request: ModelRequest,
        emit: SupportsEmit,
        path: tuple[str, ...],
    ) -> ModelResponse:
        client = await self._client()
        contents, request_system = self._contents(request)
        config = self._config_data(request, request_system)
        stream = client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        )
        if inspect.isawaitable(stream):
            stream = await stream
        text_started = False
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        raw_chunks: list[Any] = []
        final_chunk: Any = None
        async for chunk in stream:
            final_chunk = chunk
            raw_chunks.append(_dump(chunk))
            for part in _parts(chunk):
                text = getattr(part, "text", None)
                if not text:
                    continue
                if getattr(part, "thought", False):
                    reasoning_parts.append(text)
                    await emit(ReasoningDelta(path, "reasoning_0", text))
                else:
                    if not text_started:
                        text_started = True
                        await emit(TextStart(path, "text_0"))
                    text_parts.append(text)
                    await emit(TextDelta(path, "text_0", text))
        if text_started:
            await emit(TextEnd(path, "text_0"))
        if final_chunk is not None:
            parsed = self._response_from_gemini(final_chunk, contents, config)
            parsed.text = "".join(text_parts) or parsed.text
            if reasoning_parts:
                parsed.message_data["reasoning"] = [{"text": "".join(reasoning_parts)}]
            parsed.raw = {"chunks": raw_chunks}
            return parsed
        return ModelResponse(
            text="".join(text_parts),
            usage=Usage(model_calls=1),
            message_data=(
                {"reasoning": [{"text": "".join(reasoning_parts)}]} if reasoning_parts else {}
            ),
            provider="gemini",
            model=self.model,
            request_data={"contents": contents, "config": config},
            raw_request={"contents": contents, "config": config},
            raw={"chunks": raw_chunks},
            status="completed",
        )

    def _response_from_gemini(
        self,
        response: Any,
        contents: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> ModelResponse:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: list[ToolCall] = []
        for part in _parts(response):
            text = getattr(part, "text", None)
            if text and getattr(part, "thought", False):
                reasoning_parts.append(text)
            elif text:
                text_parts.append(text)
            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                calls.append(
                    ToolCall(
                        call_id=(
                            getattr(function_call, "id", None)
                            or getattr(function_call, "name", "call")
                        ),
                        tool=getattr(function_call, "name", "unknown"),
                        input=dict(getattr(function_call, "args", {}) or {}),
                    )
                )
        text = getattr(response, "text", None) or "".join(text_parts)
        message_data: dict[str, Any] = {}
        if reasoning_parts:
            message_data["reasoning"] = [{"text": "".join(reasoning_parts)}]
        usage = _usage_from_gemini(getattr(response, "usage_metadata", None), model_calls=1)
        return ModelResponse(
            text=text or "",
            tool_calls=calls,
            usage=usage,
            message_data=message_data,
            provider="gemini",
            model=self.model,
            request_data={"contents": contents, "config": config},
            raw_request={"contents": contents, "config": config},
            raw=_dump(response),
            response_id=getattr(response, "response_id", None),
            status="completed",
            finish_reason=_finish_reason(response),
        )


def _parts(response: Any) -> list[Any]:
    parts: list[Any] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        parts.extend(getattr(content, "parts", []) or [])
    return parts


def _usage_from_gemini(usage_obj: Any, *, model_calls: int) -> Usage:
    usage = Usage(model_calls=model_calls)
    if usage_obj is None:
        return usage
    usage.input_tokens = getattr(usage_obj, "prompt_token_count", 0) or 0
    usage.output_tokens = getattr(usage_obj, "candidates_token_count", 0) or 0
    usage.total_tokens = getattr(usage_obj, "total_token_count", 0) or 0
    usage.cached_input_tokens = getattr(usage_obj, "cached_content_token_count", 0) or 0
    usage.thinking_tokens = getattr(usage_obj, "thoughts_token_count", 0) or 0
    usage.reasoning_tokens = usage.thinking_tokens
    raw = _dump(usage_obj)
    usage.provider_details = raw if isinstance(raw, dict) else {}
    return usage


def _finish_reason(response: Any) -> str | None:
    for candidate in getattr(response, "candidates", []) or []:
        reason = getattr(candidate, "finish_reason", None)
        if reason:
            return str(reason)
    return None


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_json_dict"):
        return value.to_json_dict()
    if isinstance(value, dict):
        return {str(key): _dump(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_dump(child) for child in value]
    return value
