from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

from pydantic import BaseModel, TypeAdapter

from .events import (
    ApprovalRequired,
    ReasoningDelta,
    TextDelta,
    TextEnd,
    TextStart,
    ToolInputAvailable,
    ToolInputStart,
    ToolOutputAvailable,
)
from .exceptions import (
    OutputValidationError,
    RunPaused,
    ToolApprovalRejected,
)
from .hitl import approval_key, pending_from_call
from .models import ModelRequest, ModelResponse
from .serialization import to_jsonable
from .tools import ToolContext, ToolDefinition
from .types import Message, ModelCall, RunState, ToolCall


class RunEngine:
    def __init__(self, agent: Any) -> None:
        self.agent = agent

    async def invoke(self, task: Any, st: RunState) -> Any:
        attempts = self.agent.retry_attempts + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                if self.agent.timeout_seconds is None:
                    return await self._invoke_once(task, st)
                return await asyncio.wait_for(
                    self._invoke_once(task, st),
                    timeout=self.agent.timeout_seconds,
                )
            except self.agent.retry_on as exc:
                last_error = exc
                st.trace.record({"attempt": attempt + 1, "error": str(exc)})
                if attempt + 1 >= attempts:
                    break
        if self.agent.fallback_node is not None:
            return await self.agent.fallback_node.invoke(
                task,
                st.scoped(self.agent.fallback_node.name),
            )
        if last_error is not None:
            raise last_error
        raise RuntimeError("Agent invocation failed without an exception.")

    async def _invoke_once(self, task: Any, st: RunState) -> Any:
        messages = await self._build_messages(task, st)
        tools = {tool.name: tool for tool in self.agent.tools}
        repair_rounds = 0

        while True:
            request = ModelRequest(
                messages=messages,
                tools=[tool.model_tool_schema() for tool in tools.values()],
                output_schema=self._output_schema(),
                output_name=self._output_name(),
                metadata={"agent": self.agent.name, "role": self.agent.role},
                ctx=st.ctx,
                state=st,
                task=task,
            )
            response = await self._call_model(request, st)
            st.usage += response.usage

            if response.tool_calls:
                message_data = dict(response.message_data)
                message_data["tool_calls"] = [
                    call.model_dump(mode="json") for call in response.tool_calls
                ]
                messages.append(
                    Message(
                        role="assistant",
                        content=response.text,
                        data=message_data,
                    )
                )
                await self._run_tool_calls(response.tool_calls, tools, messages, st)
                if st.stop_requested:
                    st.messages = messages
                    return st.stop_output
                continue

            if response.text:
                messages.append(
                    Message(
                        role="assistant",
                        content=response.text,
                        data=dict(response.message_data),
                    )
                )

            try:
                output = self._validate_output(response)
            except OutputValidationError as exc:
                if repair_rounds >= self.agent.output_repair_attempts:
                    raise
                repair_rounds += 1
                messages.append(
                    Message(
                        role="user",
                        content=(
                            "The previous response did not validate. "
                            "Return a corrected JSON value only, with no Markdown or prose. "
                            f"Validation error: {exc}"
                        ),
                        data={"kind": "output_repair", "attempt": repair_rounds},
                    )
                )
                continue
            st.messages = messages
            st.scratch[self.agent.name] = output
            await self._remember(task, output, st)
            return output

    async def _run_tool_calls(
        self,
        calls: list[ToolCall],
        tools: dict[str, ToolDefinition],
        messages: list[Message],
        st: RunState,
    ) -> None:
        parallel_buffer: list[ToolCall] = []

        async def flush_parallel_buffer() -> None:
            nonlocal parallel_buffer
            if not parallel_buffer:
                return
            if len(parallel_buffer) == 1:
                await self._run_tool(parallel_buffer[0], tools, messages, st)
            else:
                await asyncio.gather(
                    *[
                        self._run_tool(call, tools, messages, st)
                        for call in parallel_buffer
                    ]
                )
            parallel_buffer = []

        for call in calls:
            tool = tools.get(call.tool)
            if tool is not None and bool(getattr(tool, "supports_parallel", False)):
                parallel_buffer.append(call)
                continue

            await flush_parallel_buffer()
            await self._run_tool(call, tools, messages, st)

        await flush_parallel_buffer()

    async def _build_messages(self, task: Any, st: RunState) -> list[Message]:
        messages: list[Message] = []
        instructions = await self._instructions(task, st)
        if instructions:
            messages.append(Message(role="system", content=instructions))
        messages.extend(st.message_history)
        if st.session is not None:
            history = await st.session.recall(task, st.ctx, memory=self.agent.memory)
            messages.extend(history)
        messages.append(Message(role="user", content=str(task)))
        if st.extra:
            messages.append(Message(role="user", content=st.extra, data={"kind": "extra"}))
        return messages

    async def _instructions(self, task: Any, st: RunState) -> str:
        parts: list[str] = []
        if self.agent.role:
            parts.append(f"Role: {self.agent.role}")
        for prompt in self.agent.prompts:
            if isinstance(prompt, str):
                parts.append(prompt)
                continue
            value = self._call_prompt(prompt, task, st)
            if inspect.isawaitable(value):
                value = await value
            if value:
                parts.append(str(value))
        schema = self._output_schema()
        if schema is not None:
            parts.append(
                "Return exactly one valid JSON value that validates against this output schema. "
                "Do not include Markdown, prose, comments, or trailing text. "
                "Use double quotes and valid JSON escaping.\n"
                + json.dumps(schema, ensure_ascii=False)
            )
        return "\n\n".join(parts)

    def _call_prompt(self, prompt: Any, task: Any, st: RunState) -> Any:
        signature = inspect.signature(prompt)
        params = list(signature.parameters)
        if len(params) == 0:
            return prompt()
        if len(params) == 1:
            return prompt(st.ctx)
        return prompt(st.ctx, task)

    async def _call_model(self, request: ModelRequest, st: RunState) -> ModelResponse:
        if st.bus._emit is not None:
            response = await self.agent.model.stream_complete(request, st.bus.publish, st.path)
        else:
            response = await self.agent.model.complete(request)
            await self._emit_response(response, st)
        self._record_model_call(request, response, st)
        return response

    def _record_model_call(
        self,
        request: ModelRequest,
        response: ModelResponse,
        st: RunState,
    ) -> None:
        request_data = response.request_data or {
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "tools": to_jsonable(request.tools),
            "output_schema": to_jsonable(request.output_schema),
            "output_name": request.output_name,
            "metadata": to_jsonable(request.metadata),
        }
        call = ModelCall(
            provider=response.provider or getattr(self.agent.model, "name", "").split(":", 1)[0],
            model=response.model or getattr(self.agent.model, "name", None),
            request=request_data,
            raw_request=response.raw_request or request_data,
            raw_response=response.raw,
            usage=response.usage,
            response_id=response.response_id,
            status=response.status,
            finish_reason=response.finish_reason,
            message_data=dict(response.message_data),
        )
        st.model_calls.append(call)
        st.trace.record(
            {
                "kind": "model_call",
                "provider": call.provider,
                "model": call.model,
                "response_id": call.response_id,
                "status": call.status,
                "finish_reason": call.finish_reason,
                "usage": to_jsonable(call.usage),
            }
        )

    async def _emit_response(self, response: ModelResponse, st: RunState) -> None:
        reasoning = response.message_data.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            await st.bus.publish(ReasoningDelta(st.path, "reasoning_0", reasoning))
        elif isinstance(reasoning, list):
            for index, item in enumerate(reasoning):
                text = item.get("text") if isinstance(item, dict) else str(item)
                if text:
                    await st.bus.publish(ReasoningDelta(st.path, f"reasoning_{index}", text))
        if response.text:
            await st.bus.publish(TextStart(st.path, "text_0"))
            await st.bus.publish(TextDelta(st.path, "text_0", response.text))
            await st.bus.publish(TextEnd(st.path, "text_0"))
        for call in response.tool_calls:
            await st.bus.publish(ToolInputStart(st.path, call.call_id, call.tool))
            await st.bus.publish(ToolInputAvailable(st.path, call.call_id, call.tool, call.input))

    async def _run_tool(
        self,
        call: ToolCall,
        tools: dict[str, ToolDefinition],
        messages: list[Message],
        st: RunState,
    ) -> None:
        if call.tool not in tools:
            raise KeyError(f"Agent {self.agent.name!r} has no tool named {call.tool!r}.")

        tool = tools[call.tool]
        parsed_input = tool.parse_input(call.input)
        normalized_call = ToolCall(call_id=call.call_id, tool=call.tool, input=parsed_input)
        approved_input = await self._approval_for(tool, normalized_call, st)
        if approved_input is not None and isinstance(approved_input, dict):
            parsed_input = tool.parse_input(approved_input)

        ctx = ToolContext(
            deps=st.ctx,
            usage=st.usage,
            state=st,
            agent=self.agent,
            call=normalized_call,
        )
        output = await tool.invoke(ctx, parsed_input)
        st.usage.tool_calls += 1
        await st.bus.publish(ToolOutputAvailable(st.path, call.call_id, to_jsonable(output)))
        messages.append(
            Message(
                role="tool",
                name=tool.name,
                tool_call_id=call.call_id,
                content=json.dumps(to_jsonable(output), ensure_ascii=False),
            )
        )

    async def _approval_for(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        st: RunState,
    ) -> dict[str, Any] | bool | None:
        if not tool.approval:
            return None
        key = approval_key(call.tool, call.input)
        decision = st.approvals.get(call.call_id, st.approvals.get(key))
        if decision is False:
            raise ToolApprovalRejected(f"Tool call {call.call_id} was rejected.")
        if decision is True or isinstance(decision, dict):
            return decision

        pending = pending_from_call(call)
        await st.bus.publish(ApprovalRequired(st.path, call.call_id, call.tool, call.input))
        snapshot = {
            "run_id": st.run_id,
            "path": st.path,
            "pending": pending,
            "messages": messages_dump(st.messages),
            "scratch": to_jsonable(st.scratch),
            "trace": to_jsonable(st.trace),
            "usage": to_jsonable(st.usage),
        }
        raise RunPaused(pending=pending, snapshot=snapshot)

    def _output_schema(self) -> dict[str, Any] | None:
        output_type = self.agent.output_type
        if output_type is None or output_type is Any:
            return None
        return TypeAdapter(output_type).json_schema()

    def _output_name(self) -> str | None:
        output_type = self.agent.output_type
        if output_type is None:
            return None
        return getattr(output_type, "__name__", repr(output_type))

    def _validate_output(self, response: ModelResponse) -> Any:
        output_type = self.agent.output_type
        raw = response.output if response.output is not None else response.text
        if output_type is None or output_type is Any:
            return raw
        try:
            if isinstance(raw, output_type):
                return raw
        except TypeError:
            pass
        data = raw
        if isinstance(raw, str) and output_type is not str:
            try:
                data = _load_json_text(raw)
            except json.JSONDecodeError as exc:
                raise OutputValidationError(
                    f"Agent {self.agent.name!r} expected {self._output_name()} JSON, got text."
                ) from exc
        try:
            if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                return output_type.model_validate(data)
        except TypeError:
            pass
        try:
            return TypeAdapter(output_type).validate_python(data)
        except Exception as exc:
            raise OutputValidationError(
                "Agent "
                f"{self.agent.name!r} could not validate model output as {self._output_name()}."
            ) from exc

    async def _remember(self, task: Any, output: Any, st: RunState) -> None:
        if st.session is None:
            return
        await st.session.remember(Message(role="user", content=str(task)))
        await st.session.remember(
            Message(role="assistant", content=json.dumps(to_jsonable(output), ensure_ascii=False))
        )


def messages_dump(messages: list[Message]) -> list[dict[str, Any]]:
    return [message.model_dump(mode="json") for message in messages]


def _load_json_text(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        candidate = _extract_json_candidate(stripped)
        if candidate is None:
            raise
        return json.loads(candidate)


def _extract_json_candidate(text: str) -> str | None:
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return None
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
