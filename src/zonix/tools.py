from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

from .exceptions import ToolError
from .serialization import to_jsonable
from .types import ApprovalHandler, RunState, Usage


@dataclass
class ToolContext:
    deps: Any
    usage: Usage
    state: RunState
    agent: Any
    call: Any | None = None

    @property
    def workspace(self) -> Any:
        return getattr(self.deps, "workspace", None)

    @property
    def metadata(self) -> dict[str, Any]:
        value = getattr(self.deps, "metadata", None)
        return value if isinstance(value, dict) else {}


@dataclass
class ToolMiddlewareContext:
    deps: Any
    usage: Usage
    state: RunState
    agent: Any
    tool: ToolDefinition
    call: Any
    input: dict[str, Any]


@dataclass
class ToolCallAllowed:
    input: dict[str, Any] | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallDenied:
    reason: str | None = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolApprovalRequired:
    reason: str | None = None
    input: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    approver: ApprovalHandler | None = None


@dataclass
class ToolEscalationRequired:
    reason: str | None = None
    input: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    approver: ApprovalHandler | None = None


ToolMiddlewareResult: TypeAlias = (
    ToolCallAllowed | ToolCallDenied | ToolApprovalRequired | ToolEscalationRequired
)


class ToolMiddleware(Protocol):
    def __call__(self, ctx: ToolMiddlewareContext) -> Any:
        ...


def coerce_middleware_result(value: Any) -> ToolMiddlewareResult:
    if value is None:
        return ToolCallAllowed()
    if isinstance(
        value,
        (ToolCallAllowed, ToolCallDenied, ToolApprovalRequired, ToolEscalationRequired),
    ):
        return value
    raise TypeError(
        "Tool middleware must return None, ToolCallAllowed, ToolCallDenied, "
        "ToolApprovalRequired, or ToolEscalationRequired."
    )


def _is_context_param(name: str, annotation: Any) -> bool:
    if name in {"ctx", "context"}:
        return True
    return annotation is ToolContext


def _approval_parts(
    approval: bool | ApprovalHandler,
    approver: ApprovalHandler | None = None,
) -> tuple[bool, ApprovalHandler | None]:
    if callable(approval):
        return True, approval
    return bool(approval), approver


@dataclass
class ToolDefinition:
    name: str
    func: Callable[..., Any]
    description: str
    input_model: type[BaseModel]
    schema: dict[str, Any] | None = None
    approval: bool = False
    approver: ApprovalHandler | None = None
    pass_context: bool = False
    supports_parallel: bool = False
    catch_errors: bool = False
    middleware: ToolMiddleware | None = None

    @classmethod
    def from_func(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        approval: bool | ApprovalHandler = False,
        approver: ApprovalHandler | None = None,
        supports_parallel: bool = False,
        catch_errors: bool = False,
        middleware: ToolMiddleware | None = None,
    ) -> ToolDefinition:
        signature = inspect.signature(func)
        try:
            type_hints = get_type_hints(func, include_extras=True)
        except Exception:
            type_hints = {}
        fields: dict[str, tuple[Any, Any]] = {}
        pass_context = False

        for index, (param_name, param) in enumerate(signature.parameters.items()):
            annotation = type_hints.get(
                param_name,
                Any if param.annotation is inspect.Signature.empty else param.annotation,
            )
            if index == 0 and _is_context_param(param_name, annotation):
                pass_context = True
                continue
            if param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}:
                continue
            default = ... if param.default is inspect.Signature.empty else param.default
            fields[param_name] = (annotation, default)

        model_name = f"{func.__name__.title().replace('_', '')}Input"
        input_model = create_model(
            model_name,
            __config__=ConfigDict(arbitrary_types_allowed=True),
            **fields,
        )
        needs_approval, approval_handler = _approval_parts(approval, approver)
        return cls(
            name=name or func.__name__,
            func=func,
            description=inspect.getdoc(func) or "",
            input_model=input_model,
            approval=needs_approval,
            approver=approval_handler,
            pass_context=pass_context,
            supports_parallel=supports_parallel,
            catch_errors=catch_errors,
            middleware=middleware,
        )

    @classmethod
    def from_schema_runner(
        cls,
        *,
        name: str,
        description: str,
        schema: dict[str, Any] | None,
        runner: Callable[[ToolContext, dict[str, Any]], Any],
        approval: bool | ApprovalHandler = False,
        approver: ApprovalHandler | None = None,
        supports_parallel: bool = False,
        catch_errors: bool = False,
        middleware: ToolMiddleware | None = None,
    ) -> ToolDefinition:
        def invoke_schema_runner(ctx: ToolContext, **kwargs: Any) -> Any:
            return runner(ctx, kwargs)

        input_model = create_model(
            f"{name.title().replace('_', '')}Input",
            __config__=ConfigDict(arbitrary_types_allowed=True, extra="allow"),
        )
        needs_approval, approval_handler = _approval_parts(approval, approver)
        return cls(
            name=name,
            func=invoke_schema_runner,
            description=description,
            input_model=input_model,
            schema=schema,
            approval=needs_approval,
            approver=approval_handler,
            pass_context=True,
            supports_parallel=supports_parallel,
            catch_errors=catch_errors,
            middleware=middleware,
        )

    def input_schema(self) -> dict[str, Any]:
        if self.schema is not None:
            return self.schema
        return self.input_model.model_json_schema()

    def model_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema(),
            },
        }

    def parse_input(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.schema is not None:
            return data if isinstance(data, dict) else {}
        return self.input_model.model_validate(data).model_dump(mode="python")

    async def invoke(self, ctx: ToolContext, data: dict[str, Any]) -> Any:
        try:
            parsed = self.parse_input(data)
            if self.pass_context:
                value = self.func(ctx, **parsed)
            else:
                value = self.func(**parsed)
            if inspect.isawaitable(value):
                value = await value
            return value
        except Exception as exc:  # pragma: no cover - wrapper preserves original cause
            if self.catch_errors:
                return {
                    "success": False,
                    "error_message": str(exc),
                    "error_type": type(exc).__name__,
                }
            raise ToolError(f"Tool {self.name!r} failed: {exc}") from exc

    def dump(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": to_jsonable(self.input_schema()),
            "approval": self.approval,
        }
