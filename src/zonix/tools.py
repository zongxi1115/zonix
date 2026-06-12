from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

from .exceptions import ToolError
from .serialization import to_jsonable
from .types import RunState, Usage


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


def _is_context_param(name: str, annotation: Any) -> bool:
    if name in {"ctx", "context"}:
        return True
    return annotation is ToolContext


@dataclass
class ToolDefinition:
    name: str
    func: Callable[..., Any]
    description: str
    input_model: type[BaseModel]
    schema: dict[str, Any] | None = None
    approval: bool = False
    pass_context: bool = False
    supports_parallel: bool = False
    catch_errors: bool = False

    @classmethod
    def from_func(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        approval: bool = False,
        supports_parallel: bool = False,
        catch_errors: bool = False,
    ) -> ToolDefinition:
        signature = inspect.signature(func)
        fields: dict[str, tuple[Any, Any]] = {}
        pass_context = False

        for index, (param_name, param) in enumerate(signature.parameters.items()):
            annotation = Any if param.annotation is inspect.Signature.empty else param.annotation
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
        return cls(
            name=name or func.__name__,
            func=func,
            description=inspect.getdoc(func) or "",
            input_model=input_model,
            approval=approval,
            pass_context=pass_context,
            supports_parallel=supports_parallel,
            catch_errors=catch_errors,
        )

    @classmethod
    def from_schema_runner(
        cls,
        *,
        name: str,
        description: str,
        schema: dict[str, Any] | None,
        runner: Callable[[ToolContext, dict[str, Any]], Any],
        supports_parallel: bool = False,
        catch_errors: bool = False,
    ) -> ToolDefinition:
        def invoke_schema_runner(ctx: ToolContext, **kwargs: Any) -> Any:
            return runner(ctx, kwargs)

        input_model = create_model(
            f"{name.title().replace('_', '')}Input",
            __config__=ConfigDict(arbitrary_types_allowed=True, extra="allow"),
        )
        return cls(
            name=name,
            func=invoke_schema_runner,
            description=description,
            input_model=input_model,
            schema=schema,
            pass_context=True,
            supports_parallel=supports_parallel,
            catch_errors=catch_errors,
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
