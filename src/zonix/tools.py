from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

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
    approval: bool = False
    pass_context: bool = False

    @classmethod
    def from_func(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        approval: bool = False,
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
        )

    def input_schema(self) -> dict[str, Any]:
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
            raise ToolError(f"Tool {self.name!r} failed: {exc}") from exc

    def dump(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": to_jsonable(self.input_schema()),
            "approval": self.approval,
        }
