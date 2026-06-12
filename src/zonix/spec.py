from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Generic, TypeVar

from .engine import RunEngine
from .models import BaseChatModel, Echo
from .multi.team import RouterNode, TeamBuilder
from .multi.workflow import WorkflowBuilder
from .runtime import run_node, stream_node
from .tools import ToolDefinition
from .types import MessageLike, RunResult, RunState

OutT = TypeVar("OutT")


class Agent(Generic[OutT]):
    def __init__(
        self,
        name: str,
        *,
        role: str | None = None,
        model: BaseChatModel | None = None,
        deps: type[Any] | None = None,
        output: type[OutT] | Any = None,
        memory: Any = None,
    ) -> None:
        self.name = name
        self.role = role
        self.model = model or Echo()
        self.deps_type = deps
        self.output_type = output
        self.in_type: type[Any] = str
        self.out_type: type[Any] = output if isinstance(output, type) else Any
        self.memory = memory
        self.tools: list[ToolDefinition] = []
        self.prompts: list[str | Callable[..., Any]] = []
        self.retry_attempts = 0
        self.retry_on: type[BaseException] | tuple[type[BaseException], ...] = Exception
        self.output_repair_attempts = 1
        self.timeout_seconds: float | None = None
        self.fallback_node: Any = None

    def prompt(self, value: str | Callable[..., Any]) -> Agent[OutT]:
        self.prompts.append(value)
        return self

    def use(self, *tools: Callable[..., Any] | ToolDefinition) -> Agent[OutT]:
        for tool in tools:
            if isinstance(tool, ToolDefinition):
                self.tools.append(tool)
                continue
            self.tools.append(
                ToolDefinition.from_func(
                    tool,
                    supports_parallel=bool(getattr(tool, "supports_parallel", False)),
                )
            )
        return self

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        approval: bool = False,
        supports_parallel: bool = False,
        catch_errors: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]:
        def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(
                ToolDefinition.from_func(
                    inner,
                    name=name,
                    approval=approval,
                    supports_parallel=supports_parallel,
                    catch_errors=catch_errors,
                )
            )
            return inner

        if func is None:
            return decorator
        return decorator(func)

    def retry(
        self,
        attempts: int,
        *,
        on: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    ) -> Agent[OutT]:
        self.retry_attempts = attempts
        self.retry_on = on
        return self

    def timeout(self, seconds: float) -> Agent[OutT]:
        self.timeout_seconds = seconds
        return self

    def repair_output(self, attempts: int = 1) -> Agent[OutT]:
        self.output_repair_attempts = attempts
        return self

    def fallback(self, node: Any) -> Agent[OutT]:
        self.fallback_node = node
        return self

    async def __call__(
        self,
        task: Any,
        *,
        ctx: Any = None,
        session: Any = None,
        extra: str | None = None,
        message_history: list[MessageLike] | None = None,
    ) -> OutT:
        result = await self.run(
            task,
            ctx=ctx,
            session=session,
            extra=extra,
            message_history=message_history,
        )
        return result.output

    async def run(
        self,
        task: Any,
        *,
        ctx: Any = None,
        session: Any = None,
        extra: str | None = None,
        message_history: list[MessageLike] | None = None,
        trace: bool = True,
    ) -> RunResult:
        return await run_node(
            self,
            task,
            ctx=ctx,
            session=session,
            extra=extra,
            message_history=message_history,
        )

    def stream(
        self,
        task: Any,
        *,
        ctx: Any = None,
        session: Any = None,
        extra: str | None = None,
        message_history: list[MessageLike] | None = None,
    ) -> AsyncIterator[Any]:
        return stream_node(
            self,
            task,
            ctx=ctx,
            session=session,
            extra=extra,
            message_history=message_history,
        )

    async def invoke(self, x: Any, st: RunState) -> Any:
        return await RunEngine(self).invoke(x, st)

    def dump_spec(self) -> dict[str, Any]:
        return {
            "kind": "agent",
            "name": self.name,
            "role": self.role,
            "model": getattr(self.model, "name", repr(self.model)),
            "deps": getattr(self.deps_type, "__name__", None),
            "output": getattr(self.output_type, "__name__", repr(self.output_type)),
            "tools": [tool.dump() for tool in self.tools],
            "prompts": [p if isinstance(p, str) else repr(p) for p in self.prompts],
        }


def agent(
    name: str,
    *,
    role: str | None = None,
    model: BaseChatModel | None = None,
    deps: type[Any] | None = None,
    output: type[OutT] | Any = None,
    memory: Any = None,
) -> Agent[OutT]:
    return Agent(
        name,
        role=role,
        model=model,
        deps=deps,
        output=output,
        memory=memory,
    )


def workflow(name: str) -> WorkflowBuilder:
    return WorkflowBuilder(name)


def team(name: str) -> TeamBuilder:
    return TeamBuilder(name)


def router(name: str, rule: Callable[..., Any] | Any) -> RouterNode:
    return RouterNode(name, rule)
