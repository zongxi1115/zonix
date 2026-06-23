from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from typing import Any, Generic, TypeVar

from .engine import RunEngine
from .models import BaseChatModel, Echo
from .multi.team import RouterNode, TeamBuilder
from .multi.workflow import WorkflowBuilder
from .runtime import resolve_approvals, run_node, stream_node
from .sync import iter_async_sync, run_sync as _run_sync
from .tools import ToolDefinition, ToolMiddleware
from .types import ApprovalHandler, MessageLike, RunResult, RunState

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
        tools: Iterable[Callable[..., Any] | ToolDefinition] | None = None,
        middlewares: Iterable[ToolMiddleware] | None = None,
        prompts: Iterable[str | Callable[..., Any]] | None = None,
        approver: ApprovalHandler | None = None,
        recover_tool_input_errors: bool = False,
    ) -> None:
        self.name = name
        self.role = role
        self.model = model or Echo()
        self.deps_type = deps
        self.output_type = output
        self.in_type: type[Any] = str
        self.out_type: type[Any] = output if isinstance(output, type) else Any
        self.memory = memory
        self.approver = approver
        self.tools: list[ToolDefinition] = []
        self.middlewares: list[ToolMiddleware] = list(middlewares or ())
        self.prompts: list[str | Callable[..., Any]] = list(prompts or ())
        self.retry_attempts = 0
        self.retry_on: type[BaseException] | tuple[type[BaseException], ...] = Exception
        self.output_repair_attempts = 1
        self.recover_tool_input_errors = recover_tool_input_errors
        self.timeout_seconds: float | None = None
        self.fallback_node: Any = None
        if tools is not None:
            self.use(*tools)

    def prompt(self, value: str | Callable[..., Any]) -> Agent[OutT]:
        self.prompts.append(value)
        return self

    def use(
        self,
        *tools: Callable[..., Any] | ToolDefinition,
        approval: bool | ApprovalHandler | None = None,
        middleware: ToolMiddleware | None = None,
    ) -> Agent[OutT]:
        for tool in tools:
            if isinstance(tool, ToolDefinition):
                if approval is not None:
                    if callable(approval):
                        tool.approval = True
                        tool.approver = approval
                    else:
                        tool.approval = bool(approval)
                        if not tool.approval:
                            tool.approver = None
                self.tools.append(tool)
                continue
            tool_approval = approval
            if tool_approval is None:
                tool_approval = getattr(tool, "approval", False)
            self.tools.append(
                ToolDefinition.from_func(
                    tool,
                    approval=tool_approval,
                    approver=getattr(tool, "approver", None),
                    supports_parallel=bool(getattr(tool, "supports_parallel", False)),
                    middleware=middleware or getattr(tool, "middleware", None),
                )
            )
        return self

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        approval: bool | ApprovalHandler = False,
        approver: ApprovalHandler | None = None,
        supports_parallel: bool = False,
        catch_errors: bool = False,
        middleware: ToolMiddleware | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]:
        def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(
                ToolDefinition.from_func(
                    inner,
                    name=name,
                    approval=approval,
                    approver=approver,
                    supports_parallel=supports_parallel,
                    catch_errors=catch_errors,
                    middleware=middleware,
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

    def repair_tool_inputs(self, enabled: bool = True) -> Agent[OutT]:
        self.recover_tool_input_errors = enabled
        return self

    def fallback(self, node: Any) -> Agent[OutT]:
        self.fallback_node = node
        return self

    def middleware(self, fn: ToolMiddleware) -> Agent[OutT]:
        self.middlewares.append(fn)
        return self

    async def __call__(
        self,
        task: Any,
        *,
        ctx: Any = None,
        session: Any = None,
        extra: str | None = None,
        message_history: list[MessageLike] | None = None,
        approval: ApprovalHandler | None = None,
    ) -> OutT:
        result = await self.run(
            task,
            ctx=ctx,
            session=session,
            extra=extra,
            message_history=message_history,
            approval=approval,
        )
        return result.output

    def call_sync(
        self,
        task: Any,
        *,
        ctx: Any = None,
        session: Any = None,
        extra: str | None = None,
        message_history: list[MessageLike] | None = None,
        approval: ApprovalHandler | None = None,
    ) -> OutT:
        return _run_sync(
            lambda: self(
                task,
                ctx=ctx,
                session=session,
                extra=extra,
                message_history=message_history,
                approval=approval,
            )
        )

    async def run(
        self,
        task: Any,
        *,
        ctx: Any = None,
        session: Any = None,
        extra: str | None = None,
        message_history: list[MessageLike] | None = None,
        trace: bool = True,
        approval: ApprovalHandler | None = None,
    ) -> RunResult:
        return await resolve_approvals(
            await run_node(
                self,
                task,
                ctx=ctx,
                session=session,
                extra=extra,
                message_history=message_history,
            ),
            approval,
        )

    def run_sync(
        self,
        task: Any,
        *,
        ctx: Any = None,
        session: Any = None,
        extra: str | None = None,
        message_history: list[MessageLike] | None = None,
        trace: bool = True,
        approval: ApprovalHandler | None = None,
    ) -> RunResult:
        return _run_sync(
            lambda: self.run(
                task,
                ctx=ctx,
                session=session,
                extra=extra,
                message_history=message_history,
                trace=trace,
                approval=approval,
            )
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

    def stream_sync(
        self,
        task: Any,
        *,
        ctx: Any = None,
        session: Any = None,
        extra: str | None = None,
        message_history: list[MessageLike] | None = None,
    ) -> Iterator[Any]:
        return iter_async_sync(
            lambda: self.stream(
                task,
                ctx=ctx,
                session=session,
                extra=extra,
                message_history=message_history,
            )
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
            "middlewares": [repr(middleware) for middleware in self.middlewares],
            "prompts": [p if isinstance(p, str) else repr(p) for p in self.prompts],
            "approver": repr(self.approver) if self.approver is not None else None,
            "recover_tool_input_errors": self.recover_tool_input_errors,
        }


def agent(
    name: str,
    *,
    role: str | None = None,
    model: BaseChatModel | None = None,
    deps: type[Any] | None = None,
    output: type[OutT] | Any = None,
    memory: Any = None,
    tools: Iterable[Callable[..., Any] | ToolDefinition] | None = None,
    middlewares: Iterable[ToolMiddleware] | None = None,
    prompts: Iterable[str | Callable[..., Any]] | None = None,
    approver: ApprovalHandler | None = None,
    recover_tool_input_errors: bool = False,
) -> Agent[OutT]:
    return Agent(
        name,
        role=role,
        model=model,
        deps=deps,
        output=output,
        memory=memory,
        tools=tools,
        middlewares=middlewares,
        prompts=prompts,
        approver=approver,
        recover_tool_input_errors=recover_tool_input_errors,
    )


def workflow(name: str) -> WorkflowBuilder:
    return WorkflowBuilder(name)


def team(name: str) -> TeamBuilder:
    return TeamBuilder(name)


def router(name: str, rule: Callable[..., Any] | Any) -> RouterNode:
    return RouterNode(name, rule)
