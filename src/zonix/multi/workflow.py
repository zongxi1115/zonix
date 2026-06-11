from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from zonix.runtime import run_node, stream_node
from zonix.types import Node, RunResult, RunState


@dataclass
class NodeStep:
    node: Node

    async def invoke(self, current: Any, st: RunState) -> Any:
        return await self.node.invoke(current, st.scoped(self.node.name))


@dataclass
class ParallelStep:
    nodes: tuple[Node, ...]

    async def invoke(self, current: Any, st: RunState) -> list[Any]:
        async def call(node: Node) -> Any:
            return await node.invoke(current, st.scoped(node.name))

        return list(await asyncio.gather(*(call(node) for node in self.nodes)))


@dataclass
class BranchStep:
    predicate: Callable[[Any], bool]
    then_node: Node
    else_node: Node | None = None

    async def invoke(self, current: Any, st: RunState) -> Any:
        node = self.then_node if self.predicate(current) else self.else_node
        if node is None:
            return current
        return await node.invoke(current, st.scoped(node.name))


@dataclass
class LoopStep:
    node: Node
    until: Callable[[Any], bool]
    max_iters: int = 3

    async def invoke(self, current: Any, st: RunState) -> Any:
        value = current
        for index in range(self.max_iters):
            value = await self.node.invoke(value, st.scoped(f"{self.node.name}[{index}]"))
            if self.until(value):
                break
        return value


class WorkflowNode:
    def __init__(self, name: str, steps: list[Any]) -> None:
        self.name = name
        self.steps = steps
        self.in_type: type[Any] = Any
        self.out_type: type[Any] = Any

    async def invoke(self, x: Any, st: RunState) -> Any:
        current = x
        for step in self.steps:
            current = await step.invoke(current, st)
            st.scratch[getattr(step, "node", step).__class__.__name__] = current
        return current

    async def solve(self, task: Any, *, ctx: Any = None, session: Any = None) -> Any:
        return (await self.run(task, ctx=ctx, session=session)).output

    async def run(self, task: Any, *, ctx: Any = None, session: Any = None, trace: bool = True) -> RunResult:
        return await run_node(self, task, ctx=ctx, session=session)

    def stream(self, task: Any, *, ctx: Any = None, session: Any = None) -> AsyncIterator[Any]:
        return stream_node(self, task, ctx=ctx, session=session)


class WorkflowBuilder:
    def __init__(self, name: str) -> None:
        self.name = name
        self._steps: list[Any] = []

    def start(self, node: Node) -> WorkflowBuilder:
        self._steps.append(NodeStep(node))
        return self

    def then(self, node: Node) -> WorkflowBuilder:
        self._steps.append(NodeStep(node))
        return self

    def parallel(self, *nodes: Node) -> WorkflowBuilder:
        self._steps.append(ParallelStep(tuple(nodes)))
        return self

    def join(self, node: Node) -> WorkflowBuilder:
        self._steps.append(NodeStep(node))
        return self

    def branch(
        self,
        predicate: Callable[[Any], bool],
        *,
        then: Node,
        else_: Node | None = None,
    ) -> WorkflowBuilder:
        self._steps.append(BranchStep(predicate, then, else_))
        return self

    def loop(
        self,
        node: Node,
        *,
        until: Callable[[Any], bool],
        max_iters: int = 3,
    ) -> WorkflowBuilder:
        self._steps.append(LoopStep(node, until, max_iters=max_iters))
        return self

    def build(self) -> WorkflowNode:
        if not self._steps:
            raise ValueError("workflow requires at least one step")
        return WorkflowNode(self.name, list(self._steps))
