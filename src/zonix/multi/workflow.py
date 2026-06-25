from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Any

from zonix.graph import GraphEdge, GraphNode, GraphSpec, safe_graph_id
from zonix.runtime import resolve_approvals, run_node, stream_node
from zonix.sync import iter_async_sync
from zonix.sync import run_sync as _run_sync
from zonix.types import ApprovalHandler, MessageLike, Node, RunResult, RunState


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
        for index, step in enumerate(self.steps):
            current = await step.invoke(current, st)
            node = getattr(step, "node", None)
            scratch_key = getattr(node, "name", None) or f"{index}:{type(step).__name__}"
            st.scratch[scratch_key] = current
        return current

    async def solve(
        self,
        task: Any,
        *,
        ctx: Any = None,
        session: Any = None,
        message_history: list[MessageLike] | None = None,
        approval: ApprovalHandler | None = None,
    ) -> Any:
        return (
            await self.run(
                task,
                ctx=ctx,
                session=session,
                message_history=message_history,
                approval=approval,
            )
        ).output

    def solve_sync(
        self,
        task: Any,
        *,
        ctx: Any = None,
        session: Any = None,
        message_history: list[MessageLike] | None = None,
        approval: ApprovalHandler | None = None,
    ) -> Any:
        return _run_sync(
            lambda: self.solve(
                task,
                ctx=ctx,
                session=session,
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
        message_history: list[MessageLike] | None = None,
        trace: bool = True,
        approval: ApprovalHandler | None = None,
    ) -> RunResult:
        return await resolve_approvals(
            await run_node(self, task, ctx=ctx, session=session, message_history=message_history),
            approval,
        )

    def run_sync(
        self,
        task: Any,
        *,
        ctx: Any = None,
        session: Any = None,
        message_history: list[MessageLike] | None = None,
        trace: bool = True,
        approval: ApprovalHandler | None = None,
    ) -> RunResult:
        return _run_sync(
            lambda: self.run(
                task,
                ctx=ctx,
                session=session,
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
        message_history: list[MessageLike] | None = None,
    ) -> AsyncIterator[Any]:
        return stream_node(self, task, ctx=ctx, session=session, message_history=message_history)

    def stream_sync(
        self,
        task: Any,
        *,
        ctx: Any = None,
        session: Any = None,
        message_history: list[MessageLike] | None = None,
    ) -> Iterator[Any]:
        return iter_async_sync(
            lambda: self.stream(
                task,
                ctx=ctx,
                session=session,
                message_history=message_history,
            )
        )

    def graph(self) -> GraphSpec:
        nodes: list[GraphNode] = [
            GraphNode("start", "start", "start"),
            GraphNode("end", "end", "end"),
        ]
        edges: list[GraphEdge] = []
        previous = "start"
        for index, step in enumerate(self.steps):
            previous = _append_workflow_step(nodes, edges, previous, index, step)
        edges.append(GraphEdge(previous, "end"))
        return GraphSpec(self.name, nodes, edges)

    def to_mermaid(self) -> str:
        return self.graph().mermaid()

    def save_graph(self, path: str) -> Any:
        return self.graph().save(path)


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


def _append_workflow_step(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    previous: str,
    index: int,
    step: Any,
) -> str:
    if isinstance(step, NodeStep):
        node_id = _step_id(index, step.node.name)
        nodes.append(GraphNode(node_id, step.node.name, "node"))
        edges.append(GraphEdge(previous, node_id))
        return node_id
    if isinstance(step, ParallelStep):
        fork_id = f"parallel_{index}_fork"
        join_id = f"parallel_{index}_join"
        nodes.append(GraphNode(fork_id, "parallel", "parallel"))
        nodes.append(GraphNode(join_id, "join", "join"))
        edges.append(GraphEdge(previous, fork_id))
        for node in step.nodes:
            node_id = _step_id(index, node.name)
            nodes.append(GraphNode(node_id, node.name, "node"))
            edges.append(GraphEdge(fork_id, node_id))
            edges.append(GraphEdge(node_id, join_id))
        return join_id
    if isinstance(step, BranchStep):
        branch_id = f"branch_{index}"
        join_id = f"branch_{index}_join"
        nodes.append(GraphNode(branch_id, "branch", "branch"))
        nodes.append(GraphNode(join_id, "join", "join"))
        edges.append(GraphEdge(previous, branch_id))
        then_id = _step_id(index, step.then_node.name)
        nodes.append(GraphNode(then_id, step.then_node.name, "node"))
        edges.append(GraphEdge(branch_id, then_id, "true"))
        edges.append(GraphEdge(then_id, join_id))
        if step.else_node is not None:
            else_id = _step_id(index, step.else_node.name)
            nodes.append(GraphNode(else_id, step.else_node.name, "node"))
            edges.append(GraphEdge(branch_id, else_id, "false"))
            edges.append(GraphEdge(else_id, join_id))
        else:
            edges.append(GraphEdge(branch_id, join_id, "false"))
        return join_id
    if isinstance(step, LoopStep):
        node_id = _step_id(index, step.node.name)
        nodes.append(GraphNode(node_id, f"{step.node.name} loop", "loop"))
        edges.append(GraphEdge(previous, node_id))
        edges.append(GraphEdge(node_id, node_id, f"until true / max {step.max_iters}"))
        return node_id
    node_id = _step_id(index, type(step).__name__)
    nodes.append(GraphNode(node_id, type(step).__name__, "step"))
    edges.append(GraphEdge(previous, node_id))
    return node_id


def _step_id(index: int, name: str) -> str:
    return safe_graph_id(f"s_{index}_{name}")
