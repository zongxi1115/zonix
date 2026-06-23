from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from pydantic import TypeAdapter

from zonix.exceptions import MaxStepsExceeded
from zonix.graph import GraphEdge, GraphNode, GraphSpec, safe_graph_id
from zonix.runtime import resolve_approvals, run_node, stream_node
from zonix.sync import iter_async_sync, run_sync as _run_sync
from zonix.types import ApprovalHandler, MessageLike, Node, Route, RunResult, RunState


class RouterNode:
    def __init__(self, name: str, rule: Callable[..., Any] | Node) -> None:
        self.name = name
        self.rule = rule
        self.in_type: type[Any] = Any
        self.out_type: type[Any] = Route

    async def invoke(self, x: Any, st: RunState) -> Route:
        if hasattr(self.rule, "invoke"):
            value = await self.rule.invoke(x, st.scoped(getattr(self.rule, "name", self.name)))
        else:
            signature = inspect.signature(self.rule)
            if len(signature.parameters) <= 1:
                value = self.rule(x)
            else:
                value = self.rule(x, st)
            if inspect.isawaitable(value):
                value = await value
        return TypeAdapter(Route).validate_python(value)


class TeamNode:
    def __init__(
        self,
        name: str,
        agents: dict[str, Node],
        router: Node,
        *,
        max_steps: int = 8,
    ) -> None:
        self.name = name
        self.agents = agents
        self.router = router
        self.max_steps = max_steps
        self.in_type: type[Any] = Any
        self.out_type: type[Any] = Any

    async def invoke(self, task: Any, st: RunState) -> Any:
        current = task
        for _ in range(self.max_steps):
            route = await self.router.invoke(current, st.scoped(self.router.name))
            if route.done:
                return current
            if route.next is None:
                raise ValueError(f"Router {self.router.name!r} returned no next node.")
            if route.next not in self.agents:
                raise KeyError(f"Router selected unknown node {route.next!r}.")
            node = self.agents[route.next]
            current = await node.invoke(
                route.input if route.input is not None else current,
                st.scoped(node.name),
            )
            st.scratch[node.name] = current
        raise MaxStepsExceeded(f"Team {self.name!r} exceeded {self.max_steps} steps.")

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
        router_id = safe_graph_id(f"router_{self.router.name}")
        nodes: list[GraphNode] = [
            GraphNode("start", "start", "start"),
            GraphNode(router_id, self.router.name, "router"),
            GraphNode("done", "done", "end"),
        ]
        edges: list[GraphEdge] = [
            GraphEdge("start", router_id),
            GraphEdge(router_id, "done", "done"),
        ]
        for node in self.agents.values():
            node_id = safe_graph_id(f"agent_{node.name}")
            nodes.append(GraphNode(node_id, node.name, "agent"))
            edges.append(GraphEdge(router_id, node_id, f"next={node.name}"))
            edges.append(GraphEdge(node_id, router_id, "route again"))
        return GraphSpec(self.name, nodes, edges)

    def to_mermaid(self) -> str:
        return self.graph().mermaid()

    def save_graph(self, path: str) -> Any:
        return self.graph().save(path)


class TeamBuilder:
    def __init__(self, name: str) -> None:
        self.name = name
        self._agents: dict[str, Node] = {}
        self._router: Node | None = None

    def add(self, *nodes: Node) -> TeamBuilder:
        for node in nodes:
            self._agents[node.name] = node
        return self

    def route(self, router: Node) -> TeamBuilder:
        self._router = router
        return self

    def build(self, *, max_steps: int = 8) -> TeamNode:
        if not self._agents:
            raise ValueError("team requires at least one agent")
        if self._router is None:
            raise ValueError("team requires a router")
        return TeamNode(self.name, dict(self._agents), self._router, max_steps=max_steps)
