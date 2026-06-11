from __future__ import annotations

import asyncio
import inspect
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from zonix import Finish, Message, TextDelta, agent, router, team, workflow
from zonix.exceptions import ToolError
from zonix.hitl import CheckpointStore
from zonix.memory import Session, Summarize, Vector, Window
from zonix.models import Anthropic, OpenAI
from zonix.types import Route, RunState
from zonix.wire.ai_sdk import to_ai_sdk


class Plan(BaseModel):
    goal: str
    files: list[str]
    steps: list[str]


class ToolSummary(BaseModel):
    used_tool: bool
    area: str
    facts: list[str]


class PatchResult(BaseModel):
    applied: bool
    path: str
    note: str


class StageResult(BaseModel):
    stage: str
    summary: str


class FunctionNode:
    def __init__(self, name: str, fn: Callable[[Any, RunState], Any]) -> None:
        self.name = name
        self.fn = fn
        self.in_type = Any
        self.out_type = Any

    async def invoke(self, x: Any, st: RunState) -> Any:
        value = self.fn(x, st)
        if inspect.isawaitable(value):
            value = await value
        return value


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def openai_base_url() -> str:
    return os.environ.get("ZONIX_OPENAI_BASE_URL") or os.environ.get("ZONIX_BASE_URL") or env(
        "ZONIX_OPENAI_BASE_URL"
    )


def anthropic_base_url() -> str | None:
    return os.environ.get("ZONIX_ANTHROPIC_BASE_URL")


def make_openai(
    settings: dict[str, Any] | None = None,
    *,
    native_structured_output: bool = True,
) -> OpenAI:
    return OpenAI(
        model=env("ZONIX_MODEL"),
        api_key=env("ZONIX_API_KEY"),
        base_url=openai_base_url(),
        temperature=0.0,
        native_structured_output=native_structured_output,
        settings=settings or {},
    )


def make_anthropic() -> Anthropic:
    return Anthropic(
        model=env("ZONIX_MODEL"),
        api_key=env("ZONIX_API_KEY"),
        base_url=env("ZONIX_ANTHROPIC_BASE_URL"),
        temperature=0.0,
        max_tokens=1024,
    )


def planning_prompt() -> str:
    return (
        "Return one compact JSON object only. "
        "Use goal, files, and steps to describe a concrete implementation plan."
    )


async def run_step(name: str, fn: Callable[[], Any]) -> Any:
    print(f"[RUN] {name}")
    value = fn()
    if inspect.isawaitable(value):
        value = await value
    print(f"[PASS] {name}")
    return value


async def structured_openai_case() -> Any:
    planner = agent(
        "smoke_planner",
        role="Code task planning",
        model=make_openai(),
        output=Plan,
    ).prompt(planning_prompt())

    output = await planner("Add captcha validation to a login page.")
    expect(isinstance(output, Plan), "__call__ did not return Plan")
    expect(output.steps, "Plan steps are empty")

    result = await planner.run("Add rate limiting to the login endpoint.")
    expect(isinstance(result.output, Plan), ".run() did not return Plan output")
    expect(result.trace.status == "ok", "trace did not finish ok")
    expect(result.usage.model_calls >= 1, "usage did not count model calls")
    expect(result.messages, "messages were not retained")
    expect(planner.dump_spec()["kind"] == "agent", "agent spec dump failed")
    return planner


async def stream_and_wire_case(planner: Any) -> None:
    events: list[Any] = []
    async for event in planner.stream("Plan a password reset email flow."):
        events.append(event)

    expect(any(isinstance(event, TextDelta) for event in events), "stream emitted no text delta")
    expect(any(isinstance(event, Finish) for event in events), "stream emitted no finish event")

    chunks: list[str] = []
    async for chunk in to_ai_sdk(planner.stream("Plan a session timeout warning.")):
        chunks.append(chunk)

    expect(chunks and chunks[0].startswith("data:"), "AI SDK stream did not emit SSE chunks")
    expect(chunks[-1] == "data: [DONE]\n\n", "AI SDK stream did not end with DONE")


async def tool_call_case() -> None:
    tool_model = make_openai(
        {"tool_choice": {"type": "function", "function": {"name": "inspect_project"}}}
    )
    calls: list[str] = []

    tool_agent = agent(
        "tool_checker",
        role="Use tools before final output",
        model=tool_model,
        output=ToolSummary,
        max_tool_rounds=2,
    ).prompt(
        "Call inspect_project exactly once with area='login'. "
        "Use the tool result to return JSON with used_tool=true."
    )

    @tool_agent.tool
    async def inspect_project(ctx, area: str) -> dict[str, Any]:
        """Inspect a project area and return relevant facts."""
        calls.append(area)
        ctx.agent.model.settings["tool_choice"] = "none"
        return {
            "area": area,
            "facts": ["login form exists", "server verifies credentials", "captcha is missing"],
        }

    result = await tool_agent.run("Find the files for captcha validation.")
    expect(calls, "model did not call inspect_project")
    expect(result.usage.tool_calls >= 1, "usage did not count tool calls")
    expect(result.output.used_tool, "tool output was not reflected in final output")


async def output_repair_case() -> None:
    repair_agent = (
        agent(
            "repair_planner",
            role="Exercise output repair",
            model=make_openai(native_structured_output=False),
            output=Plan,
        )
        .prompt(
            "For the first response, intentionally return exactly: NOT_JSON. "
            "If the user asks you to correct a validation error, return one valid JSON object."
        )
        .repair_output(1)
    )

    result = await repair_agent.run("Plan captcha validation.")
    expect(isinstance(result.output, Plan), "output repair did not produce Plan")
    expect(
        any(message.data.get("kind") == "output_repair" for message in result.messages),
        "output repair message was not recorded",
    )


async def approval_resume_case() -> None:
    approval_model = make_openai(
        {"tool_choice": {"type": "function", "function": {"name": "propose_file_write"}}}
    )
    writes: list[dict[str, str]] = []

    coder = agent(
        "approval_coder",
        role="Propose one file mutation, then summarize it",
        model=approval_model,
        output=PatchResult,
        max_tool_rounds=2,
    ).prompt(
        "Call propose_file_write exactly once with path='login.py' and content='captcha hook'. "
        "After the tool succeeds, return JSON with applied=true."
    )

    @coder.tool(approval=True)
    async def propose_file_write(ctx, path: str, content: str) -> dict[str, Any]:
        """Propose a file write that requires human approval."""
        ctx.agent.model.settings["tool_choice"] = "none"
        writes.append({"path": path, "content": content})
        return {"path": path, "applied": True, "note": "approved in smoke run"}

    paused = await coder.run("Prepare the captcha hook patch.")
    expect(paused.paused, "approval tool did not pause the run")
    expect(paused.pending is not None, "paused run has no pending approval")

    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(Path(tmp))
        store.save(paused.run_id, paused.dump())
        loaded = store.load(paused.run_id)
        expect(loaded["status"] == "paused", "checkpoint did not persist paused state")

    current = paused
    for _ in range(3):
        if not current.paused:
            break
        current = await current.resume(approve=True, input=current.pending.input)

    expect(not current.paused, "resume did not finish the run")
    expect(writes, "approved tool did not execute")
    expect(current.output.applied, "resumed output was not applied")


async def retry_and_timeout_case() -> None:
    retry_model = make_openai(
        {"tool_choice": {"type": "function", "function": {"name": "unstable_lookup"}}}
    )
    attempts = 0

    retry_agent = (
        agent(
            "retry_checker",
            role="Retry a transient tool failure",
            model=retry_model,
            output=ToolSummary,
            max_tool_rounds=2,
        )
        .prompt(
            "Call unstable_lookup exactly once with query='login'. "
            "When it succeeds, return JSON with used_tool=true."
        )
        .retry(1, on=ToolError)
    )

    @retry_agent.tool
    async def unstable_lookup(ctx, query: str) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient lookup failure")
        ctx.agent.model.settings["tool_choice"] = "none"
        return {"area": query, "facts": ["retry succeeded"]}

    retry_result = await retry_agent.run("Exercise retry handling.")
    expect(attempts == 2, "retry did not rerun the failing tool")
    expect(retry_result.output.used_tool, "retry final output did not validate")

    fallback_agent = agent(
        "fallback_planner",
        role="Fallback planner",
        model=make_openai(),
        output=Plan,
    ).prompt(planning_prompt())
    timeout_agent = (
        agent(
            "timeout_primary",
            role="This call should time out quickly",
            model=make_openai(),
            output=Plan,
        )
        .prompt(planning_prompt())
        .timeout(0.001)
        .fallback(fallback_agent)
    )
    fallback_output = await timeout_agent("Plan account lockout after repeated failures.")
    expect(isinstance(fallback_output, Plan), "fallback did not return Plan")


async def graph_runtime_case() -> None:
    seed = FunctionNode("seed", lambda x, st: {"task": x, "count": 0})
    security = FunctionNode("security", lambda x, st: {"name": "security", "risk": "high"})
    performance = FunctionNode("performance", lambda x, st: {"name": "performance", "risk": "low"})
    merge = FunctionNode("merge", lambda xs, st: {"reviews": xs, "risk": "high", "count": 0})
    gate = FunctionNode("human_gate", lambda x, st: {**x, "approved": True})
    increment = FunctionNode("increment", lambda x, st: {**x, "count": x["count"] + 1})

    flow = (
        workflow("graph_flow")
        .start(seed)
        .parallel(security, performance)
        .join(merge)
        .branch(lambda x: x["risk"] == "high", then=gate)
        .loop(increment, until=lambda x: x["count"] >= 2, max_iters=3)
        .build()
    )
    flow_output = await flow.solve("captcha project")
    expect(flow_output["approved"], "branch did not choose gate")
    expect(flow_output["count"] == 2, "loop did not stop on condition")

    alpha = FunctionNode("alpha", lambda x, st: {"done": "alpha", "input": x})

    def choose(task: Any, state: RunState) -> Route:
        if isinstance(task, dict) and task.get("done"):
            return Route(done=True)
        return Route(next="alpha")

    routed = team("rule_team").add(alpha).route(router("rule_router", choose)).build(max_steps=2)
    team_output = await routed.solve("route this task")
    expect(team_output["done"] == "alpha", "team router did not invoke selected node")


async def memory_case() -> None:
    class Store:
        def search(self, query: Any, k: int) -> list[str]:
            return [f"remembered context for {query}"]

    session = Session(memory=[Window(size=2), Vector(store=Store())])
    await session.remember(Message(role="user", content="Earlier: project uses FastAPI."))
    await session.remember(Message(role="assistant", content="Earlier: login route is /auth/login."))

    memory_agent = agent(
        "memory_planner",
        role="Use recalled project context",
        model=make_openai(),
        output=Plan,
        memory=[Summarize(over=80, keep=1)],
    ).prompt(planning_prompt())

    output = await memory_agent("Plan captcha work using remembered context.", session=session)
    expect(isinstance(output, Plan), "memory agent did not return Plan")
    expect(len(session.history) >= 4, "session did not remember new messages")


async def real_workflow_case() -> None:
    planner = agent(
        "wf_planner",
        role="First stage",
        model=make_openai(),
        output=StageResult,
    ).prompt("Return JSON with stage='plan' and a short summary.")
    reviewer = agent(
        "wf_reviewer",
        role="Second stage",
        model=make_openai(),
        output=StageResult,
    ).prompt("Return JSON with stage='review' and a short summary of the input.")

    flow = workflow("real_provider_flow").start(planner).then(reviewer).build()
    output = await flow.solve("Add captcha validation.")
    expect(output.stage == "review", "real provider workflow did not reach reviewer")


async def anthropic_structured_case() -> None:
    if not anthropic_base_url():
        print("[SKIP] anthropic_structured_case: ZONIX_ANTHROPIC_BASE_URL is not set")
        return

    planner = agent(
        "anthropic_planner",
        role="Code task planning",
        model=make_anthropic(),
        output=Plan,
    ).prompt(planning_prompt())

    output = await planner("Plan captcha validation for a login page.")
    expect(isinstance(output, Plan), "Anthropic-compatible adapter did not return Plan")


async def main() -> None:
    planner = await run_step("OpenAI-compatible structured agent", structured_openai_case)
    await run_step("stream events and AI SDK wire adapter", lambda: stream_and_wire_case(planner))
    await run_step("tool calls and ctx injection", tool_call_case)
    await run_step("structured output repair", output_repair_case)
    await run_step("approval pause, checkpoint, resume", approval_resume_case)
    await run_step("retry, timeout, and fallback", retry_and_timeout_case)
    await run_step("workflow graph, parallel, branch, loop, team router", graph_runtime_case)
    await run_step("session memory strategies", memory_case)
    await run_step("real provider workflow", real_workflow_case)
    await run_step("Anthropic-compatible structured output", anthropic_structured_case)
    print("[DONE] zonix smoke coverage complete")


if __name__ == "__main__":
    asyncio.run(main())
