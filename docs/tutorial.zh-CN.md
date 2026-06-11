# Zonix 多章节教程

这份教程按真实开发路径组织：先定义一个显式 Agent，再逐步加入结构化输出、工具调用、流式事件、workflow/team 编排、记忆、人审续跑和真实 provider smoke 验证。

## 第 1 章：Zonix 的核心模型

Zonix 的目标是让业务代码一眼能看出四件事：

- 这个 Agent 叫什么，负责什么角色。
- 它使用哪个模型对象，而不是一段魔法字符串。
- 它能调用哪些工具。
- 它最终返回什么 Pydantic 类型。

最小定义如下：

```python
from pydantic import BaseModel

from zonix import agent
from zonix.models import OpenAI


class Plan(BaseModel):
    goal: str
    files: list[str]
    steps: list[str]


planner = (
    agent(
        "planner",
        role="Code task planning",
        model=OpenAI(
            model="your-model",
            api_key=os.environ["ZONIX_API_KEY"],
            base_url=os.environ["ZONIX_BASE_URL"],
        ),
        output=Plan,
    )
    .prompt("Return a concrete implementation plan.")
)
```

调用分三层：

```python
plan = await planner(task)
run = await planner.run(task)

async for event in planner.stream(task):
    ...
```

`planner(task)` 只拿结构化结果，`.run()` 拿完整 trace/usage/messages/scratch，`.stream()` 拿类型化事件流。

## 第 2 章：连接真实模型 Provider

OpenAI-compatible 网关：

```python
model = OpenAI(
    model=os.environ["ZONIX_MODEL"],
    api_key=os.environ["ZONIX_API_KEY"],
    base_url=os.environ["ZONIX_OPENAI_BASE_URL"],
    temperature=0.0,
)
```

Anthropic-compatible 网关：

```python
from zonix.models import Anthropic

model = Anthropic(
    model=os.environ["ZONIX_MODEL"],
    api_key=os.environ["ZONIX_API_KEY"],
    base_url=os.environ["ZONIX_ANTHROPIC_BASE_URL"],
    temperature=0.0,
)
```

不要把 API key、内部网关地址、测试模型名写进仓库。Zonix 的示例全部通过环境变量读取。

## 第 3 章：结构化输出的正确姿势

只在 prompt 里写“请返回 JSON”不够稳。Zonix 现在按三层策略处理结构化输出：

1. OpenAI-compatible 优先使用原生 `response_format={"type": "json_schema"}`，并开启 `strict: true`。
2. Anthropic-compatible 将最终输出声明成一个 `zonix_final_output` tool，用 `input_schema` 约束结构，并在没有业务工具时用 `tool_choice` 强制调用。
3. 如果 provider 不支持原生结构化输出，Zonix 退回 prompt 约束，再用 Pydantic v2 校验。

最后一层永远是 Pydantic：

```python
class Review(BaseModel):
    risk: str
    notes: list[str]


reviewer = agent("reviewer", model=model, output=Review).prompt(
    "Review the patch and return one JSON object."
)
```

如果模型返回了 Markdown 包裹的 JSON，Zonix 会提取明确的 JSON 对象或数组再校验。如果仍然不合法，默认会把校验错误反馈给模型做一次 repair：

```python
reviewer.repair_output(1)
```

你可以关闭或调高次数：

```python
reviewer.repair_output(0)
reviewer.repair_output(2)
```

实践建议：

- 生产场景优先使用 provider 原生 schema 或 tool-output。
- prompt 里明确“one valid JSON value only, no Markdown or prose”。
- `temperature=0.0` 更适合 schema 任务。
- Pydantic 类型保持短而明确，少用过深嵌套。
- 对外部网关保留 prompt fallback，因为不是所有兼容服务都完整支持 `response_format`。

## 第 3.5 章：用户手动构建消息历史

如果历史存储在你自己的数据库里，可以手动构建 transcript，再传给当前 run：

```python
from zonix import (
    ToolCall,
    assistant_message,
    assistant_tool_call_message,
    tool_message,
    user_message,
)

call = ToolCall(call_id="call_1", tool="search_code", input={"query": "login"})

history = [
    user_message("先查一下登录逻辑。"),
    assistant_tool_call_message(call),
    tool_message("call_1", "search_code", {"files": ["auth/login.py"]}),
    assistant_message("登录逻辑在 auth/login.py。"),
]

plan = await planner("继续规划验证码改造。", message_history=history)
```

`message_history` 支持 `Message` 对象，也支持同结构的 dict。这个参数也能传给
`agent.run`、`agent.stream`、`workflow.solve/run/stream` 和 `team.solve/run/stream`。

这和 `Session` 不冲突：`message_history` 是你显式传入的外部历史，`Session`
是 Zonix 管理的历史和记忆策略。

## 第 4 章：工具调用

批量挂工具：

```python
def read_tree(root: str) -> dict:
    """Read a project tree."""
    return {"root": root, "files": ["login.py", "captcha.py"]}


planner = agent("planner", model=model, output=Plan).use(read_tree)
```

需要访问 ctx 时，用装饰器：

```python
@planner.tool
async def inspect_project(ctx, area: str) -> dict:
    return ctx.deps.repo.inspect(area)
```

Zonix 会从函数签名和 docstring 生成工具 schema。工具执行时会收到 `ToolContext`，里面有：

- `deps`：调用时传入的业务依赖。
- `usage`：共享 usage 累加器。
- `state`：当前 RunState。
- `agent`：当前 Agent。

## 第 5 章：人审、暂停和续跑

会改文件、数据库或外部系统的工具应该加 `approval=True`：

```python
@coder.tool(approval=True)
async def write_file(ctx, path: str, content: str) -> bool:
    return ctx.deps.repo.write(path, content)
```

运行时：

```python
run = await coder.run("edit login.py")

if run.paused:
    print(run.pending)
    run = await run.resume(approve=True)
```

你也可以保存 checkpoint：

```python
from pathlib import Path
from zonix.hitl import CheckpointStore

store = CheckpointStore(Path(".zonix-runs"))
store.save(run.run_id, run.dump())
snapshot = store.load(run.run_id)
```

## 第 6 章：流式事件和前端协议

Zonix 的流不是字符串，而是类型化事件：

```python
from zonix import TextDelta, ToolInputAvailable

async for event in planner.stream(task):
    if isinstance(event, TextDelta):
        print(event.delta, end="")
    elif isinstance(event, ToolInputAvailable):
        print(event.tool, event.input)
```

映射到 Vercel AI SDK Data Stream：

```python
from zonix.wire.ai_sdk import to_ai_sdk

async for chunk in to_ai_sdk(planner.stream(task)):
    yield chunk
```

HTTP 响应头：

```text
content-type: text/event-stream
x-vercel-ai-ui-message-stream: v1
```

## 第 7 章：Workflow 固定流程

workflow 适合固定步骤：

```python
flow = (
    workflow("code_flow")
    .start(planner)
    .then(coder)
    .then(reviewer)
    .build()
)

result = await flow.solve("add captcha validation")
```

高级节点：

```python
flow = (
    workflow("review")
    .start(planner)
    .parallel(security_review, perf_review)
    .join(merge_reviews)
    .branch(lambda review: review.risk == "high", then=human_gate)
    .loop(coder, until=lambda patch: patch.tests_pass, max_iters=3)
    .build()
)
```

每个节点都会进入 trace，`ctx/usage/bus` 自动透传，`messages` 默认按节点隔离。

## 第 8 章：Team 和 Router 动态调度

team 适合运行时动态选择下一个 Agent：

```python
from zonix import router, team
from zonix.types import Route


def choose(task, state) -> Route:
    if "review" in str(task).lower():
        return Route(next="reviewer")
    return Route(next="coder")


code_team = (
    team("code_team")
    .add(planner, coder, reviewer)
    .route(router("rules", choose))
    .build(max_steps=4)
)

answer = await code_team.solve("review auth changes")
```

router 可以是规则函数，也可以是另一个 Agent。契约固定为返回 `Route(next=..., done=..., input=...)`。

## 第 9 章：Session 和记忆

短期窗口：

```python
from zonix.memory import Session, Window

session = Session(memory=[Window(size=20)])
answer = await assistant("continue", session=session)
```

组合记忆：

```python
from zonix.memory import Summarize, Vector

memory = Summarize(over=170_000, keep=20) + Vector(store=vector_store)
assistant = agent("assistant", model=model, memory=memory)
```

记忆策略只处理历史消息，不混入当前 run。当前输入仍由引擎单独加入。

## 第 10 章：重试、超时和 fallback

工具或模型边界可以挂运行策略：

```python
coder = (
    agent("coder", model=model, output=Patch)
    .use(read_file, write_file)
    .retry(2, on=ToolError)
    .timeout(60)
    .fallback(simple_coder)
)
```

含义：

- `retry(2, on=ToolError)`：节点失败后最多重试 2 次。
- `timeout(60)`：单节点 60 秒超时。
- `fallback(simple_coder)`：失败后切到备用节点。

## 第 11 章：真实 Smoke 验证

仓库提供真实 provider smoke 脚本：

```bash
export ZONIX_API_KEY="..."
export ZONIX_MODEL="your-model"
export ZONIX_OPENAI_BASE_URL="https://your-openai-compatible-host/v1"
export ZONIX_ANTHROPIC_BASE_URL="https://your-anthropic-compatible-host"
python scripts/smoke_real_provider.py
```

覆盖项：

- OpenAI-compatible 结构化输出、`__call__`、`.run()`、trace、usage。
- `.stream()` 事件和 AI SDK SSE wire adapter。
- 工具调用、ctx 注入、usage 工具计数。
- 输出 repair。
- `approval=True` 暂停、checkpoint、resume。
- retry、timeout、fallback。
- workflow 的 start/then/parallel/join/branch/loop。
- team + router。
- Window/Summarize/Vector/Session。
- Anthropic-compatible 结构化输出。

这条 smoke 需要真实模型凭证，不会使用仓库内置假模型验证 provider 能力。
