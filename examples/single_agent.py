import asyncio

from pydantic import BaseModel

from zonix import agent
from zonix.models import StaticModel


class Plan(BaseModel):
    goal: str
    files: list[str]
    steps: list[str]


planner = (
    agent(
        "planner",
        role="Plan code work",
        model=StaticModel(output={"goal": "demo", "files": ["app.py"], "steps": ["inspect", "edit"]}),
        output=Plan,
    )
    .prompt("Return a concise implementation plan.")
)


async def main() -> None:
    plan = await planner("add captcha to login")
    print(plan)

    result = await planner.run("add captcha to login")
    print(result.dump()["usage"])


if __name__ == "__main__":
    asyncio.run(main())
