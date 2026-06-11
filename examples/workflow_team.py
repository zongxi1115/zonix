import asyncio

from pydantic import BaseModel

from zonix import agent, router, team, workflow
from zonix.models import StaticModel
from zonix.types import Route


class StepOutput(BaseModel):
    value: str


planner = agent("planner", model=StaticModel(output={"value": "planned"}), output=StepOutput)
coder = agent("coder", model=StaticModel(output={"value": "coded"}), output=StepOutput)
reviewer = agent("reviewer", model=StaticModel(output={"value": "reviewed"}), output=StepOutput)

flow = workflow("code_flow").start(planner).then(coder).then(reviewer).build()


def choose(task, state) -> Route:
    if isinstance(task, StepOutput):
        return Route(done=True)
    if "review" in str(task).lower():
        return Route(next="reviewer")
    return Route(next="planner")


code_team = team("code_team").add(planner, coder, reviewer).route(router("rules", choose)).build(max_steps=2)


async def main() -> None:
    print(await flow.solve("ship the feature"))
    print(await code_team.solve("review the patch"))


if __name__ == "__main__":
    asyncio.run(main())
