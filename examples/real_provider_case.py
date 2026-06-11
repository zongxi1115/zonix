import asyncio
import os

from pydantic import BaseModel

from zonix import TextDelta, agent
from zonix.models import Anthropic, OpenAI


class Plan(BaseModel):
    goal: str
    files: list[str]
    steps: list[str]


def build_model():
    provider = os.environ.get("ZONIX_PROVIDER", "openai").lower()
    api_key = os.environ["ZONIX_API_KEY"]
    model = os.environ["ZONIX_MODEL"]
    base_url = os.environ["ZONIX_BASE_URL"]

    if provider == "anthropic":
        return Anthropic(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=1024,
        )

    return OpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
    )


planner = (
    agent(
        "planner",
        role="Code task planning",
        model=build_model(),
        output=Plan,
    )
    .prompt(
        "Split the user request into a concrete code implementation plan. "
        "Return one compact JSON object only, without Markdown or prose."
    )
)


async def main() -> None:
    task = "Add captcha validation to a login page. Plan the files and steps."

    result = await planner.run(task)
    print("provider_output:")
    print(result.output.model_dump_json(indent=2))
    print("usage:")
    print(result.usage.model_dump_json(indent=2))

    print("stream_text_prefix:")
    chunks: list[str] = []
    async for event in planner.stream(task):
        if isinstance(event, TextDelta):
            chunks.append(event.delta)
    print("".join(chunks)[:240])


if __name__ == "__main__":
    asyncio.run(main())
