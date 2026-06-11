from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zonix.exceptions import ModelError

from .base import BaseChatModel, ModelRequest, ModelResponse


@dataclass
class Anthropic(BaseChatModel):
    model: str = "claude-4.5-sonnet"
    temperature: float | None = None
    api_key: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"anthropic:{self.model}"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise ModelError(
            "The Anthropic adapter is reserved as a typed provider object. "
            "Implement complete() for your deployment or use OpenAI/StaticModel."
        )
