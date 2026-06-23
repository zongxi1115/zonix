from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zonix.content import content_text
from zonix.types import Usage

from .base import BaseChatModel, ModelRequest, ModelResponse


@dataclass
class Echo(BaseChatModel):
    """Offline model useful for examples and adapter tests."""

    name: str = "echo"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        last = next((m for m in reversed(request.messages) if m.role == "user"), None)
        text = content_text(last.content, include_images=True) if last else ""
        return ModelResponse(text=text, usage=Usage(model_calls=1))


@dataclass
class StaticModel(BaseChatModel):
    """Return a fixed output or text without calling an external provider."""

    output: Any = None
    text: str | None = None
    name: str = "static"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self.text is not None:
            return ModelResponse(text=self.text, output=self.output, usage=Usage(model_calls=1))
        return ModelResponse(output=self.output, usage=Usage(model_calls=1))
