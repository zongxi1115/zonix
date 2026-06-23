from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from zonix.content import content_text
from zonix.types import Message


class Memory(Protocol):
    async def apply(self, history: list[Message], current: Any, ctx: Any) -> list[Message]:
        ...


def _as_list(memory: Any) -> list[Memory]:
    if memory is None:
        return []
    if isinstance(memory, MemoryStack):
        return list(memory.items)
    if isinstance(memory, list | tuple):
        return list(memory)
    return [memory]


@dataclass
class MemoryStack:
    items: list[Memory] = field(default_factory=list)

    def __add__(self, other: Any) -> MemoryStack:
        return MemoryStack([*self.items, *_as_list(other)])

    async def apply(self, history: list[Message], current: Any, ctx: Any) -> list[Message]:
        messages = list(history)
        for memory in self.items:
            messages = await memory.apply(messages, current, ctx)
        return messages


@dataclass
class Window:
    size: int = 20

    def __add__(self, other: Any) -> MemoryStack:
        return MemoryStack([self]) + other

    async def apply(self, history: list[Message], current: Any, ctx: Any) -> list[Message]:
        return list(history[-self.size :])


@dataclass
class Summarize:
    over: int
    keep: int = 20

    def __add__(self, other: Any) -> MemoryStack:
        return MemoryStack([self]) + other

    async def apply(self, history: list[Message], current: Any, ctx: Any) -> list[Message]:
        total_chars = sum(
            len(content_text(message.content, include_images=True)) for message in history
        )
        if total_chars <= self.over:
            return list(history)
        older = history[: -self.keep]
        recent = history[-self.keep :]
        summary = "Earlier conversation summary: " + " ".join(
            content_text(message.content, include_images=True)[:200] for message in older
        )
        return [Message(role="system", content=summary)] + recent


@dataclass
class Vector:
    store: Any
    k: int = 6

    def __add__(self, other: Any) -> MemoryStack:
        return MemoryStack([self]) + other

    async def apply(self, history: list[Message], current: Any, ctx: Any) -> list[Message]:
        results = await self._search(current)
        if not results:
            return list(history)
        body = "\n".join(str(item) for item in results)
        return [*history, Message(role="system", content=f"Relevant long-term memory:\n{body}")]

    async def _search(self, query: Any) -> list[Any]:
        for name in ("asearch", "asimilarity_search"):
            method = getattr(self.store, name, None)
            if method is not None:
                value = method(query, k=self.k)
                if inspect.isawaitable(value):
                    value = await value
                return list(value)
        for name in ("search", "similarity_search"):
            method = getattr(self.store, name, None)
            if method is not None:
                return list(method(query, k=self.k))
        return []


@dataclass
class Session:
    id: str = field(default_factory=lambda: f"session_{uuid.uuid4().hex}")
    history: list[Message] = field(default_factory=list)
    memory: Any = None

    async def recall(self, current: Any, ctx: Any, memory: Any = None) -> list[Message]:
        strategies = MemoryStack([*_as_list(self.memory), *_as_list(memory)])
        return await strategies.apply(list(self.history), current, ctx)

    async def remember(self, message: Message) -> None:
        self.history.append(message)

    def dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "history": [message.model_dump(mode="json") for message in self.history],
        }
