from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import PendingApproval


class ZonixError(Exception):
    """Base error for Zonix."""


class ModelError(ZonixError):
    """Raised when a model adapter cannot complete a request."""


class ToolError(ZonixError):
    """Raised when a tool fails."""


class OutputValidationError(ZonixError):
    """Raised when a model response cannot be validated as the requested output."""


class MaxStepsExceeded(ZonixError):
    """Raised when a team router exceeds its step budget."""


class ToolApprovalRejected(ZonixError):
    """Raised when a paused tool call is resumed with approval=False."""


@dataclass
class RunPaused(ZonixError):
    pending: PendingApproval
    snapshot: dict[str, Any]

    def __str__(self) -> str:
        return f"Run paused for approval: {self.pending.tool}"
