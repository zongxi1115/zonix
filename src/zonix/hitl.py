from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .serialization import to_jsonable
from .types import PendingApproval, ToolCall


def approval_key(tool: str, input: dict[str, Any]) -> str:
    payload = json.dumps(to_jsonable(input), ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{tool}:{digest}"


def pending_from_call(call: ToolCall) -> PendingApproval:
    return PendingApproval(
        call_id=call.call_id,
        tool=call.tool,
        input=call.input,
        approval_key=approval_key(call.tool, call.input),
    )


@dataclass
class CheckpointStore:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def save(self, run_id: str, snapshot: dict[str, Any]) -> Path:
        path = self.path_for(run_id)
        path.write_text(
            json.dumps(to_jsonable(snapshot), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def load(self, run_id: str) -> dict[str, Any]:
        return json.loads(self.path_for(run_id).read_text(encoding="utf-8"))
