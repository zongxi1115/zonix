from __future__ import annotations

from collections.abc import Callable
from typing import Any


Observer = Callable[[str, dict[str, Any]], None]
_observer: Observer | None = None


def configure(observer: Observer | None) -> None:
    global _observer
    _observer = observer


def emit(name: str, **attributes: Any) -> None:
    if _observer is not None:
        _observer(name, attributes)
