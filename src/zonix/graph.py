from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .serialization import to_jsonable


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    kind: str = "node"
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    label: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphSpec:
    name: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    direction: str = "TD"

    def dump(self) -> dict[str, Any]:
        return to_jsonable(self)

    def mermaid(self) -> str:
        lines = [f"flowchart {self.direction}"]
        for node in self.nodes:
            lines.append(f"  {node.id}[{_quote_mermaid(node.label)}]")
        for edge in self.edges:
            label = f"|{_escape_mermaid(edge.label)}|" if edge.label else ""
            lines.append(f"  {edge.source} -->{label} {edge.target}")
        return "\n".join(lines) + "\n"

    def dot(self) -> str:
        lines = [f'digraph "{_escape_dot(self.name)}" {{', "  rankdir=TB;"]
        for node in self.nodes:
            lines.append(f'  "{node.id}" [label="{_escape_dot(node.label)}"];')
        for edge in self.edges:
            label = f' [label="{_escape_dot(edge.label)}"]' if edge.label else ""
            lines.append(f'  "{edge.source}" -> "{edge.target}"{label};')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        suffix = target.suffix.lower()
        target.parent.mkdir(parents=True, exist_ok=True)
        if suffix in {".mmd", ".mermaid"}:
            target.write_text(self.mermaid(), encoding="utf-8")
            return target
        if suffix == ".md":
            target.write_text(f"```mermaid\n{self.mermaid()}```\n", encoding="utf-8")
            return target
        if suffix == ".dot":
            target.write_text(self.dot(), encoding="utf-8")
            return target
        if suffix in {".png", ".svg", ".pdf"}:
            return self._save_image(target, suffix[1:])
        raise ValueError("Graph path must end with .mmd, .mermaid, .md, .dot, .png, .svg, or .pdf.")

    def render_png(self, path: str | Path) -> Path:
        return self.save(path)

    def _save_image(self, target: Path, format: str) -> Path:
        try:
            from graphviz import Source
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install zonix[viz] to export workflow/team graphs as images."
            ) from exc
        try:
            rendered = Source(self.dot()).render(
                filename=target.with_suffix("").name,
                directory=str(target.parent),
                format=format,
                cleanup=True,
            )
        except Exception as exc:  # pragma: no cover - depends on system graphviz
            raise RuntimeError(
                "Graph image export requires the Graphviz system executable on PATH."
            ) from exc
        return Path(rendered)


def safe_graph_id(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in value)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned


def _quote_mermaid(value: str) -> str:
    return f'"{_escape_mermaid(value)}"'


def _escape_mermaid(value: str | None) -> str:
    return (value or "").replace('"', '\\"').replace("\n", "<br/>")


def _escape_dot(value: str | None) -> str:
    return (value or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
