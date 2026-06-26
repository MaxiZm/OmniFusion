from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArtifactGraph:
    nodes: dict[str, Any] = field(default_factory=dict)

    def add(self, key: str, value: Any) -> None:
        self.nodes[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.nodes.get(key, default)

    def to_trace_metadata(self) -> dict[str, Any]:
        return dict(self.nodes)
