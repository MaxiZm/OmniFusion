from __future__ import annotations

from dataclasses import dataclass, field

from omnifusion.api.schemas import ChatCompletionRequest
from omnifusion.fusion.types import Preset

from .artifacts import ArtifactGraph


@dataclass
class RunContext:
    run_id: str
    preset: Preset
    request: ChatCompletionRequest
    key_hash: str
    artifacts: ArtifactGraph = field(default_factory=ArtifactGraph)
