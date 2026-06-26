from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from fastapi.responses import StreamingResponse

from .context import RunContext


@dataclass
class StrategyResult:
    """Typed envelope a strategy returns instead of leaking a raw dict/StreamingResponse
    across the runtime boundary. `payload` is the OpenAI-shaped non-stream dict or a
    StreamingResponse; the API layer unwraps it via execute_strategy."""

    payload: Any

    @property
    def streaming(self) -> bool:
        return isinstance(self.payload, StreamingResponse)


class FusionStrategy(ABC):
    key: str

    @abstractmethod
    async def execute(self, ctx: RunContext) -> StrategyResult:
        raise NotImplementedError
