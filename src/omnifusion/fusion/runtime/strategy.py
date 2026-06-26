from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .context import RunContext


class FusionStrategy(ABC):
    key: str

    @abstractmethod
    async def execute(self, ctx: RunContext) -> Any:
        raise NotImplementedError
