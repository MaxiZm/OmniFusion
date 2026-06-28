from __future__ import annotations

from collections.abc import Iterable

from .context import RunContext
from .strategy import FusionStrategy

StrategyCallable = FusionStrategy


class StrategyRegistry:
    def __init__(self, strategies: Iterable[FusionStrategy] | None = None):
        if strategies is None:
            from ..strategies.classic import ClassicStrategy
            from ..strategies.conductor import ConductorStrategy
            from ..strategies.tool_step import ToolStepStrategy

            strategies = [ClassicStrategy(), ToolStepStrategy(), ConductorStrategy()]
        self._strategies = {strategy.key: strategy for strategy in strategies}

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._strategies))

    def has(self, key: str) -> bool:
        return key in self._strategies

    def register(self, strategy: FusionStrategy) -> None:
        self._strategies[strategy.key] = strategy

    async def execute(self, run_id, preset, body, key_hash):
        strategy_key = "_tool_step" if body.tools else preset.strategy
        strategy = self._strategies.get(strategy_key)
        if strategy is None:
            raise ValueError(f"Unknown fusion strategy: {strategy_key}")
        return await strategy.execute(
            RunContext(
                run_id=run_id,
                preset=preset,
                request=body,
                key_hash=key_hash,
            )
        )


registry = StrategyRegistry()


async def _execute_classic(run_id, preset, body, key_hash):
    from ..orchestrator import run_fusion_classic

    return await run_fusion_classic(run_id, preset, body, key_hash)


async def _execute_tool_step(run_id, preset, body, key_hash):
    from ..tool_orchestrator import run_fusion_with_tools

    return await run_fusion_with_tools(run_id, preset, body, key_hash)


async def execute_strategy(run_id, preset, body, key_hash):
    """Public entry point: run the selected strategy and unwrap its StrategyResult
    envelope into the OpenAI-shaped payload the API layer returns."""
    from .strategy import StrategyResult

    result = await registry.execute(run_id, preset, body, key_hash)
    if isinstance(result, StrategyResult):
        return result.payload
    return result
