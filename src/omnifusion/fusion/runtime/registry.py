from __future__ import annotations

from typing import Awaitable, Callable

StrategyCallable = Callable[..., Awaitable[object]]


class StrategyRegistry:
    def __init__(self):
        self._keys = {"B", "_tool_step", "conductor"}

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._keys))

    def has(self, key: str) -> bool:
        return key in self._keys


registry = StrategyRegistry()


async def _execute_classic(run_id, preset, body, key_hash):
    from ..orchestrator import run_fusion_classic

    return await run_fusion_classic(run_id, preset, body, key_hash)


async def _execute_tool_step(run_id, preset, body, key_hash):
    from ..tool_orchestrator import run_fusion_with_tools

    return await run_fusion_with_tools(run_id, preset, body, key_hash)


async def _execute_conductor(run_id, preset, body, key_hash):
    from ..strategies.conductor import execute_conductor

    return await execute_conductor(run_id, preset, body, key_hash)


async def execute_strategy(run_id, preset, body, key_hash):
    strategy_key = "_tool_step" if body.tools else preset.strategy
    if strategy_key == "_tool_step":
        return await _execute_tool_step(run_id, preset, body, key_hash)
    if strategy_key == "B":
        return await _execute_classic(run_id, preset, body, key_hash)
    if strategy_key == "conductor":
        return await _execute_conductor(run_id, preset, body, key_hash)
    raise ValueError(f"Unknown fusion strategy: {strategy_key}")
