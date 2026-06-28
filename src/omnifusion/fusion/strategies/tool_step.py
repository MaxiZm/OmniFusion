from __future__ import annotations

from omnifusion.fusion.runtime.context import RunContext
from omnifusion.fusion.runtime.strategy import FusionStrategy, StrategyResult


class ToolStepStrategy(FusionStrategy):
    key = "_tool_step"

    async def execute(self, ctx: RunContext) -> StrategyResult:
        from omnifusion.fusion.runtime import registry as registry_mod

        payload = await registry_mod._execute_tool_step(
            ctx.run_id,
            ctx.preset,
            ctx.request,
            ctx.key_hash,
        )
        return StrategyResult(payload=payload)
