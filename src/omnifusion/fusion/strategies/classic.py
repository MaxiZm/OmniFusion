from __future__ import annotations

from omnifusion.fusion.runtime.context import RunContext
from omnifusion.fusion.runtime.strategy import FusionStrategy, StrategyResult


class ClassicStrategy(FusionStrategy):
    key = "B"

    async def execute(self, ctx: RunContext) -> StrategyResult:
        from omnifusion.fusion.runtime import registry as registry_mod

        payload = await registry_mod._execute_classic(
            ctx.run_id,
            ctx.preset,
            ctx.request,
            ctx.key_hash,
        )
        return StrategyResult(payload=payload)
