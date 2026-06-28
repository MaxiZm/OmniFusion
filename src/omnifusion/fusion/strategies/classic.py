from __future__ import annotations

from omnifusion.fusion.runtime.context import RunContext
from omnifusion.fusion.runtime.strategy import FusionStrategy, StrategyResult


class ClassicStrategy(FusionStrategy):
    key = "B"

    async def execute(self, ctx: RunContext) -> StrategyResult:
        from omnifusion.fusion.openfusion_runtime import (
            execute_openfusion_hybrid,
            needs_hybrid_runtime,
        )
        from omnifusion.fusion.runtime import registry as registry_mod

        if needs_hybrid_runtime(ctx.preset):
            payload = await execute_openfusion_hybrid(
                ctx.run_id,
                ctx.preset,
                ctx.request,
                ctx.key_hash,
            )
            return StrategyResult(payload=payload)

        payload = await registry_mod._execute_classic(
            ctx.run_id,
            ctx.preset,
            ctx.request,
            ctx.key_hash,
        )
        return StrategyResult(payload=payload)
