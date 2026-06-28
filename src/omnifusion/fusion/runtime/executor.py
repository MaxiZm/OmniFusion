from __future__ import annotations

import asyncio
from typing import Any

from ...budget.ledger import reconcile_budget, reserve_budget
from ...llm.client import llm_client
from ...providers.pricing import (
    calculate_actual_cost,
    estimate_call_cost,
    estimate_tokens,
    get_model_cost_estimate,
    usd_to_micro,
)


class BudgetedExecutor:
    def __init__(self, run_id: str):
        self.run_id = run_id

    async def call(
        self,
        stage: str,
        *,
        provider_id: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        **kwargs,
    ):
        reservation_id = await self._reserve(stage, model, messages, max_tokens)
        actual_cost = 0.0
        reconciled = False
        try:
            response = await llm_client.acompletion(
                provider_id=provider_id,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                **kwargs,
            )
            actual_cost = calculate_actual_cost(response, model)
            setattr(response, "_omnifusion_cost_usd", actual_cost)
            await self._reconcile(reservation_id, actual_cost)
            reconciled = True
            return response
        finally:
            if not reconciled:
                await self._reconcile(reservation_id, actual_cost)

    async def stream(
        self,
        stage: str,
        *,
        provider_id: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        **kwargs,
    ):
        reservation_id = await self._reserve(stage, model, messages, max_tokens)
        upstream = None
        stream_returned = False
        try:
            upstream = await llm_client.acompletion(
                provider_id=provider_id,
                model=model,
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
                **kwargs,
            )
            stream_returned = True
            return BudgetedStream(
                reservation_id=reservation_id,
                model=model,
                messages=messages,
                upstream=upstream,
            )
        finally:
            if not stream_returned:
                await self._reconcile(reservation_id, 0.0)

    async def _reserve(
        self,
        stage: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> str:
        estimated = estimate_call_cost(model, messages, max_tokens)
        reserve_micro_usd = max(1, int(estimated * 1_000_000))
        return await reserve_budget(self.run_id, stage, reserve_micro_usd)

    async def _reconcile(self, reservation_id: str, actual_cost_usd: float) -> None:
        await asyncio.shield(
            reconcile_budget(reservation_id, usd_to_micro(actual_cost_usd))
        )


class BudgetedStream:
    def __init__(
        self,
        *,
        reservation_id: str,
        model: str,
        messages: list[dict[str, Any]],
        upstream,
    ):
        self.reservation_id = reservation_id
        self.model = model
        self.messages = messages
        self.upstream = upstream
        self.completion_text = ""
        self.usage = None
        self.cost_usd = 0.0
        self._reconciled = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self.upstream.__anext__()
        except StopAsyncIteration:
            await self._cleanup()
            raise
        except BaseException:
            await self._cleanup()
            raise

        self._record_chunk(chunk)
        return chunk

    async def aclose(self):
        upstream_close = getattr(self.upstream, "aclose", None)
        if upstream_close is not None:
            await upstream_close()
        await self._cleanup()

    def _record_chunk(self, chunk) -> None:
        if getattr(chunk, "usage", None):
            self.usage = chunk.usage
        if not getattr(chunk, "choices", None):
            return
        delta = getattr(chunk.choices[0], "delta", None)
        content = getattr(delta, "content", None)
        if content:
            self.completion_text += content

    async def _cleanup(self) -> None:
        if self._reconciled:
            return

        prompt_tokens = estimate_tokens(self.model, self.messages)
        if self.usage is not None:
            prompt_tokens = int(getattr(self.usage, "prompt_tokens", prompt_tokens) or 0)
            completion_tokens = int(
                getattr(self.usage, "completion_tokens", 0) or 0
            )
        else:
            completion_tokens = max(1, len(self.completion_text) // 4)
        self.cost_usd = get_model_cost_estimate(
            self.model,
            prompt_tokens,
            max(1, completion_tokens),
        )
        await asyncio.shield(
            reconcile_budget(self.reservation_id, usd_to_micro(self.cost_usd))
        )
        self._reconciled = True
