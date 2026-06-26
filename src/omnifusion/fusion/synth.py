import asyncio
from typing import List, AsyncGenerator, Union, Any
from .types import Preset, PanelResult, JudgeAnalysis
from ..llm.client import llm_client
from ..budget.ledger import reserve_budget, reconcile_budget
from ..providers.pricing import (
    calculate_actual_cost,
    estimate_call_cost,
    estimate_tokens,
    get_model_cost_estimate,
    usd_to_micro,
)
from ..api.schemas import ChatCompletionRequest


async def run_synthesis(
    run_id: str,
    preset: Preset,
    request: ChatCompletionRequest,
    panel_results: List[PanelResult],
    judge_analysis: JudgeAnalysis,
    context: dict,
) -> Union[AsyncGenerator[Any, None], Any]:

    # 1. Assemble panel answers dictionary
    panel_answers = {}
    for idx, res in enumerate([r for r in panel_results if r.status == "ok"]):
        label = f"MODEL_{chr(65 + idx)}"
        panel_answers[label] = res.content

    # 2. Render system content using prompts template
    from .prompts import render_final_prompt

    system_content = render_final_prompt(panel_answers, judge_analysis, run_id)

    # 3. Assemble messages, preserving original history.
    # If the user sent a system message, merge it into the synthesis system prompt
    # rather than appending it as a duplicate system turn.
    user_system_parts = []
    non_system_messages = []
    for m in request.messages:
        m_dict = m.model_dump(exclude_none=True)
        if m_dict.get("role") == "system":
            user_system_parts.append(m_dict.get("content", ""))
        else:
            non_system_messages.append(m_dict)

    if user_system_parts:
        merged_user_system = "\n".join(user_system_parts)
        system_content = f"{merged_user_system}\n\n{system_content}"

    final_messages = [{"role": "system", "content": system_content}]
    final_messages.extend(non_system_messages)

    # 4. Clamp user request max_tokens to preset's final stage cap (Invariant #9)
    max_tokens = preset.final.max_tokens
    if request.max_tokens is not None:
        max_tokens = min(request.max_tokens, preset.final.max_tokens)

    kwargs = {
        "timeout": preset.final.timeout,
        "max_tokens": max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stop": request.stop,
    }
    # Filter out None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # 5. Dynamic budget reservation (synth stage)
    cost_usd = estimate_call_cost(preset.final_model, final_messages, max_tokens)
    reserve_micro_usd = max(1, int(cost_usd * 1_000_000))
    reservation_id = await reserve_budget(run_id, "final", reserve_micro_usd)

    response = None
    reconciled = False
    stream_returned = False
    try:
        response = await llm_client.acompletion(
            provider_id="default",
            model=preset.final_model,
            messages=final_messages,
            stream=request.stream,
            **kwargs,
        )

        if request.stream:
            class BudgetedSynthesisStream:
                def __init__(self):
                    self._completion_text = ""
                    self._reconciled = False

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    try:
                        chunk = await response.__anext__()
                    except StopAsyncIteration:
                        await self._cleanup()
                        raise
                    except BaseException:
                        await self._cleanup()
                        raise

                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            self._completion_text += delta.content
                    return chunk

                async def aclose(self):
                    upstream_close = getattr(response, "aclose", None)
                    if upstream_close is not None:
                        await upstream_close()
                    await self._cleanup()

                async def _cleanup(self):
                    if self._reconciled:
                        return

                    async def run_cleanup():
                        prompt_tokens = estimate_tokens(preset.final_model, final_messages)
                        approx_completion_tokens = max(1, len(self._completion_text) // 4)
                        actual_cost_usd = get_model_cost_estimate(
                            preset.final_model, prompt_tokens, approx_completion_tokens
                        )
                        context["cost_usd"] = actual_cost_usd
                        await reconcile_budget(
                            reservation_id, usd_to_micro(actual_cost_usd)
                        )

                    await asyncio.shield(run_cleanup())
                    self._reconciled = True

            # The returned stream object is now the only owner of stream reconciliation.
            stream_returned = True
            return BudgetedSynthesisStream()
        else:
            cost_usd = calculate_actual_cost(response, preset.final_model)
            context["cost_usd"] = cost_usd
            async def run_cleanup():
                await reconcile_budget(reservation_id, usd_to_micro(cost_usd))
            await asyncio.shield(run_cleanup())
            reconciled = True
            return response

    finally:
        if not reconciled and not stream_returned:
            async def run_cleanup():
                await reconcile_budget(reservation_id, 0)
            await asyncio.shield(run_cleanup())
