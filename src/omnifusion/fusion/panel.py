import asyncio
from typing import List
from .types import Preset, PanelResult
from ..llm.client import llm_client
from ..api.errors import InsufficientPanelError, BudgetExceededError
from ..providers.pricing import calculate_actual_cost, estimate_call_cost, usd_to_micro
from ..budget.ledger import reserve_budget, reconcile_budget


async def run_panelist(
    run_id: str, model: str, preset: Preset, messages: list
) -> PanelResult:
    # 1. Convert messages to dicts
    dict_messages = [
        m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
        for m in messages
    ]

    # 2. Dynamic budget reservation (reserve-then-reconcile in microdollars)
    cost_usd = estimate_call_cost(model, dict_messages, preset.panel.max_tokens)
    reserve_micro_usd = max(
        1, int(cost_usd * 1_000_000)
    )  # Ensure at least 1 microdollar reservation

    result = PanelResult(model=model, status="error")
    reservation_id = None

    try:
        # Fix #3: reserve_budget is now INSIDE the try block so BudgetExceededError
        # propagates as-is rather than being captured as a generic exception.
        reservation_id = await reserve_budget(run_id, f"panel/{model}", reserve_micro_usd)

        # 3. Call litellm wrapper
        response = await llm_client.acompletion(
            provider_id="default",
            model=model,
            messages=dict_messages,
            timeout=preset.panel.timeout,
            max_tokens=preset.panel.max_tokens,
        )

        # 4. Handle response
        result.status = "ok"
        result.content = response.choices[0].message.content
        result.usage = getattr(response, "usage", None)
        result.cost_usd = calculate_actual_cost(response, model)

    except BudgetExceededError:
        # Re-raise budget errors so run_panel can distinguish them from ordinary failures
        # and surface the 402 instead of silently downgrading to a 503.
        raise
    except asyncio.TimeoutError:
        result.status = "timeout"
    except Exception as e:
        if "rate limit" in str(e).lower() or "429" in str(e):
            result.status = "rate_limited"
        else:
            result.status = "error"
    finally:
        # Reconcile actual cost (or release reservation if we bailed out)
        if reservation_id is not None:
            actual_micro_usd = usd_to_micro(result.cost_usd)
            await asyncio.shield(reconcile_budget(reservation_id, actual_micro_usd))

    return result


async def run_panel(
    run_id: str, preset: Preset, messages: list, min_success: int = 1
) -> List[PanelResult]:
    tasks = []
    # Cap panel at max 8
    models = preset.panel_models[:8]

    for model in models:
        tasks.append(run_panelist(run_id, model, preset, messages))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    panel_results = []
    ok_count = 0
    for r in results:
        if isinstance(r, BudgetExceededError):
            # Fix #3: Re-raise BudgetExceededError so the caller gets the 402,
            # not a swallowed 503.
            raise r
        elif isinstance(r, PanelResult):
            panel_results.append(r)
            if r.status == "ok":
                ok_count += 1
        else:
            panel_results.append(PanelResult(model="unknown", status="error"))

    if ok_count < min_success:
        raise InsufficientPanelError(
            f"Panel only got {ok_count} successes, needed {min_success}"
        )

    return panel_results
