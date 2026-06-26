import asyncio
from typing import List
from .types import Preset, PanelResult, role_prompt_content
from ..api.errors import InsufficientPanelError, BudgetExceededError
from .runtime.executor import BudgetedExecutor
from .runtime.bandit import select_panel_models


async def run_panelist(
    run_id: str,
    model: str,
    preset: Preset,
    messages: list,
    extra_kwargs: dict | None = None,
) -> PanelResult:
    # 1. Convert messages to dicts
    dict_messages = [
        m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
        for m in messages
    ]
    role_prompt = role_prompt_content(preset, "panel")
    if role_prompt:
        dict_messages = [{"role": "system", "content": role_prompt}, *dict_messages]

    result = PanelResult(model=model, status="error")

    try:
        response = await BudgetedExecutor(run_id).call(
            f"panel/{model}",
            provider_id="default",
            model=model,
            messages=dict_messages,
            timeout=preset.panel.timeout,
            max_tokens=preset.panel.max_tokens,
            **(extra_kwargs or {}),
        )

        # 4. Handle response
        result.status = "ok"
        result.content = response.choices[0].message.content
        result.usage = getattr(response, "usage", None)
        result.cost_usd = getattr(response, "_omnifusion_cost_usd", 0.0)

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

    return result


async def run_panel(
    run_id: str,
    preset: Preset,
    messages: list,
    min_success: int = 1,
    extra_kwargs: dict | None = None,
) -> List[PanelResult]:
    tasks = []
    # Cap panel at max 8
    models = select_panel_models(preset, max_count=8)

    for model in models:
        tasks.append(run_panelist(run_id, model, preset, messages, extra_kwargs))

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
