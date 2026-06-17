import litellm
import logging
from typing import Dict, Any, Tuple, List
from ..settings import settings

logger = logging.getLogger("omnifusion.pricing")

# Operator-supplied price overrides: model_name -> (input_per_token, output_per_token).
# Populate via register_price_override() (e.g. from provider config) to price
# self-hosted/custom models accurately.
PRICE_OVERRIDES: Dict[str, Tuple[float, float]] = {}

# Models we've already warned about pricing for (avoid log spam).
_warned_unknown: set = set()


def register_price_override(model: str, input_per_mtok: float, output_per_mtok: float) -> None:
    PRICE_OVERRIDES[model] = (input_per_mtok / 1_000_000, output_per_mtok / 1_000_000)


def usd_to_micro(usd: float) -> int:
    """Convert USD to integer micro-USD, rounding half-up (never truncate toward zero,
    which would systematically under-record spend) and clamping negatives to 0."""
    return int(round(max(0.0, usd) * 1_000_000))


def get_model_cost_estimate(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost in USD for a given model and token counts.
    """
    if model in PRICE_OVERRIDES:
        incost, outcost = PRICE_OVERRIDES[model]
        return (incost * input_tokens) + (outcost * output_tokens)

    try:
        # Use litellm cost calculator if possible
        cost = litellm.cost_calculator.cost_per_token(
            model, input_tokens, output_tokens
        )
        if cost is not None:
            return cost[0] + cost[1]
    except Exception:
        pass

    # Unknown model (common for self-hosted/custom providers). Fail CLOSED for budget
    # safety: price at a conservative high rate instead of the cheapest tier, so an
    # expensive unpriced model can't silently blow past the configured ceiling (~100x
    # under-charge if we assumed gpt-4o-mini rates). Operators can register the true
    # price via register_price_override().
    if model not in _warned_unknown:
        _warned_unknown.add(model)
        logger.warning(
            f"No price known for model '{model}'; budgeting at conservative fallback "
            f"(${settings.omnifusion_unknown_model_input_per_mtok}/M in, "
            f"${settings.omnifusion_unknown_model_output_per_mtok}/M out). "
            f"Register an override for accurate accounting."
        )
    incost = settings.omnifusion_unknown_model_input_per_mtok / 1_000_000
    outcost = settings.omnifusion_unknown_model_output_per_mtok / 1_000_000
    return (incost * input_tokens) + (outcost * output_tokens)


def calculate_actual_cost(response: Any, model: str) -> float:
    """
    Calculate actual cost from litellm response.
    """
    if model in PRICE_OVERRIDES:
        usage = getattr(response, "usage", None)
        if usage:
            return get_model_cost_estimate(
                model, usage.prompt_tokens, usage.completion_tokens
            )
        return 0.0

    try:
        cost = litellm.completion_cost(completion_response=response)
        return cost if cost else 0.0
    except Exception:
        # Fallback to estimate if completion_cost raises error
        usage = getattr(response, "usage", None)
        if usage:
            return get_model_cost_estimate(
                model, usage.prompt_tokens, usage.completion_tokens
            )
        return 0.0


def estimate_tokens(model: str, messages: List[dict]) -> int:
    try:
        return litellm.token_counter(model=model, messages=messages)
    except Exception:
        chars = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                chars += len(content)
        return max(1, chars // 4)


def estimate_call_cost(
    model: str, messages: List[dict], max_output_tokens: int
) -> float:
    input_tokens = estimate_tokens(model, messages)
    return get_model_cost_estimate(model, input_tokens, max_output_tokens)
