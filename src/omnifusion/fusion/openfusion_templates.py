from __future__ import annotations

from typing import Any

from omnifusion.store.providers import resolve_registered_provider_for_model


OPENFUSION_SOURCE_COMMIT = "058035c193719a5752bc579fa2505378b5f40b2d"

_TEMPLATES: dict[str, dict[str, Any]] = {
    "quality": {
        "panel_models": [
            "anthropic/claude-sonnet-4",
            "google/gemini-3-pro",
            "deepseek/deepseek-v4-pro",
        ],
        "judge_model": "anthropic/claude-sonnet-4",
        "final_model": "anthropic/claude-sonnet-4",
    },
    "budget": {
        "panel_models": [
            "openai/gpt-4o-mini",
            "deepseek/deepseek-v4-pro",
            "moonshotai/kimi-k2.6",
        ],
        "judge_model": "deepseek/deepseek-v4-pro",
        "final_model": "openai/gpt-4o-mini",
    },
}


def template_spec(name: str) -> dict[str, Any] | None:
    spec = _TEMPLATES.get(name)
    if spec is None:
        return None
    return {
        "panel_models": list(spec["panel_models"]),
        "judge_model": spec["judge_model"],
        "final_model": spec["final_model"],
    }


async def template_availability(name: str) -> dict[str, Any] | None:
    spec = template_spec(name)
    if spec is None:
        return None
    required = [*spec["panel_models"], spec["judge_model"], spec["final_model"]]
    missing = []
    for model in dict.fromkeys(required):
        if await resolve_registered_provider_for_model(model) is None:
            missing.append(model)
    return {
        "name": name,
        "available": not missing,
        "required_models": list(dict.fromkeys(required)),
        "missing_models": missing,
        **spec,
    }


async def all_template_availability() -> list[dict[str, Any]]:
    out = []
    for name in ("quality", "budget"):
        availability = await template_availability(name)
        if availability is not None:
            out.append(availability)
    return out
