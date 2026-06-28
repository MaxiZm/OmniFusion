from __future__ import annotations

from omnifusion.api.errors import OmniFusionError
from omnifusion.api.schemas import FusionPlugins
from omnifusion.fusion.types import Preset
from omnifusion.store.providers import resolve_registered_provider_for_model


async def _require_registered_model(model: str, field_name: str) -> str:
    """Confirm a plugin model resolves to a registered provider and return that
    provider's id so the override is routed to the right credentials (not 'default')."""
    provider = await resolve_registered_provider_for_model(model)
    if provider is None:
        raise OmniFusionError(
            f"plugins.{field_name} model '{model}' does not resolve to a registered provider.",
            status_code=400,
            type_="invalid_request_error",
            code="plugin_model_not_registered",
        )
    return provider["id"]


def _role_models(
    panel_models: list[str],
    judge_model: str,
    final_model: str,
    provider_for,
) -> list[dict]:
    return [
        *[
            {
                "provider_id": provider_for(model, "panel"),
                "role": "panel",
                "model": model,
                "weight": 1.0,
            }
            for model in panel_models
        ],
        {
            "provider_id": provider_for(judge_model, "judge"),
            "role": "judge",
            "model": judge_model,
            "weight": 1.0,
        },
        {
            "provider_id": provider_for(final_model, "final"),
            "role": "final",
            "model": final_model,
            "weight": 1.0,
        },
    ]


async def apply_plugins_override(preset: Preset, plugins: FusionPlugins | None) -> Preset:
    if plugins is None:
        return preset

    panel_models = (
        list(plugins.analysis_models)
        if plugins.analysis_models is not None
        else list(preset.panel_models)
    )
    final_model = plugins.synthesis_model or preset.final_model

    # Validate plugin models AND remember the provider they resolved to, so the
    # override routes to those credentials rather than discarding them for 'default'.
    provider_by_model: dict[str, str] = {}
    if plugins.analysis_models is not None:
        for model in plugins.analysis_models:
            provider_by_model[model] = await _require_registered_model(model, "analysis_models")

    if plugins.synthesis_model is not None:
        provider_by_model[plugins.synthesis_model] = await _require_registered_model(
            plugins.synthesis_model, "synthesis_model"
        )

    if plugins.max_panel is not None:
        panel_models = panel_models[: plugins.max_panel]

    def provider_for(model: str, role: str) -> str:
        # Plugin-supplied models use their resolved provider; preset-supplied models
        # keep the preset's configured provider for that role.
        if model in provider_by_model:
            return provider_by_model[model]
        return preset.provider_id_for(model, role)

    data = preset.model_dump()
    data.update(
        {
            "panel_models": panel_models,
            "final_model": final_model,
            "min_panel_success": min(preset.min_panel_success, len(panel_models)),
            "models": _role_models(panel_models, preset.judge_model, final_model, provider_for),
        }
    )
    # Precedence: request `plugins.web` overrides the preset's web setting for this
    # request only (M5 plugins-mapping contract).
    if plugins.web is not None:
        data["web_enabled"] = plugins.web
    return Preset.model_validate(data)
