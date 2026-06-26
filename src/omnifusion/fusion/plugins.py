from __future__ import annotations

from omnifusion.api.errors import OmniFusionError
from omnifusion.api.schemas import FusionPlugins
from omnifusion.fusion.types import Preset
from omnifusion.store.providers import resolve_registered_provider_for_model


async def _require_registered_model(model: str, field_name: str) -> None:
    provider = await resolve_registered_provider_for_model(model)
    if provider is None:
        raise OmniFusionError(
            f"plugins.{field_name} model '{model}' does not resolve to a registered provider.",
            status_code=400,
            type_="invalid_request_error",
            code="plugin_model_not_registered",
        )


def _role_models(panel_models: list[str], judge_model: str, final_model: str) -> list[dict]:
    return [
        *[
            {
                "provider_id": "default",
                "role": "panel",
                "model": model,
                "weight": 1.0,
            }
            for model in panel_models
        ],
        {
            "provider_id": "default",
            "role": "judge",
            "model": judge_model,
            "weight": 1.0,
        },
        {
            "provider_id": "default",
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

    if plugins.analysis_models is not None:
        for model in plugins.analysis_models:
            await _require_registered_model(model, "analysis_models")

    if plugins.synthesis_model is not None:
        await _require_registered_model(plugins.synthesis_model, "synthesis_model")

    if plugins.max_panel is not None:
        panel_models = panel_models[: plugins.max_panel]

    data = preset.model_dump()
    data.update(
        {
            "panel_models": panel_models,
            "final_model": final_model,
            "min_panel_success": min(preset.min_panel_success, len(panel_models)),
            "models": _role_models(panel_models, preset.judge_model, final_model),
        }
    )
    # Precedence: request `plugins.web` overrides the preset's web setting for this
    # request only (M5 plugins-mapping contract).
    if plugins.web is not None:
        data["web_enabled"] = plugins.web
    return Preset.model_validate(data)
