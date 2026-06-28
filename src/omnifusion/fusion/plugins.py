from __future__ import annotations

from omnifusion.api.errors import OmniFusionError
from omnifusion.api.schemas import FusionPlugins, OpenFusionOverrides
from omnifusion.fusion.openfusion_templates import template_availability, template_spec
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


async def _provider_map_for_models(models: list[str], field_name: str) -> dict[str, str]:
    providers: dict[str, str] = {}
    for model in models:
        providers[model] = await _require_registered_model(model, field_name)
    return providers


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
    judge_model = plugins.judge_model or preset.judge_model
    final_model = plugins.synthesis_model or preset.final_model

    # Validate plugin models AND remember the provider they resolved to, so the
    # override routes to those credentials rather than discarding them for 'default'.
    provider_by_model: dict[str, str] = {}
    if plugins.analysis_models is not None:
        provider_by_model.update(
            await _provider_map_for_models(plugins.analysis_models, "analysis_models")
        )

    if plugins.judge_model is not None:
        provider_by_model[plugins.judge_model] = await _require_registered_model(
            plugins.judge_model, "judge_model"
        )

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
            "judge_model": judge_model,
            "final_model": final_model,
            "min_panel_success": min(preset.min_panel_success, len(panel_models)),
            "models": _role_models(panel_models, judge_model, final_model, provider_for),
        }
    )
    # Precedence: request `plugins.web` overrides the preset's web setting for this
    # request only (M5 plugins-mapping contract).
    if plugins.web is not None:
        data["web_enabled"] = plugins.web
    if plugins.fusion_mode is not None:
        data["fusion_mode"] = plugins.fusion_mode
    if plugins.aggregator is not None:
        data["aggregator"] = plugins.aggregator
    if plugins.analysis_emit is not None:
        analysis_emit = preset.analysis_emit.model_dump()
        analysis_emit["enabled"] = plugins.analysis_emit
        data["analysis_emit"] = analysis_emit
    if plugins.routing is not None:
        router = preset.router.model_dump()
        router["enabled"] = plugins.routing
        data["router"] = router
    return Preset.model_validate(data)


async def apply_openfusion_override(
    preset: Preset, overrides: OpenFusionOverrides | None
) -> Preset:
    if overrides is None:
        return preset

    data = preset.model_dump()
    provider_by_model: dict[str, str] = {}

    if overrides.preset:
        availability = await template_availability(overrides.preset)
        if availability is None:
            from omnifusion.api.errors import OmniFusionError

            raise OmniFusionError(
                f"Unknown OpenFusion preset template '{overrides.preset}'.",
                status_code=400,
                code="openfusion_template_unknown",
            )
        if not availability["available"]:
            from omnifusion.api.errors import OmniFusionError

            missing = ", ".join(availability["missing_models"])
            raise OmniFusionError(
                f"OpenFusion preset template '{overrides.preset}' is unavailable; "
                f"register providers for: {missing}.",
                status_code=400,
                code="openfusion_template_unavailable",
            )
        template = template_spec(overrides.preset)
        if template is not None:
            data.update(template)
            for model in [
                *template["panel_models"],
                template["judge_model"],
                template["final_model"],
            ]:
                provider_by_model[model] = await _require_registered_model(
                    model, f"openfusion.preset.{overrides.preset}"
                )

    if overrides.panel_models is not None:
        panel_models = list(overrides.panel_models)
        data["panel_models"] = panel_models
        provider_by_model.update(
            await _provider_map_for_models(panel_models, "openfusion.panel_models")
        )
    if overrides.max_panel is not None:
        data["panel_models"] = list(data["panel_models"])[: overrides.max_panel]
    if overrides.judge_model is not None:
        data["judge_model"] = overrides.judge_model
        provider_by_model[overrides.judge_model] = await _require_registered_model(
            overrides.judge_model, "openfusion.judge_model"
        )
    if overrides.final_model is not None:
        data["final_model"] = overrides.final_model
        provider_by_model[overrides.final_model] = await _require_registered_model(
            overrides.final_model, "openfusion.final_model"
        )
    if overrides.fusion_mode is not None:
        data["fusion_mode"] = overrides.fusion_mode
    if overrides.aggregator is not None:
        data["aggregator"] = overrides.aggregator
    if overrides.self_fusion is not None:
        current = preset.self_fusion.model_dump()
        current.update(overrides.self_fusion.model_dump(exclude_none=True))
        data["self_fusion"] = current
    if overrides.debate is not None:
        current = preset.debate.model_dump()
        current.update(overrides.debate.model_dump(exclude_none=True))
        data["debate"] = current
    if overrides.router is not None:
        current = preset.router.model_dump()
        router_data = overrides.router.model_dump(exclude_none=True)
        route_models = router_data.get("route_models")
        if route_models:
            for route_model in route_models:
                provider_by_model[route_model["model"]] = await _require_registered_model(
                    route_model["model"], "openfusion.router.route_models"
                )
        if router_data.get("classifier_model"):
            provider_by_model[router_data["classifier_model"]] = await _require_registered_model(
                router_data["classifier_model"], "openfusion.router.classifier_model"
            )
        current.update(router_data)
        data["router"] = current
    if overrides.analysis_emit is not None:
        current = preset.analysis_emit.model_dump()
        current["enabled"] = overrides.analysis_emit
        data["analysis_emit"] = current
    if overrides.response_cache is not None:
        current = preset.response_cache.model_dump()
        current.update(overrides.response_cache.model_dump(exclude_none=True))
        data["response_cache"] = current
    if overrides.web is not None:
        data["web_enabled"] = overrides.web

    panel_models = list(data["panel_models"])
    judge_model = data["judge_model"]
    final_model = data["final_model"]
    data["min_panel_success"] = min(int(data.get("min_panel_success") or 1), len(panel_models))

    def provider_for(model: str, role: str) -> str:
        if model in provider_by_model:
            return provider_by_model[model]
        return preset.provider_id_for(model, role)

    data["models"] = _role_models(panel_models, judge_model, final_model, provider_for)
    return Preset.model_validate(data)
