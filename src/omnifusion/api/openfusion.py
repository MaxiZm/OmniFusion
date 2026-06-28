from __future__ import annotations

from fastapi import APIRouter, Depends

from omnifusion.api.auth import verify_api_key
from omnifusion.api.errors import OmniFusionError
from omnifusion.api.model_names import normalize_requested_model
from omnifusion.api.schemas import ChatCompletionRequest
from omnifusion.fusion.openfusion_runtime import (
    estimate_openfusion_request,
    mixed_server_and_function_tools,
    server_web_tools_requested,
)
from omnifusion.fusion.openfusion_templates import (
    OPENFUSION_SOURCE_COMMIT,
    all_template_availability,
)
from omnifusion.fusion.plugins import apply_openfusion_override, apply_plugins_override
from omnifusion.settings import settings
from omnifusion.store.presets import get_preset, list_presets
from omnifusion.store.providers import list_provider_metas

router = APIRouter()


@router.get("/config")
async def openfusion_config(key_hash: str = Depends(verify_api_key)):
    presets = await list_presets()
    return {
        "object": "openfusion.config",
        "source": {
            "project": "shahar-dagan/openfusion",
            "commit": OPENFUSION_SOURCE_COMMIT,
            "license": "MIT",
        },
        "defaults": {
            "model_alias": "openfusion",
            "alias_of": f"fusion/{settings.omnifusion_default_fusion_preset}",
            "fusion_mode": "panel",
            "aggregator": "judge",
            "router_enabled": False,
            "response_cache_enabled": False,
            "analysis_emit_enabled": False,
        },
        "capabilities": {
            "fusion_modes": ["panel", "self_fusion", "debate"],
            "aggregators": ["judge", "vote", "ranked"],
            "router_modes": ["heuristic", "model", "always", "never"],
            "openrouter_server_tools": ["openrouter:web_search", "openrouter:web_fetch"],
            "secrets_policy": "provider keys remain server-side in encrypted OmniFusion provider storage",
        },
        "templates": await all_template_availability(),
        "presets": [
            {
                "name": preset.name,
                "display_name": preset.display_name,
                "model": f"fusion/{preset.name}",
                "fusion_mode": preset.fusion_mode,
                "aggregator": preset.aggregator,
                "router_enabled": preset.router.enabled,
                "web_enabled": preset.web_enabled,
            }
            for preset in presets
        ],
        "providers": await list_provider_metas(),
    }


@router.post("/estimate")
async def estimate(body: ChatCompletionRequest, key_hash: str = Depends(verify_api_key)):
    normalized_model = normalize_requested_model(body.model)
    if normalized_model != body.model:
        body = body.model_copy(update={"model": normalized_model})
    if not body.model.startswith("fusion/"):
        raise OmniFusionError(
            "Estimate is only available for fusion/openfusion models.",
            status_code=400,
            type_="invalid_request_error",
            code="estimate_requires_fusion_model",
        )
    preset_name = body.model[len("fusion/") :]
    preset = await get_preset(preset_name)
    if not preset:
        raise OmniFusionError(f"Preset {preset_name} not found", status_code=404)
    preset = await apply_plugins_override(preset, body.plugins)
    preset = await apply_openfusion_override(preset, body.openfusion)
    if mixed_server_and_function_tools(body.tools):
        raise OmniFusionError(
            "OpenRouter server web tools cannot be combined with client-side function tools.",
            status_code=400,
            type_="invalid_request_error",
            code="mixed_tool_modes",
        )
    if server_web_tools_requested(body.tools):
        preset = preset.model_copy(update={"web_enabled": True})
        body = body.model_copy(update={"tools": None})
    return estimate_openfusion_request(preset, body)
