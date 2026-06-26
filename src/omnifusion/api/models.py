from fastapi import APIRouter, Depends
from .auth import verify_api_key
from .model_names import model_alias_entries
from .errors import OmniFusionError
from ..store.presets import ensure_compat_placeholder_presets, list_presets
from ..settings import settings
import time

router = APIRouter()


def preset_model_entry(preset, created: int) -> dict:
    entry = {
        "id": f"fusion/{preset.name}",
        "object": "model",
        "created": created,
        "owned_by": "omnifusion",
    }
    if preset.compat_status:
        entry["status"] = preset.compat_status
    return entry


async def all_model_entries(created: int) -> list[dict]:
    await ensure_compat_placeholder_presets()
    presets = await list_presets()
    data = [preset_model_entry(preset, created) for preset in presets]
    data.extend(model_alias_entries(created))
    for model in settings.omnifusion_passthrough_whitelist:
        data.append(
            {
                "id": model,
                "object": "model",
                "created": created,
                "owned_by": "omnifusion",
            }
        )
    return data


@router.get("/models")
async def list_models(key_hash: str = Depends(verify_api_key)):
    now = int(time.time())
    return {"object": "list", "data": await all_model_entries(now)}


@router.get("/models/{model_id:path}")
async def retrieve_model(model_id: str, key_hash: str = Depends(verify_api_key)):
    now = int(time.time())
    for entry in await all_model_entries(now):
        if entry["id"] == model_id:
            return entry
    raise OmniFusionError(
        f"Model {model_id} not found",
        status_code=404,
        code="model_not_found",
    )
