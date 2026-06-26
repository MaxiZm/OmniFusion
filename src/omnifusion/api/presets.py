from fastapi import APIRouter, Depends, status
from fastapi.responses import Response

from .auth import verify_api_key
from .errors import OmniFusionError
from ..fusion.types import Preset
from ..store.presets import delete_preset, get_preset, list_presets, save_preset

router = APIRouter()


@router.get("/presets")
async def list_preset_specs(key_hash: str = Depends(verify_api_key)):
    presets = await list_presets()
    return {"object": "list", "data": [preset.model_dump() for preset in presets]}


@router.get("/presets/{name}")
async def retrieve_preset_spec(name: str, key_hash: str = Depends(verify_api_key)):
    preset = await get_preset(name)
    if not preset:
        raise OmniFusionError(
            f"Preset {name} not found",
            status_code=404,
            code="preset_not_found",
        )
    return preset.model_dump()


@router.put("/presets/{name}")
async def upsert_preset_spec(
    name: str,
    preset: Preset,
    key_hash: str = Depends(verify_api_key),
):
    if preset.name != name:
        preset = preset.model_copy(update={"name": name, "display_name": preset.display_name or name})
    await save_preset(preset)
    return (await get_preset(name)).model_dump()


@router.delete("/presets/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_preset_spec(name: str, key_hash: str = Depends(verify_api_key)):
    await delete_preset(name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
