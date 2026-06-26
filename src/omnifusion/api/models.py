from fastapi import APIRouter, Depends
from .auth import verify_api_key
from .model_names import model_alias_entries
from ..store.presets import ensure_compat_placeholder_presets, list_presets
from ..settings import settings
import time

router = APIRouter()


@router.get("/models")
async def list_models(key_hash: str = Depends(verify_api_key)):
    await ensure_compat_placeholder_presets()
    presets = await list_presets()
    data = []
    now = int(time.time())

    # 1. Add all presets
    for p in presets:
        entry = {
            "id": f"fusion/{p.name}",
            "object": "model",
            "created": now,
            "owned_by": "omnifusion",
        }
        if p.compat_status:
            entry["status"] = p.compat_status
        data.append(entry)

    data.extend(model_alias_entries(now))

    # 2. Add all whitelisted passthrough models
    for m in settings.omnifusion_passthrough_whitelist:
        data.append(
            {
                "id": m,
                "object": "model",
                "created": now,
                "owned_by": "omnifusion",
            }
        )

    return {"object": "list", "data": data}
