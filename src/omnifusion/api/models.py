from fastapi import APIRouter, Depends
from .auth import verify_api_key
from ..store.presets import list_presets
from ..settings import settings
import time

router = APIRouter()


@router.get("/models")
async def list_models(key_hash: str = Depends(verify_api_key)):
    presets = await list_presets()
    data = []

    # 1. Add all presets
    for p in presets:
        data.append(
            {
                "id": f"fusion/{p.name}",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "omnifusion",
            }
        )

    # 2. Add all whitelisted passthrough models
    for m in settings.omnifusion_passthrough_whitelist:
        data.append(
            {
                "id": m,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "omnifusion",
            }
        )

    return {"object": "list", "data": data}
