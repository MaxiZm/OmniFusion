from fastapi import APIRouter, Depends, HTTPException
from .auth import verify_api_key
from ..store.runs import get_trace

router = APIRouter()


@router.get("/traces/{run_id}")
async def retrieve_trace(run_id: str, key_hash: str = Depends(verify_api_key)):
    trace = await get_trace(run_id, key_hash=key_hash)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found or not stored")

    return trace.model_dump()
