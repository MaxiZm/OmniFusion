"""Public provider-management API.

Mounted under both `/v1` and `/api/v1`. Reads return redacted records only —
identifiers, metadata, and a `has_encrypted_key` boolean, never plaintext or the
stored ciphertext. Writes accept a write-only `api_key`; omitting it preserves the
existing stored key, while supplying `api_key_ref` switches the provider to
env-ref mode (the stored key is cleared). This mirrors the admin console's save
semantics so the two surfaces never diverge.
"""

import time
from typing import List, Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .auth import verify_api_key
from .errors import OmniFusionError
from ..store.providers import (
    delete_provider,
    get_provider,
    get_provider_meta,
    list_provider_metas,
    save_provider,
)

router = APIRouter()


class ProviderUpsert(BaseModel):
    """Write body for PUT /providers/{id}. `api_key` is write-only and never echoed."""

    type: str
    base_url: Optional[str] = None
    api_key: Optional[str] = Field(default=None, description="write-only; never returned")
    api_key_ref: Optional[str] = None
    models: List[str] = Field(default_factory=list)


@router.get("/providers")
async def list_providers_route(key_hash: str = Depends(verify_api_key)):
    return {"object": "list", "data": await list_provider_metas()}


@router.get("/providers/{provider_id}")
async def get_provider_route(
    provider_id: str, key_hash: str = Depends(verify_api_key)
):
    meta = await get_provider_meta(provider_id)
    if not meta:
        raise OmniFusionError(
            f"Provider {provider_id} not found",
            status_code=404,
            code="provider_not_found",
        )
    return meta


@router.put("/providers/{provider_id}")
async def upsert_provider_route(
    provider_id: str,
    body: ProviderUpsert,
    key_hash: str = Depends(verify_api_key),
):
    if not body.type or not body.type.strip():
        raise OmniFusionError(
            "Provider type must not be empty",
            status_code=400,
            code="invalid_provider",
        )
    try:
        await save_provider(
            provider_id=provider_id,
            p_type=body.type,
            plain_key=body.api_key or "",
            base_url=body.base_url or None,
            api_key_ref=body.api_key_ref or None,
            models=body.models,
        )
    except OmniFusionError:
        raise
    except Exception as exc:
        # SSRF/base_url validation and friends surface here; report as a 400 rather
        # than leaking a 500. The message carries no secret material.
        raise OmniFusionError(
            f"Failed to save provider: {exc}",
            status_code=400,
            code="invalid_provider",
        )
    return await get_provider_meta(provider_id)


@router.delete("/providers/{provider_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_provider_route(
    provider_id: str, key_hash: str = Depends(verify_api_key)
):
    await delete_provider(provider_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/providers/{provider_id}/test")
async def test_provider_route(
    provider_id: str, key_hash: str = Depends(verify_api_key)
):
    """Issue a single bounded ping completion against the provider's first model.

    Returns a JSON verdict. Any error string is redacted before it leaves the
    process so a decrypted key embedded in a provider error can never leak.
    """
    from ..llm.client import llm_client
    from ..secrets.redact import redactor

    provider = await get_provider(provider_id)
    if not provider:
        raise OmniFusionError(
            f"Provider {provider_id} not found",
            status_code=404,
            code="provider_not_found",
        )

    models = provider.get("models", [])
    if not models:
        return {"provider_id": provider_id, "status": "no_models"}

    model = models[0]
    start = time.time()
    try:
        await llm_client.acompletion(
            provider_id=provider_id,
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            timeout=5,
            max_tokens=1,
        )
        return {
            "provider_id": provider_id,
            "status": "success",
            "model": model,
            "latency_ms": int((time.time() - start) * 1000),
        }
    except Exception as exc:
        return {
            "provider_id": provider_id,
            "status": "failed",
            "model": model,
            "error": redactor.redact(str(exc)),
        }
