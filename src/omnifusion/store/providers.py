import json
import time
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from .db import get_db_connection
from ..secrets.crypto import encrypt_key, decrypt_key
from ..settings import settings

logger = logging.getLogger("omnifusion.store.providers")


def _cache_enabled() -> bool:
    """The in-process provider cache is per-worker. Under multiworker mode a sibling
    worker wouldn't see a delete/rotate for up to the TTL (stale key keeps working),
    so disable the cache there and always read fresh from the shared DB."""
    return not settings.omnifusion_unsafe_allow_multiworker

# Fix #15: In-process provider cache with TTL to eliminate N+1 DB round trips.
# Invalidated on every save/delete. TTL = 30 seconds.
#
# Security boundary: the per-id cache stores only the ENCRYPTED key (enc_key
# ciphertext) plus non-secret metadata. The plaintext key is decrypted fresh on
# every get_provider() call and never retained in the cache, so provider secrets
# do not live in process memory beyond the request that uses them.
_PROVIDER_CACHE_TTL = 30.0
_provider_cache: Optional[List[Dict[str, Any]]] = None
_provider_cache_raw: Optional[Dict[str, Dict[str, Any]]] = None  # id -> raw record (enc_key ciphertext, no plaintext)
_provider_cache_time: float = 0.0
_provider_cache_raw_time: Dict[str, float] = {}


def _invalidate_provider_cache():
    global _provider_cache, _provider_cache_time, _provider_cache_raw, _provider_cache_raw_time
    _provider_cache = None
    _provider_cache_time = 0.0
    _provider_cache_raw = {}
    _provider_cache_raw_time = {}


class ProviderModel(BaseModel):
    id: str
    type: str
    base_url: Optional[str] = None
    api_key_ref: Optional[str] = None
    models_json: str = "[]"

    # We don't expose plain_key directly here for safety,
    # but we can accept it for saving


async def get_provider(provider_id: str) -> Optional[dict]:
    """Return a provider record including its decrypted `api_key`.

    The returned plaintext key is freshly decrypted on each call and must not be
    retained by callers beyond the request. The in-process cache holds only the
    ciphertext (`enc_key`) and non-secret metadata.
    """
    global _provider_cache_raw, _provider_cache_raw_time
    now = time.monotonic()

    raw = None
    if (
        _provider_cache_raw
        and provider_id in _provider_cache_raw
        and now - _provider_cache_raw_time.get(provider_id, 0) < _PROVIDER_CACHE_TTL
    ):
        raw = _provider_cache_raw[provider_id]
    else:
        async with get_db_connection() as db:
            cursor = await db.execute(
                "SELECT id, type, base_url, enc_key, api_key_ref, models_json FROM providers WHERE id=?",
                (provider_id,),
            )
            row = await cursor.fetchone()
            if not row:
                return None

            p_id, p_type, base_url, enc_key, ref, models_json = row
            raw = {
                "id": p_id,
                "type": p_type,
                "base_url": base_url,
                "enc_key": enc_key,  # ciphertext only — safe to cache
                "api_key_ref": ref,
                "models": json.loads(models_json) if models_json else [],
            }

        if _provider_cache_raw is None:
            _provider_cache_raw = {}
        _provider_cache_raw[provider_id] = raw
        _provider_cache_raw_time[provider_id] = now

    # Decrypt fresh on every call; never store plaintext in the cache.
    plain_key = decrypt_key(raw["enc_key"]) if raw.get("enc_key") else ""
    return {
        "id": raw["id"],
        "type": raw["type"],
        "base_url": raw["base_url"],
        "api_key": plain_key,
        "api_key_ref": raw["api_key_ref"],
        "models": list(raw["models"]),
    }


async def list_providers() -> List[dict]:
    global _provider_cache, _provider_cache_time
    now = time.monotonic()

    # Return cached list if fresh
    if _provider_cache is not None and now - _provider_cache_time < _PROVIDER_CACHE_TTL:
        return _provider_cache

    providers = []
    async with get_db_connection() as db:
        cursor = await db.execute(
            "SELECT id, type, base_url, api_key_ref, models_json FROM providers"
        )
        async for row in cursor:
            p_id, p_type, base_url, ref, models_json = row
            providers.append(
                {
                    "id": p_id,
                    "type": p_type,
                    "base_url": base_url,
                    "api_key_ref": ref,
                    "models": json.loads(models_json) if models_json else [],
                }
            )

    _provider_cache = providers
    _provider_cache_time = now
    return providers


async def save_provider(
    provider_id: str,
    p_type: str,
    plain_key: str,
    base_url: Optional[str] = None,
    api_key_ref: Optional[str] = None,
    models: List[str] = [],
):
    from ..providers.validation import validate_base_url

    if base_url:
        base_url = validate_base_url(base_url, p_type)

    models_json = json.dumps(models)

    # Determine how to treat the stored encrypted key on update:
    #   - new inline key provided        -> overwrite with the new ciphertext
    #   - no inline key but a ref given  -> env-ref mode: CLEAR the stored key so a
    #                                       stale secret is not used in preference
    #                                       to the env ref (get_provider prefers
    #                                       api_key over api_key_ref)
    #   - neither provided               -> preserve the existing stored key
    if plain_key:
        enc_key = encrypt_key(plain_key)
        overwrite_enc_key = True
    elif api_key_ref:
        enc_key = None
        overwrite_enc_key = True
    else:
        enc_key = None
        overwrite_enc_key = False

    async with get_db_connection() as db:
        await db.execute(
            """
            INSERT INTO providers (id, type, base_url, enc_key, api_key_ref, models_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                type=excluded.type,
                base_url=excluded.base_url,
                enc_key=CASE WHEN ? = 1 THEN excluded.enc_key
                             ELSE COALESCE(excluded.enc_key, providers.enc_key) END,
                api_key_ref=excluded.api_key_ref,
                models_json=excluded.models_json
        """,
            (
                provider_id,
                p_type,
                base_url,
                enc_key,
                api_key_ref,
                models_json,
                1 if overwrite_enc_key else 0,
            ),
        )
        await db.commit()

    # Invalidate cache after mutation
    _invalidate_provider_cache()


async def delete_provider(provider_id: str):
    async with get_db_connection() as db:
        await db.execute("DELETE FROM providers WHERE id=?", (provider_id,))
        await db.commit()

    # Invalidate cache after mutation
    _invalidate_provider_cache()


async def resolve_provider_for_model(model_name: str) -> Optional[dict]:
    # Fix #15: Use cached list_providers for the scan, then get_provider (also cached)
    # for the full record. Eliminates N+1 pattern.
    providers = await list_providers()

    # Check if a provider has this exact model in its models list
    for p in providers:
        if model_name in p.get("models", []):
            return await get_provider(p["id"])

    # Check if the model name has a prefix that matches a provider ID (e.g. "openai/gpt-4")
    if "/" in model_name:
        parts = model_name.split("/", 1)
        prov_id = parts[0]
        prov = await get_provider(prov_id)
        if prov:
            return prov

    # Try returning the 'default' provider if configured
    default_prov = await get_provider("default")
    if default_prov:
        return default_prov

    return None
