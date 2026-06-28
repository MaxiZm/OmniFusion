"""Operator diagnostics: a read-only startup/readiness snapshot.

Returns only booleans, counts, and non-secret identifiers — never plaintext keys,
passwords, or stored ciphertext. Backs both the `/admin/diagnostics` page and its
JSON route. The shape is intentionally stable so external monitoring can poll it.
"""

import logging
from typing import Any, Dict, List

from ..settings import settings, validate_startup_security
from ..store.db import get_db_connection
from ..store.presets import get_preset
from ..store.providers import list_provider_metas

logger = logging.getLogger("omnifusion.admin.diagnostics")


async def _db_health() -> Dict[str, Any]:
    info: Dict[str, Any] = {"status": "ok", "journal_mode": None, "wal": False}
    try:
        async with get_db_connection() as db:
            await db.execute("SELECT 1")
            cursor = await db.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            mode = (row[0] if row else "") or ""
            info["journal_mode"] = mode
            info["wal"] = mode.lower() == "wal"
    except Exception as exc:  # pragma: no cover - exercised via unhealthy-DB test
        info["status"] = "unhealthy"
        info["error"] = str(exc)
    return info


def _web_search_config() -> Dict[str, Any]:
    provider = settings.omnifusion_web_search_provider
    configured = False
    if provider == "searxng":
        configured = bool(settings.omnifusion_searxng_base_url)
    elif provider == "tavily":
        configured = settings.omnifusion_tavily_api_key is not None
    elif provider == "brave":
        configured = settings.omnifusion_brave_api_key is not None
    return {
        "provider": provider,
        "configured": configured,
        # Non-secret: the base URL is operator infra, not a credential.
        "searxng_base_url": settings.omnifusion_searxng_base_url
        if provider == "searxng"
        else None,
        "tavily_key_present": settings.omnifusion_tavily_api_key is not None,
        "brave_key_present": settings.omnifusion_brave_api_key is not None,
    }


async def collect_diagnostics() -> Dict[str, Any]:
    """Assemble the full diagnostics snapshot. Never raises on a partial failure;
    individual probes degrade to an error field instead."""
    warnings: List[str] = []

    # 1. Startup security (placeholder secrets, Fernet validity, admin password).
    try:
        validate_startup_security()
        startup_ok = True
        startup_error = None
    except Exception as exc:
        startup_ok = False
        startup_error = str(exc)
        warnings.append(f"Startup security check failed: {exc}")

    # 2. DB / WAL health.
    db = await _db_health()
    if db["status"] != "ok":
        warnings.append("Database health check failed.")
    elif not db["wal"]:
        warnings.append(
            f"SQLite WAL mode is not active (journal_mode={db['journal_mode']!r}); "
            "concurrency may be degraded."
        )

    # 3. API keys (count only — never the values).
    api_key_count = len(settings.omnifusion_api_keys or [])
    if api_key_count == 0:
        warnings.append("No OMNIFUSION_API_KEYS configured; all API requests will 401.")

    # 4. Default fusion preset existence.
    default_preset_name = settings.omnifusion_default_fusion_preset
    default_preset_exists = False
    try:
        default_preset_exists = (await get_preset(default_preset_name)) is not None
    except Exception:
        default_preset_exists = False
    if not default_preset_exists:
        warnings.append(
            f"Default fusion preset '{default_preset_name}' does not exist; "
            f"fusion/{default_preset_name} and the openrouter/fusion alias will 404."
        )

    # 5. Providers (redacted metadata only).
    try:
        provider_metas = await list_provider_metas()
    except Exception:
        provider_metas = []
    providers_summary = [
        {
            "id": p["id"],
            "type": p["type"],
            "has_encrypted_key": p["has_encrypted_key"],
            "api_key_ref": p["api_key_ref"],
            "model_count": len(p["models"]),
        }
        for p in provider_metas
    ]
    if not providers_summary:
        warnings.append("No providers configured; fusion calls cannot reach any model.")
    else:
        unkeyed = [
            p["id"]
            for p in provider_metas
            if not p["has_encrypted_key"] and not p["api_key_ref"]
        ]
        if unkeyed:
            warnings.append(
                "Providers without a stored key or env-ref: " + ", ".join(unkeyed)
            )

    # 6. Deployment-hardening warnings.
    if not settings.omnifusion_secure_cookie:
        warnings.append(
            "OMNIFUSION_SECURE_COOKIE is disabled; admin session cookies will be sent "
            "over plain HTTP. Enable it in production behind TLS."
        )
    if settings.omnifusion_unsafe_allow_multiworker:
        warnings.append(
            "OMNIFUSION_UNSAFE_ALLOW_MULTIWORKER is set; rate limiting and the playground "
            "job registry are per-process and not globally consistent."
        )
    if settings.omnifusion_allow_private_egress:
        warnings.append(
            "OMNIFUSION_ALLOW_PRIVATE_EGRESS is set; SSRF protection on provider base "
            "URLs is relaxed."
        )
    if settings.omnifusion_trust_proxy_headers:
        warnings.append(
            "OMNIFUSION_TRUST_PROXY_HEADERS is set; only enable behind a trusted proxy."
        )

    web_search = _web_search_config()

    # Overall verdict: unhealthy if the DB or startup checks failed; warn if any
    # advisory warnings; otherwise ok.
    if db["status"] != "ok" or not startup_ok:
        status = "unhealthy"
    elif warnings:
        status = "warn"
    else:
        status = "ok"

    return {
        "status": status,
        "startup": {"ok": startup_ok, "error": startup_error},
        "database": db,
        "auth": {
            "api_key_count": api_key_count,
            "secret_key_configured": settings.omnifusion_secret_key is not None,
            "admin_password_configured": settings.omnifusion_admin_password is not None,
            "secure_cookie": settings.omnifusion_secure_cookie,
        },
        "default_preset": {
            "name": default_preset_name,
            "exists": default_preset_exists,
        },
        "providers": {
            "count": len(providers_summary),
            "entries": providers_summary,
        },
        "web_search": web_search,
        "flags": {
            "multiworker": settings.omnifusion_unsafe_allow_multiworker,
            "allow_private_egress": settings.omnifusion_allow_private_egress,
            "trust_proxy_headers": settings.omnifusion_trust_proxy_headers,
        },
        "warnings": warnings,
    }
