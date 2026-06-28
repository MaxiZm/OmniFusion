from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hashlib
import hmac
import secrets
from ..settings import settings
from .errors import ConfigurationError

security = HTTPBearer()


def _constant_time_eq(a: str, b: str) -> bool:
    """Timing-safe string compare that tolerates non-ASCII input.

    secrets.compare_digest raises TypeError on non-ASCII str args, so a client
    sending a Unicode Bearer token would otherwise trigger a 500 instead of a
    clean 401. Compare on UTF-8 bytes to stay constant-time and never raise.
    """
    try:
        return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
    except Exception:
        return False


def get_key_hash(key: str) -> str:
    secret = settings.omnifusion_secret_key
    if not secret:
        raise ConfigurationError("OMNIFUSION_SECRET_KEY is required for API key hashing")
    digest = hmac.new(
        secret.get_secret_value().encode("utf-8"),
        key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"hmac-sha256:{digest}"


def get_legacy_key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    if not settings.omnifusion_api_keys:
        # If no keys are configured, deny all
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    token = credentials.credentials
    # Constant-time compare against every configured key (byte-safe; non-ASCII
    # tokens return a clean 401 instead of raising a 500).
    matched_key = None
    for api_key in settings.omnifusion_api_keys:
        if _constant_time_eq(token, api_key):
            matched_key = api_key

    if matched_key is not None:
        return get_key_hash(matched_key)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
    )
