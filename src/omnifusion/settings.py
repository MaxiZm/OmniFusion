from pydantic_settings import BaseSettings, SettingsConfigDict, NoDecode
from pydantic import Field, SecretStr, field_validator
from typing import List, Optional, Annotated
import json


_PLACEHOLDER_SECRET_KEYS = {
    "",
    "your_fernet_secret_key",
    "your-fernet-secret-key",
    "changeme",
}
_PLACEHOLDER_ADMIN_PASSWORDS = {
    "",
    "your_admin_password",
    "your-admin-password",
    "changeme",
}


class Settings(BaseSettings):
    # Security
    omnifusion_secret_key: Optional[SecretStr] = None
    omnifusion_admin_password: Optional[SecretStr] = None
    # NoDecode: stop pydantic-settings from JSON-decoding these list fields at the
    # env/.env source level. Without it, a comma-separated value from an env var or
    # .env file (e.g. OMNIFUSION_API_KEYS=a,b — the documented format) is run through
    # json.loads BEFORE our validator and crashes startup. NoDecode hands the raw
    # string to parse_list_field below, which accepts both comma and JSON forms.
    omnifusion_api_keys: Annotated[List[str], NoDecode] = Field(default_factory=list)
    omnifusion_unsafe_allow_multiworker: bool = False
    omnifusion_allow_private_egress: bool = False
    omnifusion_passthrough_whitelist: Annotated[List[str], NoDecode] = Field(
        default_factory=list
    )

    # Secure by default; local HTTP development can opt out with
    # OMNIFUSION_SECURE_COOKIE=0.
    omnifusion_secure_cookie: bool = True

    # Fix #8: Login brute-force protection settings
    omnifusion_max_login_attempts: int = 5
    omnifusion_login_lockout_seconds: int = 900  # 15 minutes
    # Trust X-Forwarded-For for the client IP (login lockout + logging). Enable ONLY
    # when behind a trusted reverse proxy; otherwise the lockout buckets every client
    # under the proxy IP and one attacker locks out everyone.
    omnifusion_trust_proxy_headers: bool = False

    # Conservative fallback price (USD per 1M tokens) for models whose true price is
    # unknown to litellm (e.g. self-hosted/custom). Budget fails CLOSED: unknown models
    # reserve at this rate instead of the cheapest tier, so the ceiling can't be blown.
    omnifusion_unknown_model_input_per_mtok: float = 10.0
    omnifusion_unknown_model_output_per_mtok: float = 30.0

    # Fix #11: Per-key inbound concurrency cap
    omnifusion_max_concurrent_per_key: int = 5

    # Fix #12: Input size limits
    omnifusion_max_content_chars: int = 100_000
    omnifusion_max_messages: int = 200
    omnifusion_max_tokens_limit: int = 16_384
    omnifusion_max_request_body_bytes: int = 1_000_000
    omnifusion_max_stage_timeout: int = 300

    # Budgets
    global_daily_budget_usd: float = 100.0
    request_budget_usd: float = 10.0

    # Defaults (can be overridden by yaml)
    # Fix #13: wall_timeout is now wired (was dead config).
    # Env var: OMNIFUSION_WALL_TIMEOUT (matches the omnifusion_ prefix convention).
    omnifusion_wall_timeout: int = 90
    panel_timeout: int = 30
    max_panel: int = 8
    min_panel_success: int = 1
    omnifusion_default_fusion_preset: str = "general"
    omnifusion_compat_placeholder_model: str = "compat-placeholder-model"
    omnifusion_web_search_provider: str = "searxng"
    omnifusion_searxng_base_url: str = "http://localhost:8080"
    omnifusion_tavily_api_key: Optional[SecretStr] = None
    omnifusion_brave_api_key: Optional[SecretStr] = None
    omnifusion_web_fetch_cache_ttl_seconds: float = 300.0
    omnifusion_web_fetch_per_domain_interval_seconds: float = 1.0
    omnifusion_conductor_max_repairs: int = 1

    # DB
    db_path: str = "data/omnifusion.db"

    # Logging
    omnifusion_log_level: str = "INFO"
    omnifusion_log_format: str = "plain"

    # Provider circuit breaker
    omnifusion_circuit_breaker_failure_threshold: int = 5
    omnifusion_circuit_breaker_cooldown_seconds: float = 30.0

    # extra="ignore": a self-hosted deployment's environment will contain unrelated
    # vars; never refuse to boot because of one. Unknown keys in .env are ignored.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @field_validator("omnifusion_api_keys", "omnifusion_passthrough_whitelist", mode="before")
    @classmethod
    def parse_list_field(cls, v):
        """
        Fix #19: Accept both JSON array format (["key1","key2"]) and
        comma-separated format (key1,key2) for list fields.
        This prevents startup crashes when users copy .env.example verbatim.
        """
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            # Try JSON first
            if v.startswith("["):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            # Fall back to comma-separated
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


settings = Settings()


def validate_startup_security(s: Settings = settings) -> None:
    """Reject placeholder secrets before the app starts serving traffic."""
    if not s.omnifusion_secret_key:
        raise ValueError(
            "OMNIFUSION_SECRET_KEY is not set. Generate one with: "
            "uv run python -m src.omnifusion.cli genkey"
        )
    secret_key = s.omnifusion_secret_key.get_secret_value().strip()
    if secret_key.lower() in _PLACEHOLDER_SECRET_KEYS:
        raise ValueError(
            "OMNIFUSION_SECRET_KEY is still a placeholder value. Generate a real key."
        )

    try:
        from cryptography.fernet import Fernet

        Fernet(secret_key.encode("utf-8"))
    except Exception as exc:
        raise ValueError(f"OMNIFUSION_SECRET_KEY is not a valid Fernet key: {exc}") from exc

    if not s.omnifusion_admin_password:
        raise ValueError("OMNIFUSION_ADMIN_PASSWORD is not set.")
    admin_password = s.omnifusion_admin_password.get_secret_value().strip()
    if admin_password.lower() in _PLACEHOLDER_ADMIN_PASSWORDS:
        raise ValueError(
            "OMNIFUSION_ADMIN_PASSWORD is still a placeholder value. Set a real password."
        )
