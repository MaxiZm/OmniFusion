from pydantic_settings import BaseSettings, SettingsConfigDict, NoDecode
from pydantic import Field, SecretStr, field_validator
from typing import List, Optional, Annotated
import json


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

    # Fix #7: Secure cookie flag (must be True in production behind HTTPS)
    omnifusion_secure_cookie: bool = False

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

    # DB
    db_path: str = "data/omnifusion.db"

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
