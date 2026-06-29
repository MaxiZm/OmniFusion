import os

# Configure env variables before importing anything from omnifusion
os.environ["OMNIFUSION_ADMIN_PASSWORD"] = "admin-password-security-123"
os.environ["OMNIFUSION_SECRET_KEY"] = "U1NfdlhjdmJubWwwMTIzNDU2Nzg5MGFiY2RlZmdoaWo="

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from omnifusion.admin import routes as admin_routes
from omnifusion.main import app
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.providers.validation import validate_base_url
from omnifusion.api.errors import ConfigurationError
from omnifusion.secrets.redact import redactor


@pytest.fixture(autouse=True)
def setup_settings_and_db():
    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    old_allow_egress = settings.omnifusion_allow_private_egress
    old_secure_cookie = settings.omnifusion_secure_cookie
    old_admin_password = settings.omnifusion_admin_password

    settings.db_path = "test_security.db"
    settings.omnifusion_api_keys = ["sec-key-1", "sec-key-2"]
    settings.omnifusion_allow_private_egress = False
    settings.omnifusion_secure_cookie = False
    settings.omnifusion_admin_password = SecretStr("test-password-123")
    admin_routes._admin_hash = None

    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)

    client = TestClient(app)
    yield client

    if os.path.exists(settings.db_path):
        try:
            os.remove(settings.db_path)
        except Exception:
            pass

    settings.db_path = old_db
    settings.omnifusion_api_keys = old_keys
    settings.omnifusion_allow_private_egress = old_allow_egress
    settings.omnifusion_secure_cookie = old_secure_cookie
    settings.omnifusion_admin_password = old_admin_password
    admin_routes._admin_hash = None


def test_ssrf_egress_protection():
    # 1. Block cloud metadata IP
    with pytest.raises(ConfigurationError) as exc:
        validate_base_url("http://169.254.169.254/v1", "openai")
    assert "blocked" in str(exc.value)

    # 2. Block loopback IP
    with pytest.raises(ConfigurationError) as exc:
        validate_base_url("http://127.0.0.1/v1", "openai")
    assert "blocked" in str(exc.value)

    # 3. Block private RFC-1918 range
    with pytest.raises(ConfigurationError) as exc:
        validate_base_url("http://192.168.1.50:8000/v1", "openai")
    assert "blocked" in str(exc.value)

    # 4. Allow private IP for local / self-hosted provider types
    assert (
        validate_base_url("http://127.0.0.1:11434/v1", "ollama")
        == "http://127.0.0.1:11434/v1"
    )
    assert (
        validate_base_url("http://192.168.1.100:1234/v1", "lmstudio")
        == "http://192.168.1.100:1234/v1"
    )
    # Custom (OpenAI/Anthropic-compatible) providers are admin-configured
    # self-hosted endpoints and are commonly on loopback (vLLM, llama.cpp, …).
    assert (
        validate_base_url("http://[::1]:8000/v1", "custom_openai")
        == "http://[::1]:8000/v1"
    )
    assert (
        validate_base_url("http://127.0.0.1:8000/v1", "custom_anthropic")
        == "http://127.0.0.1:8000/v1"
    )

    # 5. Allow private IP when OMNIFUSION_ALLOW_PRIVATE_EGRESS is enabled
    settings.omnifusion_allow_private_egress = True
    assert validate_base_url("http://127.0.0.1/v1", "openai") == "http://127.0.0.1/v1"
    settings.omnifusion_allow_private_egress = False


def test_secrets_logging_redaction():
    # Clear / setup test message
    test_secret = "sk-proj-supersecretkey12345abcd"
    redactor.add_secret(test_secret)

    # Test redaction method directly
    redacted = redactor.redact(f"Running LLM call with key={test_secret} in it.")
    assert test_secret not in redacted
    assert "[REDACTED]" in redacted

    # Test bearer token pattern redaction
    redacted_bearer = redactor.redact(
        "Authorization: Bearer 12345abcdefghijklmnopqrstuvwxyz"
    )
    assert "12345abcdef" not in redacted_bearer
    assert "[REDACTED]" in redacted_bearer


@pytest.mark.asyncio
async def test_admin_csrf_protection(setup_settings_and_db):
    client = setup_settings_and_db
    await init_db()

    # 1. Login to get a valid session cookie
    login_res = client.post(
        "/admin/login",
        data={"username": "admin", "password": "test-password-123"},
        follow_redirects=False,
    )
    assert login_res.status_code == 303

    # Verify secure cookie properties
    set_cookie = login_res.headers.get("set-cookie", "")
    assert "HttpOnly" in set_cookie
    assert "samesite=strict" in set_cookie.lower()

    # Get session cookies
    client.cookies.update(login_res.cookies)

    # 2. Try to post without CSRF token -> 403 Forbidden
    post_res = client.post(
        "/admin/providers/save",
        data={"id": "prov1", "type": "openai", "api_key": "some-key"},
    )
    assert post_res.status_code == 403
    assert "CSRF token missing" in post_res.text

    # 3. Fetch dashboard page (GET does not check CSRF, but provides token in HTML/context if we parsed it)
    # We can fetch dashboard or presets pages
    dash_res = client.get("/admin/providers")
    assert dash_res.status_code == 200

    # Let's test with an invalid CSRF token -> 403
    post_res_invalid = client.post(
        "/admin/providers/save",
        data={"id": "prov1", "type": "openai", "api_key": "some-key"},
        headers={"x-csrf-token": "invalid_token"},
    )
    assert post_res_invalid.status_code == 403
    assert "CSRF token invalid" in post_res_invalid.text
