"""Public provider-management API: CRUD, redaction, test route, /v1 + /api/v1 parity."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from omnifusion.settings import settings
from omnifusion.store.db import init_db


SECRET_KEY_VALUE = "sk-provider-secret-abcdef1234567890"


class _MockMessage:
    content = "pong"


class _MockChoice:
    message = _MockMessage()


class _MockResponse:
    choices = [_MockChoice()]
    usage = None


@pytest.fixture
def api_client(tmp_path):
    # No `with` (no lifespan): tests call init_db() themselves in the pytest event
    # loop so the WAL init lock binds there, not in the TestClient portal loop.
    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "providers-api.db")
    settings.omnifusion_api_keys = ["prov-key"]
    try:
        from omnifusion.main import app

        yield TestClient(app)
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys


def _auth():
    return {"Authorization": "Bearer prov-key"}


@pytest.mark.asyncio
async def test_provider_crud_and_redaction(api_client, tmp_path):
    await init_db()
    client = api_client

    # Create via PUT with an inline (write-only) api_key.
    create = client.put(
        "/v1/providers/openai",
        headers=_auth(),
        json={
            "type": "openai",
            "api_key": SECRET_KEY_VALUE,
            "models": ["gpt-4o-mini"],
        },
    )
    assert create.status_code == 200
    body = create.json()
    # Redacted shape only — never the key or stored ciphertext.
    assert body["id"] == "openai"
    assert body["type"] == "openai"
    assert body["has_encrypted_key"] is True
    assert body["models"] == ["gpt-4o-mini"]
    assert "api_key" not in body
    assert "enc_key" not in body
    assert SECRET_KEY_VALUE not in create.text

    # Read back, redacted, on the other mount.
    got = client.get("/api/v1/providers/openai", headers=_auth())
    assert got.status_code == 200
    assert got.json()["has_encrypted_key"] is True
    assert SECRET_KEY_VALUE not in got.text

    # Update WITHOUT api_key preserves the stored key.
    upd = client.put(
        "/v1/providers/openai",
        headers=_auth(),
        json={"type": "openai", "models": ["gpt-4o-mini", "gpt-4o"]},
    )
    assert upd.status_code == 200
    assert upd.json()["has_encrypted_key"] is True
    assert upd.json()["models"] == ["gpt-4o-mini", "gpt-4o"]

    # List contains it (redacted).
    listing = client.get("/v1/providers", headers=_auth())
    assert listing.status_code == 200
    ids = [p["id"] for p in listing.json()["data"]]
    assert "openai" in ids
    assert SECRET_KEY_VALUE not in listing.text

    # Delete → 204, then 404.
    assert client.delete("/v1/providers/openai", headers=_auth()).status_code == 204
    assert client.get("/v1/providers/openai", headers=_auth()).status_code == 404


@pytest.mark.asyncio
async def test_api_key_ref_switches_to_env_ref_mode(api_client):
    await init_db()
    client = api_client

    # Start with a stored key.
    client.put(
        "/v1/providers/p1",
        headers=_auth(),
        json={"type": "openai", "api_key": SECRET_KEY_VALUE, "models": ["m"]},
    )
    # Switch to env-ref mode: stored key cleared, ref recorded.
    upd = client.put(
        "/v1/providers/p1",
        headers=_auth(),
        json={"type": "openai", "api_key_ref": "OPENAI_API_KEY", "models": ["m"]},
    )
    assert upd.status_code == 200
    body = upd.json()
    assert body["has_encrypted_key"] is False
    assert body["api_key_ref"] == "OPENAI_API_KEY"


@pytest.mark.asyncio
async def test_v1_and_api_v1_parity(api_client):
    await init_db()
    client = api_client
    client.put(
        "/v1/providers/dup",
        headers=_auth(),
        json={"type": "openai", "api_key": SECRET_KEY_VALUE, "models": ["m"]},
    )
    a = client.get("/v1/providers/dup", headers=_auth()).json()
    b = client.get("/api/v1/providers/dup", headers=_auth()).json()
    assert a == b


@pytest.mark.asyncio
async def test_provider_requires_auth(api_client):
    await init_db()
    client = api_client
    # Both missing and wrong bearer are rejected (no anonymous provider access).
    assert client.get("/v1/providers").status_code in (401, 403)
    assert client.get(
        "/v1/providers", headers={"Authorization": "Bearer nope"}
    ).status_code == 401


@pytest.mark.asyncio
async def test_provider_test_route_success_and_failure(api_client):
    await init_db()
    client = api_client
    client.put(
        "/v1/providers/openai",
        headers=_auth(),
        json={"type": "openai", "api_key": SECRET_KEY_VALUE, "models": ["gpt-4o-mini"]},
    )

    with patch(
        "omnifusion.llm.client.llm_client.acompletion", return_value=_MockResponse()
    ):
        ok = client.post("/v1/providers/openai/test", headers=_auth())
    assert ok.status_code == 200
    assert ok.json()["status"] == "success"
    assert "latency_ms" in ok.json()

    # A provider error that embeds a secret must be redacted in the response.
    async def _boom(*args, **kwargs):
        raise RuntimeError(f"upstream rejected key {SECRET_KEY_VALUE}")

    with patch("omnifusion.llm.client.llm_client.acompletion", side_effect=_boom):
        fail = client.post("/v1/providers/openai/test", headers=_auth())
    assert fail.status_code == 200
    assert fail.json()["status"] == "failed"
    assert SECRET_KEY_VALUE not in fail.text
    assert "[REDACTED]" in fail.json()["error"]


@pytest.mark.asyncio
async def test_provider_test_route_edge_cases(api_client):
    await init_db()
    client = api_client
    # Missing provider → 404.
    assert (
        client.post("/v1/providers/ghost/test", headers=_auth()).status_code == 404
    )
    # Provider with no models → no_models verdict.
    client.put(
        "/v1/providers/empty",
        headers=_auth(),
        json={"type": "openai", "api_key": SECRET_KEY_VALUE, "models": []},
    )
    res = client.post("/v1/providers/empty/test", headers=_auth())
    assert res.status_code == 200
    assert res.json()["status"] == "no_models"
