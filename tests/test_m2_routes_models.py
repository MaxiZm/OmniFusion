import pytest
from fastapi.testclient import TestClient

from omnifusion.settings import settings
from omnifusion.store.db import init_db


@pytest.fixture
def client(tmp_path):
    from omnifusion.main import app

    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "m2_routes.db")
    settings.omnifusion_api_keys = ["route-key"]

    with TestClient(app) as test_client:
        yield test_client

    settings.db_path = old_db
    settings.omnifusion_api_keys = old_keys


def test_missing_bearer_auth_returns_401(client):
    response = client.get("/v1/models")
    assert response.status_code == 401


def test_api_v1_models_mirrors_v1(client):
    headers = {"Authorization": "Bearer route-key"}

    v1_response = client.get("/v1/models", headers=headers)
    api_v1_response = client.get("/api/v1/models", headers=headers)

    assert v1_response.status_code == 200
    assert api_v1_response.status_code == 200
    assert api_v1_response.json()["data"] == v1_response.json()["data"]


def test_model_retrieve_supports_aliases_and_fusion_ids(client):
    headers = {"Authorization": "Bearer route-key"}

    fugu = client.get("/v1/models/fugu", headers=headers)
    fusion_fugu = client.get("/v1/models/fusion/fugu", headers=headers)
    api_v1_alias = client.get("/api/v1/models/openrouter/fusion", headers=headers)

    assert fugu.status_code == 200
    assert fugu.json()["id"] == "fugu"
    assert fugu.json()["alias_of"] == "fusion/fugu"
    assert fugu.json()["status"] == "compat_placeholder - not conductor-backed yet"

    assert fusion_fugu.status_code == 200
    assert fusion_fugu.json()["id"] == "fusion/fugu"
    assert fusion_fugu.json()["status"] == "compat_placeholder - not conductor-backed yet"

    assert api_v1_alias.status_code == 200
    assert api_v1_alias.json()["id"] == "openrouter/fusion"
    assert api_v1_alias.json()["alias_of"] == "fusion/general"


@pytest.mark.asyncio
async def test_api_v1_chat_mirror_uses_same_route(tmp_path, monkeypatch):
    import omnifusion.api.chat as chat_mod
    from omnifusion.main import app

    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "m2_chat_mirror.db")
    settings.omnifusion_api_keys = ["chat-key"]

    async def fake_run_fusion(run_id, preset, body, key_hash):
        return {
            "id": "chatcmpl-route",
            "object": "chat.completion",
            "created": 1,
            "model": f"fusion/{preset.name}",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": body.model},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    try:
        await init_db()
        monkeypatch.setattr(chat_mod, "run_fusion", fake_run_fusion)
        with TestClient(app) as test_client:
            response = test_client.post(
                "/api/v1/chat/completions",
                headers={"Authorization": "Bearer chat-key"},
                json={
                    "model": "fugu-ultra",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys

    assert response.status_code == 200
    assert response.json()["model"] == "fusion/fugu-ultra"
