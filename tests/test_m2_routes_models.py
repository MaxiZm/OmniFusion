import pytest
from fastapi.testclient import TestClient

from omnifusion.settings import settings
from omnifusion.store.db import init_db


async def _seed_general():
    from omnifusion.fusion.types import Preset, PresetStage
    from omnifusion.store.presets import save_preset

    stage = PresetStage(max_tokens=64, timeout=10)
    await save_preset(
        Preset(
            name="general",
            strategy="B",
            panel_models=["panel-a"],
            panel=stage,
            judge_model="judge-a",
            judge=stage,
            final_model="final-a",
            final=stage,
            cost_ceiling=1.0,
        )
    )


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


def test_model_retrieve_supports_alias_and_rejects_removed_fugu(client):
    headers = {"Authorization": "Bearer route-key"}

    api_v1_alias = client.get("/api/v1/models/openrouter/fusion", headers=headers)
    removed_fugu = client.get("/v1/models/fugu", headers=headers)

    assert api_v1_alias.status_code == 200
    assert api_v1_alias.json()["id"] == "openrouter/fusion"
    assert api_v1_alias.json()["alias_of"] == "fusion/general"

    # The fugu compatibility aliases were removed.
    assert removed_fugu.status_code == 404


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
        await _seed_general()
        monkeypatch.setattr(chat_mod, "run_fusion", fake_run_fusion)
        with TestClient(app) as test_client:
            response = test_client.post(
                "/api/v1/chat/completions",
                headers={"Authorization": "Bearer chat-key"},
                json={
                    "model": "fusion/general",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys

    assert response.status_code == 200
    assert response.json()["model"] == "fusion/general"
