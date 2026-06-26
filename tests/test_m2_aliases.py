import pytest
from fastapi.testclient import TestClient

from omnifusion.api.model_names import normalize_requested_model
from omnifusion.fusion.types import FusionTrace
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.store.runs import get_trace, save_trace


PLACEHOLDER_STATUS = "compat_placeholder - not conductor-backed yet"


def test_model_aliases_normalize_to_fusion_presets():
    assert normalize_requested_model("openrouter/fusion") == "fusion/general"
    assert normalize_requested_model("fugu") == "fusion/fugu"
    assert normalize_requested_model("fugu-ultra") == "fusion/fugu-ultra"
    assert normalize_requested_model("openai/fusion/general") == "fusion/general"


@pytest.mark.asyncio
async def test_fugu_alias_creates_placeholder_and_runs(tmp_path, monkeypatch):
    import omnifusion.api.chat as chat_mod
    from omnifusion.main import app

    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "m2_alias.db")
    settings.omnifusion_api_keys = ["alias-key"]

    captured = {}

    async def fake_run_fusion(run_id, preset, body, key_hash):
        captured["preset"] = preset.name
        captured["status"] = preset.compat_status
        captured["model"] = body.model
        return {
            "id": "chatcmpl-alias",
            "object": "chat.completion",
            "created": 1,
            "model": f"fusion/{preset.name}",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    try:
        await init_db()
        monkeypatch.setattr(chat_mod, "run_fusion", fake_run_fusion)

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer alias-key"},
                json={
                    "model": "fugu",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys

    assert response.status_code == 200
    assert captured == {
        "preset": "fugu",
        "status": PLACEHOLDER_STATUS,
        "model": "fusion/fugu",
    }


def test_models_list_self_labels_placeholder_aliases(tmp_path):
    from omnifusion.main import app

    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "m2_models.db")
    settings.omnifusion_api_keys = ["models-key"]

    try:
        with TestClient(app) as client:
            response = client.get(
                "/v1/models",
                headers={"Authorization": "Bearer models-key"},
            )
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys

    assert response.status_code == 200
    by_id = {model["id"]: model for model in response.json()["data"]}
    assert by_id["fugu"]["status"] == PLACEHOLDER_STATUS
    assert by_id["fugu-ultra"]["status"] == PLACEHOLDER_STATUS
    assert by_id["fugu"]["alias_of"] == "fusion/fugu"
    assert by_id["openrouter/fusion"]["alias_of"] == "fusion/general"


@pytest.mark.asyncio
async def test_trace_metadata_round_trips_placeholder_status(tmp_path):
    old_db = settings.db_path
    settings.db_path = str(tmp_path / "m2_trace.db")

    try:
        await init_db()
        trace = FusionTrace(
            run_id="trace-placeholder",
            preset="fugu",
            cost_usd=0.0,
            wall_ms=1,
            panel_results=[],
            metadata={"model_status": PLACEHOLDER_STATUS},
        )
        await save_trace(trace, True, "keyhash")
        stored = await get_trace("trace-placeholder", "keyhash")
    finally:
        settings.db_path = old_db

    assert stored.metadata["model_status"] == PLACEHOLDER_STATUS
