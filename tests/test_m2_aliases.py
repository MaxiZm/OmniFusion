import pytest
from fastapi.testclient import TestClient

from omnifusion.api.model_names import normalize_requested_model
from omnifusion.fusion.types import FusionTrace
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.store.runs import get_trace, save_trace


def test_model_aliases_normalize_to_fusion_presets():
    assert normalize_requested_model("openrouter/fusion") == "fusion/general"
    assert normalize_requested_model("openai/fusion/general") == "fusion/general"
    # Plain preset references pass through unchanged.
    assert normalize_requested_model("fusion/general") == "fusion/general"


def test_models_list_exposes_openrouter_alias_and_no_fugu(tmp_path):
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
    assert by_id["openrouter/fusion"]["alias_of"] == "fusion/general"
    # The fugu compatibility aliases were removed.
    assert "fugu" not in by_id
    assert "fugu-ultra" not in by_id


@pytest.mark.asyncio
async def test_trace_metadata_round_trips(tmp_path):
    old_db = settings.db_path
    settings.db_path = str(tmp_path / "m2_trace.db")

    try:
        await init_db()
        trace = FusionTrace(
            run_id="trace-metadata",
            preset="general",
            cost_usd=0.0,
            wall_ms=1,
            panel_results=[],
            metadata={"preset_version": 2},
        )
        await save_trace(trace, True, "keyhash")
        stored = await get_trace("trace-metadata", "keyhash")
    finally:
        settings.db_path = old_db

    assert stored.metadata["preset_version"] == 2
