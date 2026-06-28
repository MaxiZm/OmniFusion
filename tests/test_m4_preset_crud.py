import json

import pytest
from fastapi.testclient import TestClient

from omnifusion.fusion.types import Preset, PresetPrompts
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.store.presets import get_preset


def v2_payload(name="api-v2"):
    stage = {"max_tokens": 16, "timeout": 5}
    return {
        "name": name,
        "display_name": "API V2",
        "mode": "fusion",
        "version": 2,
        "strategy": "B",
        "models": [
            {"provider_id": "default", "role": "panel", "model": "panel-a", "weight": 1.0},
            {"provider_id": "default", "role": "judge", "model": "judge-a", "weight": 1.0},
            {"provider_id": "default", "role": "final", "model": "final-a", "weight": 1.0},
        ],
        "prompts": {
            "global_prompt": "global",
            "role_prompts": {"panel": "panel prompt"},
        },
        "budgets": {
            "panel": stage,
            "judge": stage,
            "final": stage,
            "cost_ceiling": 1.0,
            "min_panel_success": 1,
        },
    }


@pytest.mark.asyncio
async def test_api_preset_crud_round_trips_v2(tmp_path):
    from omnifusion.main import app

    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "m4-api-crud.db")
    settings.omnifusion_api_keys = ["preset-key"]

    try:
        await init_db()
        with TestClient(app) as client:
            headers = {"Authorization": "Bearer preset-key"}
            create_response = client.put(
                "/v1/presets/api-v2",
                headers=headers,
                json=v2_payload(),
            )
            get_response = client.get("/api/v1/presets/api-v2", headers=headers)
            list_response = client.get("/v1/presets", headers=headers)
            delete_response = client.delete("/v1/presets/api-v2", headers=headers)
            missing_response = client.get("/v1/presets/api-v2", headers=headers)
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys

    assert create_response.status_code == 200
    assert get_response.status_code == 200
    assert get_response.json()["version"] == 2
    assert get_response.json()["prompts"]["role_prompts"]["panel"] == "panel prompt"
    assert "api-v2" in [preset["name"] for preset in list_response.json()["data"]]
    assert delete_response.status_code == 204
    assert missing_response.status_code == 404


@pytest.mark.asyncio
async def test_cli_preset_crud_helpers_round_trip_v2(tmp_path, capsys):
    from omnifusion.cli import preset_delete_async, preset_get_async, preset_list_async, preset_save_async

    old_db = settings.db_path
    settings.db_path = str(tmp_path / "m4-cli-crud.db")
    payload_path = tmp_path / "preset.json"
    payload_path.write_text(json.dumps(v2_payload("cli-v2")))

    try:
        await init_db()
        await preset_save_async(str(payload_path))
        saved = await get_preset("cli-v2")
        await preset_get_async("cli-v2")
        await preset_list_async()
        await preset_delete_async("cli-v2")
        deleted = await get_preset("cli-v2")
    finally:
        settings.db_path = old_db

    output = capsys.readouterr().out
    assert saved.version == 2
    assert "cli-v2" in output
    assert deleted is None


@pytest.mark.asyncio
async def test_admin_console_save_creates_v2_preset(tmp_path):
    from omnifusion.admin.routes import save_preset_route

    old_db = settings.db_path
    settings.db_path = str(tmp_path / "m4-admin-crud.db")

    try:
        await init_db()
        response = await save_preset_route(
            name="console-v2",
            strategy="B",
            panel_models_raw=["panel-a"],
            panel_max_tokens=16,
            panel_timeout=5,
            judge_model="judge-a",
            judge_max_tokens=16,
            judge_timeout=5,
            final_model="final-a",
            final_max_tokens=16,
            final_timeout=5,
            cost_ceiling=1.0,
            on_final_failure="error",
            min_panel_success=1,
            display_name="Console V2",
            mode="fusion",
            web_enabled=True,
            prompt_global="be concise",
            prompt_panel="draft",
            prompt_judge="score",
            prompt_final="merge",
            session={"user": "admin"},
        )
        saved = await get_preset("console-v2")
    finally:
        settings.db_path = old_db

    assert response.status_code == 303
    assert isinstance(saved, Preset)
    assert saved.version == 2
    # The console now authors the full PresetV2 surface, not just the flat subset.
    assert saved.display_name == "Console V2"
    assert saved.web_enabled is True
    assert isinstance(saved.prompts, PresetPrompts)
    assert saved.prompts.global_prompt == "be concise"
    assert saved.prompts.role_prompts["judge"] == "score"


@pytest.mark.asyncio
async def test_admin_console_save_validation_error_returns_400(tmp_path):
    from omnifusion.admin.routes import save_preset_route

    old_db = settings.db_path
    settings.db_path = str(tmp_path / "m4-admin-invalid.db")

    try:
        await init_db()
        response = await save_preset_route(
            name="bad-console",
            strategy="B",
            panel_models_raw=["panel-a"],
            panel_max_tokens=16,
            panel_timeout=5,
            judge_model="judge-a",
            judge_max_tokens=settings.omnifusion_max_tokens_limit + 1,
            judge_timeout=5,
            final_model="final-a",
            final_max_tokens=16,
            final_timeout=5,
            cost_ceiling=1.0,
            on_final_failure="error",
            min_panel_success=1,
            display_name="Bad Console",
            mode="fusion",
            web_enabled=False,
            prompt_global="",
            prompt_panel="",
            prompt_judge="",
            prompt_final="",
            session={"user": "admin"},
        )
    finally:
        settings.db_path = old_db

    assert response.status_code == 400
    assert b"stage max_tokens must be &lt;=" in response.body
