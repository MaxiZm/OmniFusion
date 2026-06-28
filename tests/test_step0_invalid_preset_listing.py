"""Regression: a single stored preset that violates the *current* operational
limits (e.g. an operator lowered OMNIFUSION_MAX_TOKENS_LIMIT after it was saved)
must not 500 every listing path.

Before the fix, ``list_presets()`` hard-validated each row with
``Preset.model_validate_json`` and let the first ValidationError propagate, which
took down ``/admin/presets``, ``/admin/playground``, ``/v1/models`` and
``/v1/presets`` all at once. The bad row should now be isolated, logged, and
surfaced to the admin instead of crashing the page.
"""

import json
import os

# Configure env before importing anything from omnifusion (mirrors test_security).
os.environ.setdefault("OMNIFUSION_ADMIN_PASSWORD", "admin-password-invalid-preset")
os.environ.setdefault(
    "OMNIFUSION_SECRET_KEY", "U1NfdlhjdmJubWwwMTIzNDU2Nzg5MGFiY2RlZmdoaWo="
)

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from omnifusion.admin import routes as admin_routes
from omnifusion.api.errors import OmniFusionError
from omnifusion.main import app
from omnifusion.settings import settings
from omnifusion.store.db import get_db_connection, init_db
from omnifusion.store.presets import (
    get_preset,
    list_presets,
    list_presets_with_invalid,
    save_preset,
)
from omnifusion.fusion.types import Preset, PresetStage


POISON_NAME = "Anton"
VALID_NAME = "good-preset"


def _poison_spec_json() -> str:
    """A structurally valid preset whose max_tokens/timeout exceed the limits
    enforced by the current Preset validators (the real-world 'Anton' shape)."""
    over_tokens = settings.omnifusion_max_tokens_limit + 1
    over_timeout = settings.omnifusion_max_stage_timeout + 1
    stage = {"max_tokens": over_tokens, "timeout": over_timeout}
    return json.dumps(
        {
            "name": POISON_NAME,
            "strategy": "B",
            "panel_models": ["deepseek/deepseek-v4-pro"],
            "panel": stage,
            "judge_model": "deepseek/deepseek-v4-pro",
            "judge": stage,
            "final_model": "deepseek/deepseek-v4-pro",
            "final": stage,
            "cost_ceiling": 1.0,
            "min_panel_success": 1,
        }
    )


async def _insert_raw_preset(name: str, spec_json: str) -> None:
    """Insert a spec_json directly, bypassing the model so we can simulate a row
    that was valid when written but is invalid under today's limits."""
    async with get_db_connection() as db:
        await db.execute(
            "INSERT INTO presets (name, strategy, spec_json) VALUES (?, ?, ?)",
            (name, "B", spec_json),
        )
        await db.commit()


@pytest.fixture
def client(tmp_path):
    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    old_secure_cookie = settings.omnifusion_secure_cookie
    old_admin_password = settings.omnifusion_admin_password

    settings.db_path = str(tmp_path / "invalid-preset.db")
    settings.omnifusion_api_keys = ["preset-key"]
    settings.omnifusion_secure_cookie = False
    settings.omnifusion_admin_password = SecretStr("test-password-123")
    admin_routes._admin_hash = None

    test_client = TestClient(app)
    yield test_client

    settings.db_path = old_db
    settings.omnifusion_api_keys = old_keys
    settings.omnifusion_secure_cookie = old_secure_cookie
    settings.omnifusion_admin_password = old_admin_password
    admin_routes._admin_hash = None


def _admin_login(client: TestClient) -> None:
    res = client.post(
        "/admin/login",
        data={"username": "admin", "password": "test-password-123"},
        follow_redirects=False,
    )
    assert res.status_code == 303
    client.cookies.update(res.cookies)


@pytest.mark.asyncio
async def test_list_presets_skips_invalid_rows(tmp_path):
    settings.db_path = str(tmp_path / "store-level.db")
    await init_db()

    await save_preset(
        Preset(
            name=VALID_NAME,
            strategy="B",
            panel_models=["m-a"],
            panel=PresetStage(max_tokens=16, timeout=5),
            judge_model="m-j",
            judge=PresetStage(max_tokens=16, timeout=5),
            final_model="m-f",
            final=PresetStage(max_tokens=16, timeout=5),
            cost_ceiling=1.0,
            min_panel_success=1,
        )
    )
    await _insert_raw_preset(POISON_NAME, _poison_spec_json())

    # list_presets must not raise and must drop the poison row.
    presets = await list_presets()
    names = {p.name for p in presets}
    assert VALID_NAME in names
    assert POISON_NAME not in names

    # Partitioned view exposes the bad row with a human-readable reason.
    valid, invalid = await list_presets_with_invalid()
    assert {p.name for p in valid} == {VALID_NAME}
    assert [ip.name for ip in invalid] == [POISON_NAME]
    assert "max_tokens" in invalid[0].error

    # get_preset on the poison row raises a clean, typed 422 (not a raw 500).
    with pytest.raises(OmniFusionError) as exc:
        await get_preset(POISON_NAME)
    assert exc.value.status_code == 422
    assert exc.value.code == "preset_invalid"

    # A valid preset still loads fine through get_preset.
    assert (await get_preset(VALID_NAME)).name == VALID_NAME


@pytest.mark.asyncio
async def test_listing_endpoints_survive_poison_preset(client, tmp_path):
    await init_db()
    await _insert_raw_preset(POISON_NAME, _poison_spec_json())

    headers = {"Authorization": "Bearer preset-key"}

    # OpenAI-compatible endpoints that also call list_presets() must stay up.
    models_res = client.get("/v1/models", headers=headers)
    assert models_res.status_code == 200
    assert POISON_NAME not in json.dumps(models_res.json())

    presets_res = client.get("/v1/presets", headers=headers)
    assert presets_res.status_code == 200

    # The two admin pages from the bug report.
    _admin_login(client)

    presets_page = client.get("/admin/presets")
    assert presets_page.status_code == 200
    # The bad preset is surfaced with a delete affordance, not hidden entirely.
    assert POISON_NAME in presets_page.text
    assert f"/admin/presets/{POISON_NAME}/delete" in presets_page.text

    playground_page = client.get("/admin/playground")
    assert playground_page.status_code == 200


@pytest.mark.asyncio
async def test_corrupt_row_with_non_validation_error_is_isolated(tmp_path):
    """A hand-edited/corrupt row can fail inside Preset's mode='before' validator
    with a non-ValidationError (e.g. a non-iterable ``models``), which pydantic
    does not wrap. The resilient loaders must still isolate it rather than 500."""
    settings.db_path = str(tmp_path / "corrupt-row.db")
    await init_db()

    # `models` as a scalar makes the before-validator do `for model in 123` -> TypeError.
    corrupt = json.dumps({"name": "corrupt", "strategy": "B", "models": 123})
    await _insert_raw_preset("corrupt", corrupt)

    valid, invalid = await list_presets_with_invalid()
    assert "corrupt" not in {p.name for p in valid}
    assert "corrupt" in {ip.name for ip in invalid}

    with pytest.raises(OmniFusionError) as exc:
        await get_preset("corrupt")
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_poisoned_compat_preset_self_heals(client, tmp_path):
    """A poisoned reserved compat preset (fugu/fugu-ultra) must NOT take down
    /v1/models or app startup: ensure_compat_placeholder_presets() runs before
    the resilient list_presets(), so it has to regenerate the bad row."""
    await init_db()

    # Seed a 'fugu' row whose stored stage values exceed the current limits.
    over = settings.omnifusion_max_tokens_limit + 1
    stage = {"max_tokens": over, "timeout": settings.omnifusion_max_stage_timeout + 1}
    poison_fugu = json.dumps(
        {
            "name": "fugu",
            "strategy": "B",
            "panel_models": ["x"],
            "panel": stage,
            "judge_model": "x",
            "judge": stage,
            "final_model": "x",
            "final": stage,
        }
    )
    await _insert_raw_preset("fugu", poison_fugu)

    headers = {"Authorization": "Bearer preset-key"}
    res = client.get("/v1/models", headers=headers)
    assert res.status_code == 200
    ids = {entry["id"] for entry in res.json()["data"]}
    assert "fusion/fugu" in ids

    # The poison row was regenerated in place, so it now loads cleanly.
    healed = await get_preset("fugu")
    assert healed is not None
    assert healed.mode == "fugu_compat"
