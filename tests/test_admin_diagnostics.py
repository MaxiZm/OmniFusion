"""Operator diagnostics: clean vs configured states, unhealthy path, no secret leakage,
and end-to-end rendering of the diagnostics/budget admin pages."""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import omnifusion.admin.routes as admin_routes
from omnifusion.admin.diagnostics import collect_diagnostics
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.store.presets import save_preset
from omnifusion.store.providers import save_provider


@pytest.fixture
def diag_db(tmp_path):
    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "diagnostics.db")
    settings.omnifusion_api_keys = []
    try:
        yield
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys


def _assert_no_secrets(diag):
    blob = json.dumps(diag)
    assert "test-password-123" not in blob
    assert settings.omnifusion_secret_key.get_secret_value() not in blob


@pytest.mark.asyncio
async def test_clean_db_warns(diag_db):
    await init_db()
    diag = await collect_diagnostics()

    assert diag["database"]["status"] == "ok"
    assert diag["startup"]["ok"] is True
    # Nothing configured yet → advisory warnings, not failure.
    assert diag["status"] == "warn"
    assert diag["auth"]["api_key_count"] == 0
    assert diag["default_preset"]["exists"] is False
    assert diag["providers"]["count"] == 0
    joined = " ".join(diag["warnings"]).lower()
    assert "api_keys" in joined or "api keys" in joined
    assert "preset" in joined
    assert "provider" in joined
    _assert_no_secrets(diag)


@pytest.mark.asyncio
async def test_configured_state_is_cleaner(diag_db):
    await init_db()
    settings.omnifusion_api_keys = ["live-key"]
    await save_provider(
        provider_id="default", p_type="openai", plain_key="sk-x", models=["m"]
    )
    await save_preset(
        Preset(
            name=settings.omnifusion_default_fusion_preset,
            strategy="B",
            panel_models=["m"],
            panel=PresetStage(max_tokens=16, timeout=5),
            judge_model="m",
            judge=PresetStage(max_tokens=16, timeout=5),
            final_model="m",
            final=PresetStage(max_tokens=16, timeout=5),
            cost_ceiling=1.0,
            min_panel_success=1,
        )
    )

    diag = await collect_diagnostics()
    assert diag["auth"]["api_key_count"] == 1
    assert diag["default_preset"]["exists"] is True
    assert diag["providers"]["count"] == 1
    assert diag["providers"]["entries"][0]["has_encrypted_key"] is True
    # The provider summary never carries key material.
    _assert_no_secrets(diag)


@pytest.mark.asyncio
async def test_unhealthy_when_startup_check_fails(diag_db, monkeypatch):
    await init_db()

    def _boom():
        raise ValueError("OMNIFUSION_SECRET_KEY is still a placeholder value.")

    monkeypatch.setattr(
        "omnifusion.admin.diagnostics.validate_startup_security", _boom
    )
    diag = await collect_diagnostics()
    assert diag["startup"]["ok"] is False
    assert diag["status"] == "unhealthy"
    assert any("startup" in w.lower() for w in diag["warnings"])


# ── End-to-end admin rendering (catches template errors) ──────────────────────


@pytest.fixture
def admin_client(tmp_path):
    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    old_secure = settings.omnifusion_secure_cookie
    old_pw = settings.omnifusion_admin_password
    settings.db_path = str(tmp_path / "admin-diag.db")
    settings.omnifusion_api_keys = ["live-key"]
    settings.omnifusion_secure_cookie = False
    settings.omnifusion_admin_password = SecretStr("test-password-123")
    admin_routes._admin_hash = None
    try:
        from omnifusion.main import app

        yield TestClient(app)
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys
        settings.omnifusion_secure_cookie = old_secure
        settings.omnifusion_admin_password = old_pw
        admin_routes._admin_hash = None


def _login(client):
    res = client.post(
        "/admin/login",
        data={"username": "admin", "password": "test-password-123"},
        follow_redirects=False,
    )
    assert res.status_code == 303
    client.cookies.update(res.cookies)


@pytest.mark.asyncio
async def test_diagnostics_and_budget_pages_render(admin_client):
    await init_db()
    client = admin_client
    _login(client)

    page = client.get("/admin/diagnostics")
    assert page.status_code == 200
    assert "Diagnostics" in page.text

    diag_json = client.get("/admin/diagnostics.json")
    assert diag_json.status_code == 200
    assert "status" in diag_json.json()

    budget = client.get("/admin/budget")
    assert budget.status_code == 200
    assert "Budget ledger" in budget.text

    budget_json = client.get("/admin/budget.json")
    assert budget_json.status_code == 200
    assert "global_daily" in budget_json.json()
