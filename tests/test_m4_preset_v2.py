import json

import pytest

from omnifusion.fusion.types import PresetStage
from omnifusion.settings import settings
from omnifusion.store.db import get_db_connection, init_db


async def insert_legacy_preset(spec: dict):
    async with get_db_connection() as db:
        await db.execute(
            "INSERT INTO presets (name, strategy, spec_json) VALUES (?, ?, ?)",
            (spec["name"], spec.get("strategy", "B"), json.dumps(spec)),
        )
        await db.commit()


def legacy_spec(name="legacy"):
    stage = {"max_tokens": 16, "timeout": 5}
    return {
        "name": name,
        "strategy": "B",
        "panel_models": ["panel-a", "panel-b"],
        "panel": stage,
        "judge_model": "judge-a",
        "judge": stage,
        "final_model": "final-a",
        "final": stage,
        "usage_reporting": "aggregate",
        "cost_ceiling": 1.5,
        "on_final_failure": "best_panel",
        "min_panel_success": 2,
    }


@pytest.mark.asyncio
async def test_store_upgrades_legacy_preset_rows_to_v2_compatible_objects(tmp_path):
    from omnifusion.fusion.types import PresetV2
    from omnifusion.store.presets import get_preset

    old_db = settings.db_path
    settings.db_path = str(tmp_path / "m4-legacy.db")

    try:
        await init_db()
        await insert_legacy_preset(legacy_spec())
        preset = await get_preset("legacy")
    finally:
        settings.db_path = old_db

    assert isinstance(preset, PresetV2)
    assert preset.version == 2
    assert preset.display_name == "legacy"
    assert preset.mode == "fusion"
    assert preset.panel_models == ["panel-a", "panel-b"]
    assert preset.judge_model == "judge-a"
    assert preset.final_model == "final-a"
    assert preset.budgets.cost_ceiling == 1.5
    assert preset.budgets.min_panel_success == 2
    assert [(model.role, model.model) for model in preset.models] == [
        ("panel", "panel-a"),
        ("panel", "panel-b"),
        ("judge", "judge-a"),
        ("final", "final-a"),
    ]


@pytest.mark.asyncio
async def test_fugu_placeholders_are_v2_configs(tmp_path):
    from omnifusion.store.presets import ensure_compat_placeholder_presets, get_preset

    old_db = settings.db_path
    settings.db_path = str(tmp_path / "m4-placeholders.db")

    try:
        await init_db()
        await ensure_compat_placeholder_presets()
        fugu = await get_preset("fugu")
        ultra = await get_preset("fugu-ultra")
    finally:
        settings.db_path = old_db

    assert fugu.version == 2
    assert fugu.display_name == "Fugu"
    assert fugu.mode == "fugu_compat"
    assert fugu.compat_status == "compat_placeholder - not conductor-backed yet"
    assert {model.role for model in fugu.models} == {"panel", "judge", "final"}
    assert ultra.version == 2
    assert ultra.display_name == "Fugu Ultra"
    assert ultra.mode == "fugu_compat"


@pytest.mark.asyncio
async def test_role_prompts_are_consumed_and_redacted(monkeypatch):
    from omnifusion.fusion.panel import run_panelist
    from omnifusion.fusion.types import Preset, PresetPrompts, trace_metadata_for_preset

    captured = {}

    class FakeMessage:
        content = "panel answer"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]
        usage = None

    async def fake_call(self, stage, **kwargs):
        captured["messages"] = kwargs["messages"]
        return FakeResponse()

    monkeypatch.setattr(
        "omnifusion.fusion.panel.BudgetedExecutor.call",
        fake_call,
    )

    stage = PresetStage(max_tokens=16, timeout=5)
    preset = Preset(
        name="prompted",
        strategy="B",
        panel_models=["panel-a"],
        panel=stage,
        judge_model="judge-a",
        judge=stage,
        final_model="final-a",
        final=stage,
        prompts=PresetPrompts(
            global_prompt="GLOBAL SECRET PROMPT",
            role_prompts={"panel": "PANEL SECRET PROMPT"},
        ),
    )

    result = await run_panelist("run-prompt", "panel-a", preset, [])

    assert result.status == "ok"
    assert captured["messages"][0]["role"] == "system"
    assert "GLOBAL SECRET PROMPT" in captured["messages"][0]["content"]
    assert "PANEL SECRET PROMPT" in captured["messages"][0]["content"]

    metadata = trace_metadata_for_preset(preset)
    assert metadata["preset_version"] == 2
    assert metadata["role_prompts_redacted"] is True
    assert "GLOBAL SECRET PROMPT" not in json.dumps(metadata)
    assert "PANEL SECRET PROMPT" not in json.dumps(metadata)
