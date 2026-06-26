"""Regression tests for M4 unread-field violations (Batch B)."""
import pytest

from omnifusion.fusion.types import Preset, PresetStage


def _preset(**overrides):
    stage = PresetStage(max_tokens=64, timeout=5)
    data = dict(
        name="general",
        strategy="B",
        panel_models=["panel-a"],
        panel=stage,
        judge_model="judge-a",
        judge=stage,
        final_model="final-a",
        final=stage,
    )
    data.update(overrides)
    return Preset(**data)


def test_budgets_is_a_consumed_computed_field_that_roundtrips():
    """[11] budgets must be derived + serialized, not a dead stored field."""
    preset = _preset(cost_ceiling=1.5, min_panel_success=1)
    assert preset.budgets is not None
    assert preset.budgets.cost_ceiling == 1.5
    assert preset.budgets.panel.max_tokens == 64
    dumped = preset.model_dump_json()
    assert '"budgets"' in dumped
    # Round-trips through the store serialization without drift.
    reloaded = Preset.model_validate_json(dumped)
    assert reloaded.budgets.cost_ceiling == 1.5
    assert reloaded.panel_models == ["panel-a"]


def test_provider_id_for_honors_pool_entry():
    """[12] PresetModel.provider_id must actually drive provider resolution."""
    preset = Preset.model_validate(
        {
            "name": "g",
            "version": 2,
            "strategy": "B",
            "models": [
                {"provider_id": "prov-x", "role": "panel", "model": "m-a"},
                {"provider_id": "prov-y", "role": "judge", "model": "m-j"},
                {"provider_id": "prov-z", "role": "final", "model": "m-f"},
            ],
            "budgets": {
                "panel": {"max_tokens": 64, "timeout": 5},
                "judge": {"max_tokens": 64, "timeout": 5},
                "final": {"max_tokens": 64, "timeout": 5},
            },
        }
    )
    assert preset.provider_id_for("m-a", "panel") == "prov-x"
    assert preset.provider_id_for("m-j", "judge") == "prov-y"
    assert preset.provider_id_for("m-f", "final") == "prov-z"
    # Unknown model falls back to the default provider.
    assert preset.provider_id_for("nope") == "default"


@pytest.mark.asyncio
async def test_panel_uses_pool_provider_id(monkeypatch):
    """The panel executor call must receive the pool entry's provider_id."""
    import omnifusion.fusion.panel as panel_mod

    captured = {}

    class _Resp:
        class _C:
            class message:
                content = "ok"

        choices = [_C()]
        usage = None
        _omnifusion_cost_usd = 0.0

    async def fake_call(self, stage, *, provider_id, model, messages, max_tokens, **kwargs):
        captured["provider_id"] = provider_id
        return _Resp()

    monkeypatch.setattr(panel_mod.BudgetedExecutor, "call", fake_call)
    preset = Preset.model_validate(
        {
            "name": "g",
            "version": 2,
            "strategy": "B",
            "models": [
                {"provider_id": "prov-x", "role": "panel", "model": "m-a"},
                {"provider_id": "default", "role": "judge", "model": "m-j"},
                {"provider_id": "default", "role": "final", "model": "m-f"},
            ],
            "budgets": {
                "panel": {"max_tokens": 64, "timeout": 5},
                "judge": {"max_tokens": 64, "timeout": 5},
                "final": {"max_tokens": 64, "timeout": 5},
            },
        }
    )
    result = await panel_mod.run_panelist(
        "run", "m-a", preset, [{"role": "user", "content": "hi"}]
    )
    assert result.status == "ok"
    assert captured["provider_id"] == "prov-x"
