from omnifusion.fusion.types import PanelResult, Preset, PresetModel, PresetStage


def preset(**kwargs):
    stage = PresetStage(max_tokens=64, timeout=5)
    data = {
        "name": "bandit-test",
        "strategy": "B",
        "panel_models": ["panel-a", "panel-b"],
        "panel": stage,
        "judge_model": "judge",
        "judge": stage,
        "final_model": "final",
        "final": stage,
    }
    data.update(kwargs)
    return Preset(**data)


def test_bandit_disabled_preserves_panel_model_order():
    from omnifusion.fusion.runtime.bandit import select_panel_models

    assert select_panel_models(preset()) == ["panel-a", "panel-b"]


def test_bandit_enabled_uses_preset_model_weights():
    from omnifusion.fusion.runtime.bandit import select_panel_models

    weighted = preset(
        bandit={"enabled": True, "exploration": 0.0},
        models=[
            PresetModel(role="panel", model="panel-a", weight=0.1),
            PresetModel(role="panel", model="panel-b", weight=3.0),
            PresetModel(role="judge", model="judge"),
            PresetModel(role="final", model="final"),
        ],
    )

    assert select_panel_models(weighted) == ["panel-b", "panel-a"]


def test_trace_stats_rank_observed_high_reward_model():
    from omnifusion.fusion.runtime.bandit import (
        model_stats_from_panel_results,
        select_panel_models,
    )

    stats = model_stats_from_panel_results(
        [
            PanelResult(model="panel-a", status="ok", cost_usd=0.10),
            PanelResult(model="panel-a", status="error", cost_usd=0.10),
            PanelResult(model="panel-b", status="ok", cost_usd=0.01),
        ]
    )
    configured = preset(bandit={"enabled": True, "exploration": 0.0})

    assert stats["panel-b"].reward_mean > stats["panel-a"].reward_mean
    assert select_panel_models(configured, stats=stats) == ["panel-b", "panel-a"]
