import json

import pytest

from omnifusion.fusion.types import PanelResult, Preset, PresetStage


def preset():
    stage = PresetStage(max_tokens=64, timeout=5)
    return Preset(
        name="judge-parity",
        strategy="B",
        panel_models=["panel-a"],
        panel=stage,
        judge_model="judge-a",
        judge=stage,
        final_model="final-a",
        final=stage,
    )


def test_judge_prompt_requests_openrouter_parity_fields():
    from omnifusion.fusion.prompts import render_judge_prompt

    prompt = render_judge_prompt("question", {"MODEL_A": "answer"})

    for field in [
        "contradictions",
        "partial_coverage",
        "unique_insights",
        "blind_spots",
        "model_strengths",
        "synthesis_plan",
    ]:
        assert field in prompt


@pytest.mark.asyncio
async def test_run_judge_parses_extended_structured_fields(monkeypatch):
    import omnifusion.fusion.judge as judge_mod

    captured = {}

    class FakeMessage:
        content = json.dumps(
            {
                "consensus": "shared",
                "contradictions": "conflict",
                "partial_coverage": "partial",
                "unique_insights": {"MODEL_A": ["novel"]},
                "blind_spots": "missing edge",
                "model_strengths": {"MODEL_A": "fast"},
                "synthesis_plan": "merge carefully",
            }
        )

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]
        usage = None
        _omnifusion_cost_usd = 0.01

    async def fake_call(self, stage, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(judge_mod.BudgetedExecutor, "call", fake_call)

    analysis = await judge_mod.run_judge(
        "run-judge",
        preset(),
        [{"role": "user", "content": "question"}],
        [PanelResult(model="panel-a", status="ok", content="answer")],
    )

    assert captured["temperature"] == 0
    assert analysis.consensus == "shared"
    assert analysis.contradictions == "conflict"
    assert analysis.partial_coverage == "partial"
    assert analysis.unique_insights == {"MODEL_A": ["novel"]}
    assert analysis.blind_spots == "missing edge"
    assert analysis.model_strengths == {"MODEL_A": "fast"}
    assert analysis.synthesis_plan == "merge carefully"
