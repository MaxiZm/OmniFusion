import json
import os

import pytest
from pydantic import BaseModel

from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage
from omnifusion.fusion.orchestrator import run_fusion
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.store.runs import get_trace


class MockMessage:
    def __init__(self, content):
        self.content = content


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)
        self.finish_reason = "stop"


class MockUsage(BaseModel):
    prompt_tokens: int = 1
    completion_tokens: int = 2


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]
        self.usage = MockUsage()
        self._omnifusion_cost_usd = 0.001


def conductor_preset():
    stage = PresetStage(max_tokens=64, timeout=5)
    return Preset(
        name="conductor-test",
        strategy="conductor",
        panel_models=["worker-a"],
        panel=stage,
        judge_model="judge-model",
        judge=stage,
        final_model="final-model",
        final=stage,
        min_panel_success=1,
    )


def test_conductor_strategy_is_registered_but_not_default():
    from omnifusion.fusion.runtime.registry import registry

    assert registry.has("conductor")
    assert Preset(
        name="classic-default",
        panel_models=["worker-a"],
        panel=PresetStage(max_tokens=64, timeout=5),
        judge_model="judge-model",
        judge=PresetStage(max_tokens=64, timeout=5),
        final_model="final-model",
        final=PresetStage(max_tokens=64, timeout=5),
    ).strategy == "B"


@pytest.mark.asyncio
async def test_conductor_runs_budgeted_stages_repairs_once_and_traces(
    tmp_path, monkeypatch
):
    import omnifusion.fusion.strategies.conductor as conductor_mod

    old_db = settings.db_path
    old_repairs = settings.omnifusion_conductor_max_repairs
    settings.db_path = str(tmp_path / "m6_conductor.db")
    settings.omnifusion_conductor_max_repairs = 1
    calls = []

    async def fake_call(self, stage, **kwargs):
        calls.append({"stage": stage, "model": kwargs["model"], "messages": kwargs["messages"]})
        if stage == "plan":
            return MockResponse("plan: inspect, patch, test")
        if stage == "worker/worker-a":
            assert "plan: inspect" in kwargs["messages"][0]["content"]
            return MockResponse("worker draft")
        if stage == "verify":
            return MockResponse(
                json.dumps(
                    {
                        "consensus": "draft is close",
                        "contradictions": "missing edge case",
                        "synthesis_plan": "repair then merge",
                        "needs_repair": True,
                        "repair_instructions": "cover the edge case",
                    }
                )
            )
        if stage == "repair/1":
            assert "cover the edge case" in kwargs["messages"][0]["content"]
            return MockResponse("repaired draft")
        if stage == "merge":
            assert "repaired draft" in kwargs["messages"][0]["content"]
            return MockResponse("merged final")
        raise AssertionError(f"unexpected stage: {stage}")

    try:
        await init_db()
        monkeypatch.setattr(conductor_mod.BudgetedExecutor, "call", fake_call)
        request = ChatCompletionRequest(
            model="fusion/conductor-test",
            messages=[ChatMessage(role="user", content="solve it")],
            stream=False,
            store=True,
        )

        response = await run_fusion(
            "m6-conductor-run", conductor_preset(), request, "keyhash"
        )
        trace = await get_trace("m6-conductor-run", "keyhash")
    finally:
        if os.path.exists(settings.db_path):
            os.remove(settings.db_path)
        settings.db_path = old_db
        settings.omnifusion_conductor_max_repairs = old_repairs

    assert [call["stage"] for call in calls] == [
        "plan",
        "worker/worker-a",
        "verify",
        "repair/1",
        "merge",
    ]
    assert response["choices"][0]["message"]["content"] == "merged final"
    assert response["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 10,
        "total_tokens": 15,
    }
    assert trace is not None
    assert trace.final_answer == "merged final"
    assert trace.judge_analysis.contradictions == "missing edge case"
    assert trace.metadata["conductor"]["experimental"] is True
    assert trace.metadata["conductor"]["ablation_required"] is True
    assert trace.metadata["conductor"]["repair_count"] == 1
