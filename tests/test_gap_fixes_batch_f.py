"""M6 conductor repair-loop coverage (Batch F): no-repair, max_repairs>1, and clean
degradation when a repair stage errors."""
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


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)
        self.finish_reason = "stop"


class _Usage(BaseModel):
    prompt_tokens: int = 1
    completion_tokens: int = 2


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()
        self._omnifusion_cost_usd = 0.001


def _preset():
    stage = PresetStage(max_tokens=64, timeout=5)
    return Preset(
        name="conductor-f",
        strategy="conductor",
        panel_models=["worker-a"],
        panel=stage,
        judge_model="judge-model",
        judge=stage,
        final_model="final-model",
        final=stage,
        min_panel_success=1,
    )


async def _run(monkeypatch, tmp_path, fake_call, max_repairs=1, db="m6f.db"):
    import omnifusion.fusion.strategies.conductor as conductor_mod

    old_db = settings.db_path
    old_repairs = settings.omnifusion_conductor_max_repairs
    settings.db_path = str(tmp_path / db)
    settings.omnifusion_conductor_max_repairs = max_repairs
    try:
        await init_db()
        monkeypatch.setattr(conductor_mod.BudgetedExecutor, "call", fake_call)
        request = ChatCompletionRequest(
            model="fusion/conductor-f",
            messages=[ChatMessage(role="user", content="solve it")],
            stream=False,
            store=True,
        )
        run_id = f"m6f-{db}"
        response = await run_fusion(run_id, _preset(), request, "k")
        trace = await get_trace(run_id, "k")
        return response, trace
    finally:
        if os.path.exists(settings.db_path):
            os.remove(settings.db_path)
        settings.db_path = old_db
        settings.omnifusion_conductor_max_repairs = old_repairs


@pytest.mark.asyncio
async def test_conductor_no_repair_path(tmp_path, monkeypatch):
    stages = []

    async def fake_call(self, stage, **kwargs):
        stages.append(stage)
        if stage == "verify":
            return _Resp(json.dumps({"consensus": "good", "needs_repair": False}))
        if stage == "merge":
            return _Resp("merged")
        return _Resp("draft")

    response, trace = await _run(tmp_path=tmp_path, monkeypatch=monkeypatch, fake_call=fake_call, db="norepair.db")
    assert stages == ["plan", "worker/worker-a", "verify", "merge"]
    assert trace.metadata["conductor"]["repair_count"] == 0
    assert response["choices"][0]["message"]["content"] == "merged"


@pytest.mark.asyncio
async def test_conductor_honors_max_repairs_greater_than_one(tmp_path, monkeypatch):
    stages = []

    async def fake_call(self, stage, **kwargs):
        stages.append(stage)
        if stage == "plan":
            return _Resp("plan")
        if stage.startswith("worker/"):
            return _Resp("draft")
        # The initial verify and every re-verify keep asking for repair.
        if stage == "verify" or stage.startswith("verify/repair-"):
            return _Resp(json.dumps({"consensus": "x", "needs_repair": True, "repair_instructions": "again"}))
        if stage.startswith("repair/"):
            return _Resp("repaired")
        if stage == "merge":
            return _Resp("merged")
        raise AssertionError(stage)

    response, trace = await _run(
        tmp_path=tmp_path, monkeypatch=monkeypatch, fake_call=fake_call, max_repairs=2, db="max2.db"
    )
    # Two repairs ran (with a re-verify between them), proving max_repairs>1 is live.
    assert "repair/1" in stages and "repair/2" in stages
    assert "verify/repair-1" in stages
    assert trace.metadata["conductor"]["repair_count"] == 2


@pytest.mark.asyncio
async def test_conductor_degrades_cleanly_when_repair_errors(tmp_path, monkeypatch):
    async def fake_call(self, stage, **kwargs):
        if stage == "verify":
            return _Resp(json.dumps({"consensus": "x", "needs_repair": True, "repair_instructions": "fix"}))
        if stage.startswith("repair/"):
            raise RuntimeError("repair model exploded")
        if stage == "merge":
            return _Resp("merged from unrepaired draft")
        return _Resp("draft")

    response, trace = await _run(
        tmp_path=tmp_path, monkeypatch=monkeypatch, fake_call=fake_call, db="degrade.db"
    )
    # The run completes (merge) rather than crashing, and records the degradation.
    assert response["choices"][0]["message"]["content"] == "merged from unrepaired draft"
    assert trace.metadata["conductor"]["repair_count"] == 0
    assert trace.metadata["conductor"]["repair_degraded"] is True
