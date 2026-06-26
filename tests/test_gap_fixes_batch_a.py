"""Regression tests for audit-found correctness bugs (Batch A)."""
import os

import pytest
from pydantic import BaseModel

from omnifusion.api.normalize import generation_passthrough_kwargs
from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage, ToolCall, ToolCallFunction
from omnifusion.fusion.orchestrator import run_fusion
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.settings import settings
from omnifusion.store.db import init_db


def test_generation_passthrough_kwargs_collects_named_fields():
    body = ChatCompletionRequest(
        model="m",
        messages=[ChatMessage(role="user", content="hi")],
        seed=7,
        presence_penalty=0.5,
        frequency_penalty=-0.25,
        service_tier="default",
        parallel_tool_calls=False,
    )
    base = generation_passthrough_kwargs(body)
    assert base == {
        "seed": 7,
        "presence_penalty": 0.5,
        "frequency_penalty": -0.25,
        "service_tier": "default",
    }
    # parallel_tool_calls only included for tool-bearing paths.
    assert "parallel_tool_calls" not in base
    with_tools = generation_passthrough_kwargs(body, include_tool_params=True)
    assert with_tools["parallel_tool_calls"] is False


def test_generation_passthrough_kwargs_omits_unset():
    body = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])
    assert generation_passthrough_kwargs(body, include_tool_params=True) == {}


def test_tool_notes_extracted_from_typed_tool_calls():
    """[18] Multi-turn tool context must survive M1c typed tool_calls."""
    from omnifusion.fusion.tool_orchestrator import _normalize_tool_calls

    tc = ToolCall(id="c1", type="function", function=ToolCallFunction(name="search", arguments="{}"))
    normalized = _normalize_tool_calls([tc])
    assert normalized[0]["function"]["name"] == "search"


class _MockMessage:
    def __init__(self, content):
        self.content = content


class _MockChoice:
    def __init__(self, content):
        self.message = _MockMessage(content)
        self.finish_reason = "stop"


class _MockUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class _MockResponse:
    def __init__(self, content, pt=1, ct=1):
        self.choices = [_MockChoice(content)]
        self.usage = _MockUsage(prompt_tokens=pt, completion_tokens=ct)


@pytest.mark.asyncio
async def test_seed_and_penalties_reach_panel_and_synthesis(tmp_path, monkeypatch):
    """[missed] seed/penalties must reach provider calls, not be silently dropped."""
    import omnifusion.llm.client as client_mod

    old_db = settings.db_path
    settings.db_path = str(tmp_path / "gen.db")
    seen = {}

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        seen[model] = kwargs
        if model == "judge-a":
            return _MockResponse('{"consensus": "ok"}')
        return _MockResponse("answer")

    try:
        await init_db()
        monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
        stage = PresetStage(max_tokens=64, timeout=5)
        preset = Preset(
            name="general",
            strategy="B",
            panel_models=["panel-a"],
            panel=stage,
            judge_model="judge-a",
            judge=stage,
            final_model="final-a",
            final=stage,
        )
        request = ChatCompletionRequest(
            model="fusion/general",
            messages=[ChatMessage(role="user", content="q")],
            seed=42,
            presence_penalty=0.3,
            stream=False,
            store=True,
        )
        await run_fusion("gen-run", preset, request, "keyhash")
    finally:
        if os.path.exists(settings.db_path):
            os.remove(settings.db_path)
        settings.db_path = old_db

    # Panel and final synthesis both received the caller's seed and penalty.
    assert seen["panel-a"].get("seed") == 42
    assert seen["panel-a"].get("presence_penalty") == 0.3
    assert seen["final-a"].get("seed") == 42
    assert seen["final-a"].get("presence_penalty") == 0.3
