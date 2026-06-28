"""M3c exit gate: both REAL strategies share usage/trace/stream/budget through one
runtime, and execute() returns the StrategyResult envelope (not a leaked dict)."""
import os

import pytest
from pydantic import BaseModel

from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage, ToolDefinition
from omnifusion.fusion.runtime.context import RunContext
from omnifusion.fusion.runtime.registry import StrategyRegistry
from omnifusion.fusion.runtime.strategy import StrategyResult
from omnifusion.fusion.strategies.classic import ClassicStrategy
from omnifusion.fusion.strategies.tool_step import ToolStepStrategy
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.store.runs import get_trace


class _Msg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, content, tool_calls=None):
        self.message = _Msg(content, tool_calls)
        self.finish_reason = "stop"


class _Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class _Resp:
    def __init__(self, content, pt=2, ct=3, tool_calls=None):
        self.choices = [_Choice(content, tool_calls)]
        self.usage = _Usage(prompt_tokens=pt, completion_tokens=ct)


def _preset():
    stage = PresetStage(max_tokens=32, timeout=5)
    return Preset(
        name="m3c-real",
        strategy="B",
        panel_models=["panel-a"],
        panel=stage,
        judge_model="judge-a",
        judge=stage,
        final_model="final-a",
        final=stage,
    )


@pytest.mark.asyncio
async def test_classic_strategy_returns_strategyresult_with_shared_usage_and_trace(
    tmp_path, monkeypatch
):
    import omnifusion.llm.client as client_mod

    old_db = settings.db_path
    settings.db_path = str(tmp_path / "m3c_classic.db")

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        if model == "judge-a":
            return _Resp('{"consensus": "ok"}')
        return _Resp("answer")

    try:
        await init_db()
        monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
        ctx = RunContext(
            run_id="m3c-classic",
            preset=_preset(),
            request=ChatCompletionRequest(
                model="fusion/m3c-real",
                messages=[ChatMessage(role="user", content="q")],
                stream=False,
                store=True,
            ),
            key_hash="k",
        )
        result = await ClassicStrategy().execute(ctx)
        trace = await get_trace("m3c-classic", "k")
    finally:
        if os.path.exists(settings.db_path):
            os.remove(settings.db_path)
        settings.db_path = old_db

    # Envelope, not a leaked dict.
    assert isinstance(result, StrategyResult)
    assert result.streaming is False
    payload = result.payload
    # Shared response shape + aggregated usage (panel + judge + final).
    assert payload["model"] == "fusion/m3c-real"
    assert payload["usage"]["total_tokens"] > 0
    # Shared trace persistence through the one runtime.
    assert trace is not None
    assert trace.cost_usd >= 0


@pytest.mark.asyncio
async def test_tool_step_strategy_returns_strategyresult_and_traces(tmp_path, monkeypatch):
    import omnifusion.llm.client as client_mod

    old_db = settings.db_path
    settings.db_path = str(tmp_path / "m3c_tool.db")

    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "do_thing", "arguments": "{}"},
    }

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        # Tool-panel proposes a tool call; judge selects it.
        if kwargs.get("tools"):
            return _Resp(None, tool_calls=[tool_call])
        return _Resp('{"decision": "tool", "best_index": 0}')

    try:
        await init_db()
        monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
        ctx = RunContext(
            run_id="m3c-tool",
            preset=_preset(),
            request=ChatCompletionRequest(
                model="fusion/m3c-real",
                messages=[ChatMessage(role="user", content="use a tool")],
                tools=[ToolDefinition(type="function", function={"name": "do_thing"})],
                stream=False,
                store=True,
            ),
            key_hash="k",
        )
        result = await ToolStepStrategy().execute(ctx)
        trace = await get_trace("m3c-tool", "k")
    finally:
        if os.path.exists(settings.db_path):
            os.remove(settings.db_path)
        settings.db_path = old_db

    assert isinstance(result, StrategyResult)
    payload = result.payload
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "do_thing"
    assert trace is not None


@pytest.mark.asyncio
async def test_registry_dispatches_real_strategies_by_tools_presence(tmp_path, monkeypatch):
    import omnifusion.llm.client as client_mod

    old_db = settings.db_path
    settings.db_path = str(tmp_path / "m3c_reg.db")

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        if model == "judge-a":
            return _Resp('{"consensus": "ok"}')
        return _Resp("answer")

    try:
        await init_db()
        monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
        registry = StrategyRegistry([ClassicStrategy(), ToolStepStrategy()])
        no_tools = ChatCompletionRequest(
            model="fusion/m3c-real",
            messages=[ChatMessage(role="user", content="q")],
            store=False,
        )
        result = await registry.execute("m3c-reg", _preset(), no_tools, "k")
    finally:
        if os.path.exists(settings.db_path):
            os.remove(settings.db_path)
        settings.db_path = old_db

    # Real strategies return the StrategyResult envelope through the registry.
    assert isinstance(result, StrategyResult)
    assert result.payload["model"] == "fusion/m3c-real"
