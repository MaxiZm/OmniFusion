import pytest

from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage
from omnifusion.fusion.types import Preset, PresetStage


def preset(strategy="B"):
    stage = PresetStage(max_tokens=8, timeout=5)
    return Preset(
        name="strategy-test",
        strategy=strategy,
        panel_models=["m"],
        panel=stage,
        judge_model="m",
        judge=stage,
        final_model="m",
        final=stage,
    )


def test_strategy_registry_exposes_classic_and_tool_step():
    from omnifusion.fusion.runtime.registry import registry

    assert {"B", "_tool_step"} <= set(registry.keys())


@pytest.mark.asyncio
async def test_execute_strategy_selects_tool_step_when_request_has_tools(monkeypatch):
    import omnifusion.fusion.runtime.registry as registry_mod

    calls = []

    async def fake_classic(run_id, preset, body, key_hash):
        calls.append("classic")
        return {"strategy": "classic"}

    async def fake_tool(run_id, preset, body, key_hash):
        calls.append("tool")
        return {"strategy": "tool"}

    monkeypatch.setattr(registry_mod, "_execute_classic", fake_classic)
    monkeypatch.setattr(registry_mod, "_execute_tool_step", fake_tool)

    no_tools = ChatCompletionRequest(
        model="fusion/strategy-test",
        messages=[ChatMessage(role="user", content="hi")],
    )
    with_tools = ChatCompletionRequest(
        model="fusion/strategy-test",
        messages=[ChatMessage(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "do_thing"}}],
    )

    assert await registry_mod.execute_strategy("r1", preset(), no_tools, "key") == {
        "strategy": "classic"
    }
    assert await registry_mod.execute_strategy("r2", preset(), with_tools, "key") == {
        "strategy": "tool"
    }
    assert calls == ["classic", "tool"]
