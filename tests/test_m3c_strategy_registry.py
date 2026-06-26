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


def test_strategy_contract_modules_are_present():
    from omnifusion.fusion.runtime.artifacts import ArtifactGraph
    from omnifusion.fusion.runtime.context import RunContext
    from omnifusion.fusion.runtime.strategy import FusionStrategy
    from omnifusion.fusion.strategies.classic import ClassicStrategy
    from omnifusion.fusion.strategies.conductor import ConductorStrategy
    from omnifusion.fusion.strategies.tool_step import ToolStepStrategy

    assert issubclass(ClassicStrategy, FusionStrategy)
    assert issubclass(ToolStepStrategy, FusionStrategy)
    assert issubclass(ConductorStrategy, FusionStrategy)
    assert ArtifactGraph().to_trace_metadata() == {}
    assert RunContext(
        run_id="r",
        preset=preset(),
        request=ChatCompletionRequest(
            model="fusion/strategy-test",
            messages=[ChatMessage(role="user", content="hi")],
        ),
        key_hash="key",
    ).artifacts.to_trace_metadata() == {}


@pytest.mark.asyncio
async def test_strategy_registry_dispatches_strategy_objects_with_context():
    from omnifusion.fusion.runtime.artifacts import ArtifactGraph
    from omnifusion.fusion.runtime.context import RunContext
    from omnifusion.fusion.runtime.registry import StrategyRegistry
    from omnifusion.fusion.runtime.strategy import FusionStrategy

    calls = []

    class FakeStrategy(FusionStrategy):
        def __init__(self, key):
            self.key = key

        async def execute(self, ctx: RunContext):
            assert isinstance(ctx, RunContext)
            assert isinstance(ctx.artifacts, ArtifactGraph)
            ctx.artifacts.add(self.key, {"model": ctx.request.model})
            calls.append((self.key, ctx.run_id, ctx.key_hash))
            return {"strategy": self.key, "artifacts": ctx.artifacts.to_trace_metadata()}

    registry = StrategyRegistry([FakeStrategy("B"), FakeStrategy("_tool_step")])
    no_tools = ChatCompletionRequest(
        model="fusion/strategy-test",
        messages=[ChatMessage(role="user", content="hi")],
    )
    with_tools = ChatCompletionRequest(
        model="fusion/strategy-test",
        messages=[ChatMessage(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "do_thing"}}],
    )

    assert await registry.execute("r1", preset(), no_tools, "key") == {
        "strategy": "B",
        "artifacts": {"B": {"model": "fusion/strategy-test"}},
    }
    assert await registry.execute("r2", preset(), with_tools, "key") == {
        "strategy": "_tool_step",
        "artifacts": {"_tool_step": {"model": "fusion/strategy-test"}},
    }
    assert calls == [("B", "r1", "key"), ("_tool_step", "r2", "key")]


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
