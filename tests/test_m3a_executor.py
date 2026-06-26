import pytest


class Delta:
    def __init__(self, content):
        self.content = content


class Choice:
    def __init__(self, content):
        self.delta = Delta(content)


class Chunk:
    def __init__(self, content):
        self.choices = [Choice(content)]


async def fake_stream(chunks, fail=False):
    for chunk in chunks:
        yield Chunk(chunk)
    if fail:
        raise RuntimeError("upstream exploded")


@pytest.mark.asyncio
async def test_budgeted_executor_stream_reconciles_once_on_normal_completion(monkeypatch):
    from omnifusion.fusion.runtime.executor import BudgetedExecutor
    import omnifusion.fusion.runtime.executor as executor_mod

    reconciles = []

    async def fake_reserve(run_id, stage, amount):
        return "reservation-normal"

    async def fake_reconcile(reservation_id, amount):
        reconciles.append((reservation_id, amount))

    async def fake_completion(**kwargs):
        return fake_stream(["hello", " world"])

    monkeypatch.setattr(executor_mod, "reserve_budget", fake_reserve)
    monkeypatch.setattr(executor_mod, "reconcile_budget", fake_reconcile)
    monkeypatch.setattr(executor_mod.llm_client, "acompletion", fake_completion)

    stream = await BudgetedExecutor("run-normal").stream(
        "final",
        provider_id="default",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=20,
    )
    async for _ in stream:
        pass

    assert len(reconciles) == 1
    assert reconciles[0][0] == "reservation-normal"
    assert reconciles[0][1] > 0


@pytest.mark.asyncio
async def test_budgeted_executor_stream_reconciles_once_on_error(monkeypatch):
    from omnifusion.fusion.runtime.executor import BudgetedExecutor
    import omnifusion.fusion.runtime.executor as executor_mod

    reconciles = []

    async def fake_reserve(run_id, stage, amount):
        return "reservation-error"

    async def fake_reconcile(reservation_id, amount):
        reconciles.append((reservation_id, amount))

    async def fake_completion(**kwargs):
        return fake_stream(["partial"], fail=True)

    monkeypatch.setattr(executor_mod, "reserve_budget", fake_reserve)
    monkeypatch.setattr(executor_mod, "reconcile_budget", fake_reconcile)
    monkeypatch.setattr(executor_mod.llm_client, "acompletion", fake_completion)

    stream = await BudgetedExecutor("run-error").stream(
        "final",
        provider_id="default",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=20,
    )
    with pytest.raises(RuntimeError):
        async for _ in stream:
            pass

    assert len(reconciles) == 1
    assert reconciles[0][1] > 0


@pytest.mark.asyncio
async def test_budgeted_executor_stream_reconciles_once_on_close(monkeypatch):
    from omnifusion.fusion.runtime.executor import BudgetedExecutor
    import omnifusion.fusion.runtime.executor as executor_mod

    reconciles = []

    async def fake_reserve(run_id, stage, amount):
        return "reservation-close"

    async def fake_reconcile(reservation_id, amount):
        reconciles.append((reservation_id, amount))

    async def fake_completion(**kwargs):
        return fake_stream(["partial", "ignored"])

    monkeypatch.setattr(executor_mod, "reserve_budget", fake_reserve)
    monkeypatch.setattr(executor_mod, "reconcile_budget", fake_reconcile)
    monkeypatch.setattr(executor_mod.llm_client, "acompletion", fake_completion)

    stream = await BudgetedExecutor("run-close").stream(
        "final",
        provider_id="default",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=20,
    )
    await stream.__anext__()
    await stream.aclose()
    await stream.aclose()

    assert len(reconciles) == 1
    assert reconciles[0][1] > 0


def test_tool_orchestrator_owns_no_direct_reconcile_shield():
    """M3a single-shield invariant: tool orchestration must route every model call
    through BudgetedExecutor, never run its own reserve/reconcile path."""
    import omnifusion.fusion.tool_orchestrator as tool_mod

    # The direct budget-ledger and llm_client handles are no longer imported into
    # the tool orchestrator namespace — only the executor owns reconciliation.
    assert not hasattr(tool_mod, "reserve_budget")
    assert not hasattr(tool_mod, "reconcile_budget")
    assert not hasattr(tool_mod, "llm_client")
    assert hasattr(tool_mod, "BudgetedExecutor")

    import inspect

    source = inspect.getsource(tool_mod)
    # No stray reconcile_budget / reserve_budget calls survive in the source.
    assert "reconcile_budget(" not in source
    assert "reserve_budget(" not in source
