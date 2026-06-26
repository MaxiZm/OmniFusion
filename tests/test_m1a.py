import asyncio
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_rate_limiter_acquire_returns_idempotent_slot():
    from omnifusion.ratelimit.limiter import RateLimiter

    limiter = RateLimiter()
    limiter.global_semaphore = asyncio.Semaphore(1)
    provider_sem, _bucket = await limiter.get_provider_limiter("provider-a")
    provider_sem._value = 1

    slot = await limiter.acquire("provider-a")
    assert provider_sem._value == 0
    assert limiter.global_semaphore._value == 0

    slot.release()
    assert provider_sem._value == 1
    assert limiter.global_semaphore._value == 1

    slot.release()
    assert provider_sem._value == 1
    assert limiter.global_semaphore._value == 1


@pytest.mark.asyncio
async def test_rate_limiter_cancel_while_waiting_for_global_releases_provider_slot():
    from omnifusion.ratelimit.limiter import RateLimiter

    limiter = RateLimiter()
    limiter.global_semaphore = asyncio.Semaphore(0)
    provider_sem, _bucket = await limiter.get_provider_limiter("provider-a")
    provider_sem._value = 1

    task = asyncio.create_task(limiter.acquire("provider-a"))
    for _ in range(100):
        if provider_sem._value == 0:
            break
        await asyncio.sleep(0)

    assert provider_sem._value == 0
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider_sem._value == 1
    assert limiter.global_semaphore._value == 0


@pytest.mark.asyncio
async def test_streaming_response_wrapper_releases_slot_once():
    from omnifusion.llm.client import StreamingResponseWrapper

    class FakeSlot:
        def __init__(self):
            self.release_count = 0

        def release(self):
            self.release_count += 1

    class FakeStream:
        def __init__(self):
            self._items = iter(["chunk"])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._items)
            except StopIteration:
                raise StopAsyncIteration

    slot = FakeSlot()
    wrapper = StreamingResponseWrapper(FakeStream(), slot, chunk_timeout=None)

    chunks = []
    async for chunk in wrapper:
        chunks.append(chunk)

    assert chunks == ["chunk"]
    assert slot.release_count == 1
    wrapper.release()
    assert slot.release_count == 1


@pytest.mark.asyncio
async def test_llm_retry_releases_each_acquired_slot(monkeypatch):
    import omnifusion.llm.client as client_mod

    class RateLimitError(Exception):
        status_code = 429

    class FakeSlot:
        def __init__(self):
            self.release_count = 0

        def release(self):
            self.release_count += 1

    class FakeLimiter:
        def __init__(self):
            self.slots = []

        async def acquire(self, provider_id):
            slot = FakeSlot()
            self.slots.append((provider_id, slot))
            return slot

    class FakeCircuitBreaker:
        def allow_request(self, provider_id):
            return True

        def record_success(self, provider_id):
            pass

        def record_failure(self, provider_id):
            pass

    calls = 0

    async def fake_acompletion(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimitError("429 rate limit")
        return SimpleNamespace(choices=[])

    async def no_sleep(_seconds):
        return None

    async def no_provider(_model):
        return None

    limiter = FakeLimiter()
    monkeypatch.setattr(client_mod, "rate_limiter", limiter)
    monkeypatch.setattr(client_mod, "circuit_breaker", FakeCircuitBreaker())
    monkeypatch.setattr(client_mod.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(client_mod.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(client_mod, "resolve_provider_for_model", no_provider)

    result = await client_mod.LLMClient.acompletion(
        provider_id="default",
        model="mock-model",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result.choices == []
    assert calls == 2
    assert [provider_id for provider_id, _slot in limiter.slots] == ["default", "default"]
    assert [slot.release_count for _provider_id, slot in limiter.slots] == [1, 1]


def test_pyproject_declares_build_script_and_package_data():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"
    assert pyproject["project"]["scripts"]["omnifusion"] == "omnifusion.cli:main"

    package_data = pyproject["tool"]["setuptools"]["package-data"]["omnifusion"]
    assert "web/templates/*.html" in package_data
    assert "fusion/prompts/*.j2" in package_data


def test_admin_template_directory_uses_package_resources():
    from omnifusion.admin import routes

    template_dir = routes.template_directory()
    assert template_dir.is_absolute()
    assert (template_dir / "login.html").exists()


def test_dockerfile_installs_dependencies_before_project():
    dockerfile = Path("deploy/Dockerfile").read_text()
    assert "RUN uv sync --locked --no-dev --no-install-project" in dockerfile
    assert "COPY . ." in dockerfile
    dependency_sync_index = dockerfile.index("RUN uv sync --locked --no-dev --no-install-project")
    copy_source_index = dockerfile.index("COPY . .")
    project_sync_index = dockerfile.index("RUN uv sync --locked --no-dev", copy_source_index)
    assert dependency_sync_index < copy_source_index < project_sync_index
    assert "omnifusion.main:app" in dockerfile


class ReplayDelta:
    def __init__(self, content: str):
        self.content = content


class ReplayChoice:
    def __init__(self, content: str):
        self.delta = ReplayDelta(content)


class ReplayChunk:
    def __init__(self, content: str):
        self.choices = [ReplayChoice(content)]


async def replay_stream(chunks: list[str]):
    for chunk in chunks:
        yield ReplayChunk(chunk)


@pytest.mark.asyncio
async def test_streaming_cost_replay_reconciles_nonzero_final_stage(tmp_path, monkeypatch):
    from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage
    from omnifusion.budget.ledger import initialize_request_budget
    from omnifusion.fusion.synth import run_synthesis
    from omnifusion.fusion.types import JudgeAnalysis, PanelResult, Preset, PresetStage
    from omnifusion.providers.pricing import PRICE_OVERRIDES, register_price_override
    from omnifusion.settings import settings
    from omnifusion.store.db import get_db_connection, init_db

    fixture_path = Path(__file__).parent / "fixtures" / "replay" / "streaming_synthesis_cost.json"
    replay = json.loads(fixture_path.read_text())
    assert replay["tier"] == "B-replay"

    old_db = settings.db_path
    settings.db_path = str(tmp_path / "m1a-replay.db")
    model = replay["model"]
    register_price_override(
        model,
        input_per_mtok=replay["pricing"]["input_per_mtok"],
        output_per_mtok=replay["pricing"]["output_per_mtok"],
    )

    async def replay_completion(*args, **kwargs):
        return replay_stream(replay["stream_chunks"])

    monkeypatch.setattr(
        "omnifusion.llm.client.llm_client.acompletion",
        replay_completion,
    )

    try:
        await init_db()
        run_id = "m1a-replay-stream"
        await initialize_request_budget(run_id, 1_000_000)

        stage = PresetStage(max_tokens=replay["request"]["max_tokens"], timeout=5)
        preset = Preset(
            name="m1a-replay",
            strategy="B",
            panel_models=[model],
            panel=stage,
            judge_model=model,
            judge=stage,
            final_model=model,
            final=stage,
            cost_ceiling=1.0,
            min_panel_success=1,
        )
        request = ChatCompletionRequest(
            model="fusion/m1a-replay",
            messages=[ChatMessage(**message) for message in replay["request"]["messages"]],
            stream=True,
        )
        context = {}

        stream = await run_synthesis(
            run_id,
            preset,
            request,
            [
                PanelResult(
                    model=replay["panel"]["model"],
                    status="ok",
                    content=replay["panel"]["content"],
                )
            ],
            JudgeAnalysis(**replay["judge"]),
            context,
        )
        async for _chunk in stream:
            pass

        async with get_db_connection() as db:
            cursor = await db.execute(
                "SELECT reserved_micro_usd, spent_micro_usd FROM budget_ledger WHERE scope='request' AND window_key=?",
                (run_id,),
            )
            reserved, spent = await cursor.fetchone()
            reservation_cursor = await db.execute(
                "SELECT state FROM budget_reservations WHERE run_id=? AND stage='final'",
                (run_id,),
            )
            reservation_state = (await reservation_cursor.fetchone())[0]

        assert reserved == 0
        assert spent > 0
        assert reservation_state == "reconciled"
        assert context["cost_usd"] > 0
    finally:
        PRICE_OVERRIDES.pop(model, None)
        settings.db_path = old_db
