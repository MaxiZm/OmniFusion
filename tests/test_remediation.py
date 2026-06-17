import pytest
import os
import asyncio
from omnifusion.providers.validation import validate_base_url
from omnifusion.api.errors import ConfigurationError
from omnifusion.store.db import init_db, get_db_connection
from omnifusion.store.providers import save_provider, get_provider
from omnifusion.settings import settings
from omnifusion.ratelimit.limiter import RateLimiter
from omnifusion.budget.ledger import initialize_request_budget
from omnifusion.fusion.panel import run_panelist
from omnifusion.fusion.types import Preset, PresetStage


def test_ssrf_ipv6_validation():
    # Loopback IPv6
    with pytest.raises(ConfigurationError):
        validate_base_url("http://[::1]:8080/v1", "openai")

    # Private/Link-local IPv6
    with pytest.raises(ConfigurationError):
        validate_base_url("http://[fe80::1]/v1", "openai")

    # AWS metadata IPv6
    with pytest.raises(ConfigurationError):
        validate_base_url("http://[fd00:ec2::254]/v1", "openai")


@pytest.mark.asyncio
async def test_key_erasure_prevention():
    old_db = settings.db_path
    settings.db_path = "test_remediation_keys.db"
    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)

    await init_db()
    try:
        # Save initially with a key
        await save_provider("test-p-remed", "openai", "my-secret-api-key")
        prov = await get_provider("test-p-remed")
        assert prov["api_key"] == "my-secret-api-key"

        # Update without specifying a key (plain_key = "")
        await save_provider("test-p-remed", "openai", "")
        prov_after = await get_provider("test-p-remed")
        # Assert key is preserved and NOT erased
        assert prov_after["api_key"] == "my-secret-api-key"
    finally:
        if os.path.exists(settings.db_path):
            try:
                os.remove(settings.db_path)
            except Exception:
                pass
        settings.db_path = old_db


@pytest.mark.asyncio
async def test_rate_limiter_acquire_cancellation():
    limiter = RateLimiter()
    # Mock global_semaphore to have 0 capacity to force wait
    for _ in range(50):
        await limiter.global_semaphore.acquire()

    # Get provider limiter to verify initial semaphore count
    sem, _ = await limiter.get_provider_limiter("test-provider-cxl")
    assert sem._value == 10

    async def try_acquire():
        await limiter.acquire("test-provider-cxl")

    # Start acquire task. It will wait on global semaphore since it is exhausted.
    task = asyncio.create_task(try_acquire())
    await asyncio.sleep(0.1)

    # The provider semaphore should have been acquired (value decreased to 9)
    assert sem._value == 9

    # Cancel the task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Verify that the provider semaphore was released and returned to 10
    assert sem._value == 10


@pytest.mark.asyncio
async def test_budget_cancellation_shielding():
    old_db = settings.db_path
    settings.db_path = "test_remediation_budget.db"
    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)

    await init_db()
    try:
        run_id = "cancel-test-run"
        preset = Preset(
            name="test-preset",
            panel_models=["mock-model"],
            panel=PresetStage(max_tokens=100, timeout=10),
            judge_model="mock-model",
            judge=PresetStage(max_tokens=100, timeout=10),
            final_model="mock-model",
            final=PresetStage(max_tokens=100, timeout=10),
        )

        await initialize_request_budget(run_id, 10000)

        # Run panelist and cancel immediately
        task = asyncio.create_task(
            run_panelist(
                run_id,
                "mock-model",
                preset,
                [{"role": "user", "content": "hello"}],
            )
        )
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Allow the shielded task to complete its database writes
        await asyncio.sleep(0.1)

        # Verify that the reserved budget on the ledger is 0 (fully reconciled/released)
        async with get_db_connection() as db:
            cursor = await db.execute(
                "SELECT reserved_micro_usd FROM budget_ledger WHERE scope='request' AND window_key=?",
                (run_id,),
            )
            reserved = (await cursor.fetchone())[0]
            assert reserved == 0

    finally:
        if os.path.exists(settings.db_path):
            try:
                os.remove(settings.db_path)
            except Exception:
                pass
        settings.db_path = old_db
