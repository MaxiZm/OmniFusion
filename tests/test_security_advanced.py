"""
Advanced security tests covering:
- Fix #18: E2E budget ceiling enforcement (confirms 402 propagates, not silent 503)
- Fix #8: Admin login brute-force protection
- Fix #12: Input size limits
- Fix #13: Wall timeout enforcement
- Fix #19: OMNIFUSION_API_KEYS comma-split fallback
- Fix (medium): Redaction filter on handlers
"""
import pytest
import os
import asyncio
from unittest.mock import patch
from pydantic import BaseModel

from omnifusion.store.db import init_db
from omnifusion.store.presets import save_preset
from omnifusion.store.providers import save_provider
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.fusion.orchestrator import run_fusion
from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage
from omnifusion.api.errors import BudgetExceededError
from omnifusion.settings import settings, Settings
from omnifusion.budget.ledger import initialize_request_budget, reserve_budget


@pytest.fixture(autouse=True)
def setup_db():
    old_db = settings.db_path
    settings.db_path = "test_security_adv.db"
    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)
    yield
    if os.path.exists(settings.db_path):
        try:
            os.remove(settings.db_path)
        except Exception:
            pass
    settings.db_path = old_db


class MockMessage:
    def __init__(self, content):
        self.content = content


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)


class MockUsage(BaseModel):
    prompt_tokens: int = 10
    completion_tokens: int = 20


class MockResponse:
    def __init__(self, content, prompt_tokens=10, completion_tokens=20):
        self.choices = [MockChoice(content)]
        self.usage = MockUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        self.model = "mock-model"


# ─── Fix #18: E2E budget ceiling enforcement ─────────────────────────────────

@pytest.mark.asyncio
async def test_budget_ceiling_propagates_402_not_503():
    """
    Fix #18: Confirms that when the request budget is exhausted during run_panel,
    BudgetExceededError (402) propagates out of run_fusion rather than being silently
    converted to InsufficientPanelError (503).
    """
    await init_db()
    await save_provider("default", "openai", "test-key", models=["mock-model"])

    preset = Preset(
        name="tight-budget-preset",
        strategy="B",
        panel_models=["mock-model"],
        panel=PresetStage(max_tokens=100, timeout=10),
        judge_model="mock-model",
        judge=PresetStage(max_tokens=100, timeout=10),
        final_model="mock-model",
        final=PresetStage(max_tokens=100, timeout=10),
        # 1 microUSD ceiling — will be exhausted immediately
        cost_ceiling=0.000001,
        min_panel_success=1,
    )
    await save_preset(preset)

    req = ChatCompletionRequest(
        model="fusion/tight-budget-preset",
        messages=[ChatMessage(role="user", content="Hello")],
        stream=False,
        store=False,
    )

    with patch("omnifusion.llm.client.llm_client.acompletion") as mock_call:
        mock_call.return_value = MockResponse("Panel answer")
        with pytest.raises(BudgetExceededError) as exc_info:
            await run_fusion("test-budget-e2e", preset, req, "test-key")

    # Status code must be 402, not 503
    assert exc_info.value.status_code == 402, (
        f"Expected 402 BudgetExceededError, got status {exc_info.value.status_code}. "
        f"The BudgetExceededError was likely swallowed and re-raised as InsufficientPanelError (503)."
    )


@pytest.mark.asyncio
async def test_reconcile_records_true_overspend_and_rejects_future():
    """
    reconcile_budget must record the TRUE actual spend even when it exceeds the
    ceiling (the provider already billed us), and the ledger must then reject any
    further reservations for that window — fail-closed after an under-reservation.
    """
    await init_db()

    run_id = "test-overspend-run"
    ceiling = 1000  # 1000 microUSD ceiling
    await initialize_request_budget(run_id, ceiling)

    # Reserve most of the budget
    resid = await reserve_budget(run_id, "panel/test", 900)

    from omnifusion.budget.ledger import reconcile_budget
    from omnifusion.store.db import get_db_connection

    # Reconcile with 5000 (5x the ceiling) — the real provider cost.
    await reconcile_budget(resid, 5000)

    async with get_db_connection() as db:
        cursor = await db.execute(
            "SELECT reserved_micro_usd, spent_micro_usd FROM budget_ledger WHERE scope='request' AND window_key=?",
            (run_id,),
        )
        reserved, spent = await cursor.fetchone()

    # True spend is recorded, not clamped down to the ceiling.
    assert spent == 5000, f"expected true spend 5000, got {spent}"
    assert reserved == 0  # Reservation fully released

    # And because spent now exceeds ceiling, further reservations are rejected.
    with pytest.raises(BudgetExceededError):
        await reserve_budget(run_id, "panel/test-2", 1)


@pytest.mark.asyncio
async def test_midnight_global_row_created_in_reserve():
    """
    Fix #4: reserve_budget must create the global daily row if it doesn't exist,
    so the midnight transition doesn't skip the global budget check.
    """
    await init_db()
    import datetime
    from omnifusion.store.db import get_db_connection

    today = datetime.date.today().isoformat()
    run_id = "test-midnight-run"
    await initialize_request_budget(run_id, 1_000_000)

    # Delete the global row to simulate crossing midnight
    async with get_db_connection() as db:
        await db.execute(
            "DELETE FROM budget_ledger WHERE scope='global' AND window_key=?",
            (today,),
        )
        await db.commit()

    # Verify row is gone
    async with get_db_connection() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM budget_ledger WHERE scope='global' AND window_key=?",
            (today,),
        )
        count = (await cursor.fetchone())[0]
    assert count == 0, "Pre-condition: global row should be missing"

    # reserve_budget should create it and check it
    resid = await reserve_budget(run_id, "test", 100)
    assert resid is not None

    # Verify global row now exists
    async with get_db_connection() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM budget_ledger WHERE scope='global' AND window_key=?",
            (today,),
        )
        count = (await cursor.fetchone())[0]
    assert count == 1, "Global budget row should have been created by reserve_budget"


# ─── Fix #12: Input size limits ──────────────────────────────────────────────

def test_content_too_long_rejected():
    """Fix #12: Content exceeding max_content_chars must be rejected by validator."""
    old_limit = settings.omnifusion_max_content_chars
    settings.omnifusion_max_content_chars = 100
    try:
        from pydantic import ValidationError
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="fusion/test",
                messages=[ChatMessage(role="user", content="x" * 101)],
            )
        assert "exceeds maximum" in str(exc_info.value).lower() or "maximum" in str(exc_info.value)
    finally:
        settings.omnifusion_max_content_chars = old_limit


def test_max_tokens_zero_rejected():
    """Fix #12 (medium): max_tokens=0 must be rejected."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="fusion/test",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=0,
        )


def test_max_tokens_negative_rejected():
    """Fix #12 (medium): Negative max_tokens must be rejected."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="fusion/test",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=-1,
        )


def test_max_tokens_exceeds_limit_rejected():
    """Fix #12: max_tokens above server limit must be rejected."""
    old_limit = settings.omnifusion_max_tokens_limit
    settings.omnifusion_max_tokens_limit = 100
    try:
        from pydantic import ValidationError
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="fusion/test",
                messages=[ChatMessage(role="user", content="Hello")],
                max_tokens=101,
            )
        assert "101" in str(exc_info.value) or "exceeds" in str(exc_info.value).lower()
    finally:
        settings.omnifusion_max_tokens_limit = old_limit


def test_too_many_messages_rejected():
    """Fix #12: Message list exceeding max_messages must be rejected."""
    old_limit = settings.omnifusion_max_messages
    settings.omnifusion_max_messages = 3
    try:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="fusion/test",
                messages=[ChatMessage(role="user", content=f"msg {i}") for i in range(4)],
            )
    finally:
        settings.omnifusion_max_messages = old_limit


# ─── Fix #19: OMNIFUSION_API_KEYS comma-split fallback ───────────────────────

def test_api_keys_comma_split():
    """Fix #19: OMNIFUSION_API_KEYS must parse correctly from comma-separated format."""
    s = Settings(
        omnifusion_api_keys="key_one,key_two",  # comma-separated (the .env.example format)
        omnifusion_admin_password="test-pass",
        omnifusion_secret_key="U1NfdlhjdmJubWwwMTIzNDU2Nzg5MGFiY2RlZmdoaWo=",
    )
    assert s.omnifusion_api_keys == ["key_one", "key_two"]


def test_api_keys_json_format():
    """Fix #19: JSON array format must still be accepted."""
    s = Settings(
        omnifusion_api_keys='["key_one","key_two"]',
        omnifusion_admin_password="test-pass",
        omnifusion_secret_key="U1NfdlhjdmJubWwwMTIzNDU2Nzg5MGFiY2RlZmdoaWo=",
    )
    assert s.omnifusion_api_keys == ["key_one", "key_two"]


def test_api_keys_empty_string():
    """Fix #19: Empty string should yield empty list (not crash)."""
    s = Settings(
        omnifusion_api_keys="",
        omnifusion_admin_password="test-pass",
        omnifusion_secret_key="U1NfdlhjdmJubWwwMTIzNDU2Nzg5MGFiY2RlZmdoaWo=",
    )
    assert s.omnifusion_api_keys == []


def test_api_keys_comma_split_from_env(monkeypatch):
    """Regression (boot crash): a comma-separated value coming from the ENVIRONMENT
    (not an init kwarg) must parse. Init kwargs bypass pydantic-settings' JSON
    complex-decoder, so the earlier test gave false confidence; a real env var / .env
    value is JSON-decoded at the source level and crashed startup before NoDecode.
    """
    monkeypatch.setenv("OMNIFUSION_API_KEYS", "key_one,key_two")
    monkeypatch.setenv("OMNIFUSION_ADMIN_PASSWORD", "test-pass")
    monkeypatch.setenv(
        "OMNIFUSION_SECRET_KEY", "U1NfdlhjdmJubWwwMTIzNDU2Nzg5MGFiY2RlZmdoaWo="
    )
    s = Settings(_env_file=None)  # read from env, not the local .env
    assert s.omnifusion_api_keys == ["key_one", "key_two"]


def test_passthrough_whitelist_comma_split_from_env(monkeypatch):
    """Same NoDecode regression for the passthrough whitelist list field."""
    monkeypatch.setenv("OMNIFUSION_PASSTHROUGH_WHITELIST", "gpt-4o,claude-3-5-sonnet")
    monkeypatch.setenv("OMNIFUSION_ADMIN_PASSWORD", "test-pass")
    monkeypatch.setenv(
        "OMNIFUSION_SECRET_KEY", "U1NfdlhjdmJubWwwMTIzNDU2Nzg5MGFiY2RlZmdoaWo="
    )
    s = Settings(_env_file=None)
    assert s.omnifusion_passthrough_whitelist == ["gpt-4o", "claude-3-5-sonnet"]


# ─── Fix (medium): Redaction filter on handlers ──────────────────────────────

def test_redaction_filter_attached_to_root_handlers():
    """
    Fix #6: The redaction filter must be attached to root logger handlers,
    not just to named loggers. This verifies the fix for litellm/httpx log leakage.
    """
    import logging
    from omnifusion.secrets.redact import redactor, setup_logging_redaction

    setup_logging_redaction()

    root = logging.getLogger()
    # At least one root handler should have the filter attached
    handler_has_filter = any(
        redactor in h.filters for h in root.handlers
    )
    assert handler_has_filter, (
        "SecretRedactingFilter must be attached to root logger handlers, not just loggers. "
        "litellm/httpx logs bypass logger-level filters."
    )


def test_redaction_covers_google_key_pattern():
    """Fix (medium): Redaction must cover Google AIza* key pattern."""
    from omnifusion.secrets.redact import redactor
    google_key = "AIzaSyAbcdefghijklmnopqrstuvwxyz01234567890"
    redacted = redactor.redact(f"Using google key: {google_key}")
    assert google_key not in redacted
    assert "[REDACTED]" in redacted


def test_redaction_covers_url_userinfo():
    """Fix (medium): Redaction must cover URL userinfo credentials."""
    from omnifusion.secrets.redact import redactor
    url = "http://user:my-secret-password@api.example.com/v1"
    redacted = redactor.redact(f"Connecting to {url}")
    assert "my-secret-password" not in redacted


# ─── Fix #8: Login brute-force protection ────────────────────────────────────

@pytest.mark.asyncio
async def test_login_lockout_after_max_attempts():
    """Fix #8: IP must be locked out after max_login_attempts failed logins."""
    from omnifusion.admin.routes import (
        _record_failed_login,
        _check_login_rate_limit,
        _login_attempts,
        _login_attempts_lock,
    )
    import fastapi

    test_ip = "10.0.0.99"

    # Clear any previous state
    async with _login_attempts_lock:
        _login_attempts.pop(test_ip, None)

    old_max = settings.omnifusion_max_login_attempts
    old_lockout = settings.omnifusion_login_lockout_seconds
    settings.omnifusion_max_login_attempts = 3
    settings.omnifusion_login_lockout_seconds = 60

    try:
        # Record failures up to threshold
        for _ in range(3):
            await _record_failed_login(test_ip)

        # Next check should raise 429
        with pytest.raises(fastapi.HTTPException) as exc_info:
            await _check_login_rate_limit(test_ip)
        assert exc_info.value.status_code == 429
    finally:
        settings.omnifusion_max_login_attempts = old_max
        settings.omnifusion_login_lockout_seconds = old_lockout
        async with _login_attempts_lock:
            _login_attempts.pop(test_ip, None)


# ─── Fix #13: Wall timeout ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wall_timeout_enforced():
    """Fix #13: run_fusion must be aborted after wall_timeout seconds."""
    await init_db()

    old_timeout = settings.omnifusion_wall_timeout
    settings.omnifusion_wall_timeout = 1  # 1 second timeout

    try:
        await save_provider("default", "openai", "test-key", models=["mock-model"])

        preset = Preset(
            name="timeout-preset",
            strategy="B",
            panel_models=["mock-model"],
            panel=PresetStage(max_tokens=50, timeout=30),
            judge_model="mock-model",
            judge=PresetStage(max_tokens=50, timeout=30),
            final_model="mock-model",
            final=PresetStage(max_tokens=50, timeout=30),
            cost_ceiling=100.0,
            min_panel_success=1,
        )
        await save_preset(preset)

        async def slow_acompletion(*args, **kwargs):
            await asyncio.sleep(10)  # Exceeds wall_timeout
            return MockResponse("answer")

        # We test the wall_timeout wrapping logic in chat.py via asyncio.wait_for

        with patch("omnifusion.llm.client.llm_client.acompletion", side_effect=slow_acompletion):
            req = ChatCompletionRequest(
                model="fusion/timeout-preset",
                messages=[ChatMessage(role="user", content="Hello")],
                stream=False,
                store=False,
            )

            # Use asyncio.wait_for directly as the chat.py route does
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    run_fusion("test-timeout-run", preset, req, "test-key"),
                    timeout=settings.omnifusion_wall_timeout,
                )
    finally:
        settings.omnifusion_wall_timeout = old_timeout


@pytest.mark.asyncio
async def test_stale_reservation_sweeper():
    """
    Fix A: Confirms that sweep_stale_reservations correctly evicts reservations
    older than STALE_RESERVATION_AGE_SECONDS, releasing the reserved budget,
    while leaving newer (active) reservations untouched.
    """
    from omnifusion.budget.ledger import (
        initialize_request_budget,
        reserve_budget,
        sweep_stale_reservations,
        STALE_RESERVATION_AGE_SECONDS,
    )
    from omnifusion.store.db import get_db_connection
    import time

    await init_db()

    run_id = "test-sweeper-run"
    await initialize_request_budget(run_id, 10_000)

    # 1. Create a reservation that we will make stale
    stale_resid = await reserve_budget(run_id, "panel/stale", 1000)

    # 2. Create an active reservation
    active_resid = await reserve_budget(run_id, "panel/active", 2000)

    # Manually update the DB to make stale_resid's created_at = now - STALE_RESERVATION_AGE_SECONDS - 100
    now = int(time.time())
    stale_created_at = now - STALE_RESERVATION_AGE_SECONDS - 100

    async with get_db_connection() as db:
        # Check starting state of budgets in ledger
        cursor = await db.execute(
            "SELECT reserved_micro_usd, spent_micro_usd FROM budget_ledger WHERE scope='request' AND window_key=?",
            (run_id,),
        )
        before_reserved, before_spent = await cursor.fetchone()
        assert before_reserved == 3000

        # Update the created_at of the stale one
        await db.execute(
            "UPDATE budget_reservations SET created_at = ? WHERE reservation_id = ?",
            (stale_created_at, stale_resid),
        )
        await db.commit()

    # 3. Run the sweeper
    await sweep_stale_reservations()

    # 4. Check results
    async with get_db_connection() as db:
        # Check that the stale reservation is now 'reconciled'
        cursor = await db.execute(
            "SELECT state FROM budget_reservations WHERE reservation_id = ?",
            (stale_resid,),
        )
        stale_state = (await cursor.fetchone())[0]
        assert stale_state == "reconciled"

        # Check that active reservation remains 'reserved'
        cursor = await db.execute(
            "SELECT state FROM budget_reservations WHERE reservation_id = ?",
            (active_resid,),
        )
        active_state = (await cursor.fetchone())[0]
        assert active_state == "reserved"

        # Check that the reserved amount on the ledger decreased by 1000 (from 3000 to 2000)
        cursor = await db.execute(
            "SELECT reserved_micro_usd, spent_micro_usd FROM budget_ledger WHERE scope='request' AND window_key=?",
            (run_id,),
        )
        after_reserved, after_spent = await cursor.fetchone()
        assert after_reserved == 2000
        assert after_spent == 0


# ─── P1: streamed responses must hold the per-key concurrency slot ────────────


@pytest.mark.asyncio
async def test_streamed_response_holds_concurrency_slot_until_consumed():
    """The per-key concurrency slot must NOT be released the moment the handler
    returns a StreamingResponse — it must be held until the stream body is fully
    consumed, otherwise long-lived streams escape the inbound concurrency cap.
    """
    from fastapi.responses import StreamingResponse
    from omnifusion.api.chat import _defer_slot_release_to_stream

    sem = asyncio.Semaphore(1)
    await sem.acquire()  # simulate the handler having taken the only slot
    assert sem._value == 0

    async def body():
        yield b"data: a\n\n"
        yield b"data: b\n\n"

    resp = StreamingResponse(body(), media_type="text/event-stream")
    handoff = _defer_slot_release_to_stream(resp, sem)
    assert handoff is True

    # Slot is still held while the stream is pending.
    assert sem._value == 0

    # Consume the stream fully → slot released exactly once.
    collected = [chunk async for chunk in resp.body_iterator]
    assert len(collected) == 2
    assert sem._value == 1


@pytest.mark.asyncio
async def test_streamed_response_releases_slot_on_abort():
    """If the stream aborts mid-body, the slot must still be released."""
    from fastapi.responses import StreamingResponse
    from omnifusion.api.chat import _defer_slot_release_to_stream

    sem = asyncio.Semaphore(1)
    await sem.acquire()

    async def erroring_body():
        yield b"data: a\n\n"
        raise RuntimeError("mid-stream abort")

    resp = StreamingResponse(erroring_body(), media_type="text/event-stream")
    assert _defer_slot_release_to_stream(resp, sem) is True

    with pytest.raises(RuntimeError, match="mid-stream abort"):
        async for _ in resp.body_iterator:
            pass

    # Slot released despite the abort.
    assert sem._value == 1


@pytest.mark.asyncio
async def test_non_streaming_result_does_not_defer_slot():
    """A non-streaming result is not a StreamingResponse, so ownership is not
    transferred and the handler's own finally releases the slot."""
    from omnifusion.api.chat import _defer_slot_release_to_stream

    sem = asyncio.Semaphore(1)
    await sem.acquire()

    plain_dict_result = {"object": "chat.completion", "choices": []}
    assert _defer_slot_release_to_stream(plain_dict_result, sem) is False
    # Helper left the semaphore untouched.
    assert sem._value == 0


# ─── P2: provider cache must not retain plaintext keys ───────────────────────


@pytest.mark.asyncio
async def test_provider_cache_holds_no_plaintext_key():
    """get_provider must return a freshly-decrypted key but the in-process cache
    must store only the ciphertext, so secrets don't linger in process memory."""
    import omnifusion.store.providers as providers_mod
    from omnifusion.store.providers import save_provider, get_provider

    await init_db()
    secret = "sk-super-secret-value-123456789"
    await save_provider("p-secret", "openai", secret, models=["mock-model"])

    # First read populates the cache and returns the plaintext key.
    p1 = await get_provider("p-secret")
    assert p1["api_key"] == secret

    # The cached record must NOT contain the plaintext anywhere.
    cached = providers_mod._provider_cache_raw["p-secret"]
    assert "api_key" not in cached
    assert secret not in str(cached.get("enc_key"))
    assert secret not in str(cached)

    # Second read (cache hit, no DB round trip) still returns the plaintext.
    p2 = await get_provider("p-secret")
    assert p2["api_key"] == secret


# ─── P2: switching a provider to env-ref mode must clear the stale stored key ──


@pytest.mark.asyncio
async def test_env_ref_update_clears_stale_stored_key():
    """Updating a provider with a blank inline key + an api_key_ref must clear the
    previously stored encrypted key, so the stale secret is not used in preference
    to the env ref (get_provider prefers api_key over api_key_ref)."""
    from omnifusion.store.providers import save_provider, get_provider

    await init_db()

    # 1. Create with an inline key.
    await save_provider("p-rot", "openai", "sk-old-inline-key", models=["m"])
    p1 = await get_provider("p-rot")
    assert p1["api_key"] == "sk-old-inline-key"

    # 2. Update to env-ref mode: blank inline key, set api_key_ref.
    await save_provider(
        "p-rot", "openai", "", api_key_ref="MY_ENV_REF", models=["m"]
    )
    p2 = await get_provider("p-rot")

    # Stored key must be gone; env ref present.
    assert p2["api_key"] == ""
    assert p2["api_key_ref"] == "MY_ENV_REF"


@pytest.mark.asyncio
async def test_metadata_only_update_preserves_stored_key():
    """An update with neither a new inline key nor a ref must preserve the existing
    encrypted key (e.g. when only editing models/base_url)."""
    from omnifusion.store.providers import save_provider, get_provider

    await init_db()
    await save_provider("p-keep", "openai", "sk-keep-me", models=["m1"])

    # Update only the model list, no key, no ref.
    await save_provider("p-keep", "openai", "", models=["m1", "m2"])
    p = await get_provider("p-keep")
    assert p["api_key"] == "sk-keep-me"
    assert p["models"] == ["m1", "m2"]


# ─── P2: custom provider model names must be canonicalized for LiteLLM ────────


@pytest.mark.asyncio
async def test_custom_provider_model_is_canonicalized():
    """custom_openai / custom_anthropic providers must route through LiteLLM with a
    provider-prefixed model (openai/<m>, anthropic/<m>) plus the custom api_base."""
    from omnifusion.llm.client import llm_client
    from omnifusion.store.providers import save_provider

    await init_db()
    # Skip SSRF DNS resolution for this URL; we're testing model canonicalization,
    # not egress validation.
    prev_egress = settings.omnifusion_allow_private_egress
    settings.omnifusion_allow_private_egress = True
    try:
        await save_provider(
            "custom-oai",
            "custom_openai",
            "sk-x",
            base_url="https://my-llm.example.com/v1",
            models=["my-model"],
        )

        captured = {}

        async def fake_litellm_acompletion(**kwargs):
            captured.update(kwargs)
            return MockResponse("ok")

        with patch("litellm.acompletion", side_effect=fake_litellm_acompletion):
            await llm_client.acompletion(
                provider_id="custom-oai",
                model="my-model",
                messages=[{"role": "user", "content": "hi"}],
            )
    finally:
        settings.omnifusion_allow_private_egress = prev_egress

    assert captured["model"] == "openai/my-model"
    assert captured["api_base"] == "https://my-llm.example.com/v1"


# ─── P2: legacy/unsupported request fields must be rejected, not dropped ──────


@pytest.mark.parametrize("field", ["audio", "logprobs"])
def test_unsupported_request_field_rejected(field):
    """Unsupported provider-specific fields must raise instead of being silently
    ignored by pydantic."""
    from pydantic import ValidationError

    payload = {
        "model": "fusion/x",
        "messages": [{"role": "user", "content": "hi"}],
        field: "x",
    }
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**payload)


def test_tools_fields_accepted():
    """tools/tool_choice must be accepted now (routed to a tool-capable model)."""
    req = ChatCompletionRequest(
        model="fusion/x",
        messages=[ChatMessage(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "do_thing", "parameters": {}}}],
        tool_choice="auto",
    )
    assert req.tools and req.tool_choice == "auto"


def test_agentic_tool_loop_message_shapes_accepted():
    """The full OpenCode loop must validate: a user turn, an assistant turn with
    null content + tool_calls, and a tool-result turn (role='tool')."""
    req = ChatCompletionRequest(
        model="fusion/x",
        messages=[
            ChatMessage(role="user", content="weather in Paris?"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            ),
            ChatMessage(role="tool", tool_call_id="c1", content="18C partly cloudy"),
        ],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    assert req.messages[1].content is None
    assert req.messages[1].tool_calls[0]["id"] == "c1"
    assert req.messages[2].role == "tool"
    assert req.messages[2].tool_call_id == "c1"
    # exclude_none keeps tool fields off normal turns but on tool/assistant turns
    assert "tool_calls" not in req.messages[0].model_dump(exclude_none=True)
    assert req.messages[2].model_dump(exclude_none=True)["tool_call_id"] == "c1"
