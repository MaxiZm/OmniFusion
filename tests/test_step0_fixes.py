import hashlib
import json
import logging
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage
from omnifusion.budget.ledger import (
    initialize_request_budget,
    reconcile_budget,
    reserve_budget,
)
from omnifusion.fusion.synth import run_synthesis
from omnifusion.fusion.types import JudgeAnalysis, PanelResult, Preset, PresetStage
from omnifusion.providers.pricing import PRICE_OVERRIDES, register_price_override
from omnifusion.settings import Settings, settings
from omnifusion.store.db import get_db_connection, init_db


FERNET_KEY = "U1NfdlhjdmJubWwwMTIzNDU2Nzg5MGFiY2RlZmdoaWo="


class _Delta:
    def __init__(self, content: str):
        self.content = content


class _StreamChoice:
    def __init__(self, content: str):
        self.delta = _Delta(content)


class _StreamChunk:
    def __init__(self, content: str):
        self.choices = [_StreamChoice(content)]

    def model_dump_json(self):
        return json.dumps({"choices": [{"delta": {"content": self.choices[0].delta.content}}]})


async def _stream_chunks(*contents: str):
    for content in contents:
        yield _StreamChunk(content)


def _preset(model: str = "step0-priced-model") -> Preset:
    stage = PresetStage(max_tokens=20, timeout=5)
    return Preset(
        name="step0",
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


@pytest.fixture
def isolated_db(tmp_path):
    old_db = settings.db_path
    settings.db_path = str(tmp_path / "step0.db")
    yield
    settings.db_path = old_db


async def _request_ledger(run_id: str) -> tuple[int, int]:
    async with get_db_connection() as db:
        cursor = await db.execute(
            "SELECT reserved_micro_usd, spent_micro_usd FROM budget_ledger WHERE scope='request' AND window_key=?",
            (run_id,),
        )
        return tuple(await cursor.fetchone())


async def _reservation_states(run_id: str, stage: str) -> list[str]:
    async with get_db_connection() as db:
        cursor = await db.execute(
            "SELECT state FROM budget_reservations WHERE run_id=? AND stage=? ORDER BY created_at, reservation_id",
            (run_id, stage),
        )
        return [row[0] for row in await cursor.fetchall()]


@pytest.mark.asyncio
async def test_streaming_synthesis_reconciles_once(isolated_db, monkeypatch):
    await init_db()
    model = "step0-priced-model"
    register_price_override(model, input_per_mtok=1.0, output_per_mtok=1.0)

    async def mock_stream_completion(*args, **kwargs):
        return _stream_chunks("hello ", "world")

    monkeypatch.setattr(
        "omnifusion.llm.client.llm_client.acompletion",
        mock_stream_completion,
    )

    async def seed_panel_and_judge(run_id: str) -> None:
        await initialize_request_budget(run_id, 1_000_000)
        panel_reservation = await reserve_budget(run_id, "panel/model-a", 111)
        await reconcile_budget(panel_reservation, 101)
        judge_reservation = await reserve_budget(run_id, "judge", 222)
        await reconcile_budget(judge_reservation, 202)

    async def start_stream(run_id: str, context: dict):
        request = ChatCompletionRequest(
            model="fusion/step0",
            messages=[ChatMessage(role="user", content="stream a small answer")],
            stream=True,
        )
        return await run_synthesis(
            run_id,
            _preset(model),
            request,
            [PanelResult(model=model, status="ok", content="panel answer")],
            JudgeAnalysis(consensus="ok", recommended_final_answer_plan="answer"),
            context,
        )

    close_run_id = "step0-stream-close"
    close_context = {}
    await seed_panel_and_judge(close_run_id)
    unconsumed_stream = await start_stream(close_run_id, close_context)

    reserved_before_close, spent_before_close = await _request_ledger(close_run_id)
    assert spent_before_close == 303
    assert reserved_before_close > 0
    assert await _reservation_states(close_run_id, "final") == ["reserved"]

    await unconsumed_stream.aclose()
    reserved_after_close, spent_after_close = await _request_ledger(close_run_id)
    assert reserved_after_close == 0
    assert spent_after_close > spent_before_close
    assert await _reservation_states(close_run_id, "final") == ["reconciled"]
    assert unconsumed_stream.cost_usd > 0

    run_id = "step0-stream-ledger"
    context = {}
    await seed_panel_and_judge(run_id)
    stream = await start_stream(run_id, context)

    reserved_before, spent_before = await _request_ledger(run_id)
    assert spent_before == 303
    assert reserved_before > 0
    assert await _reservation_states(run_id, "final") == ["reserved"]

    async for _chunk in stream:
        pass

    reserved_after, spent_after = await _request_ledger(run_id)
    assert reserved_after == 0
    assert spent_after > spent_before
    assert await _reservation_states(run_id, "final") == ["reconciled"]
    assert stream.cost_usd > 0

    spent_once = spent_after
    await stream.aclose()
    assert await _request_ledger(run_id) == (0, spent_once)
    PRICE_OVERRIDES.pop(model, None)


def test_circuit_breaker_uses_settings():
    from omnifusion.ratelimit.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker()
    cfg = SimpleNamespace(
        omnifusion_circuit_breaker_failure_threshold=2,
        omnifusion_circuit_breaker_cooldown_seconds=0,
    )
    breaker.configure_from_settings(cfg)

    assert breaker.failure_threshold == 2
    assert breaker.cooldown_seconds == 0
    breaker.record_failure("provider-a")
    assert breaker.allow_request("provider-a") is True
    breaker.record_failure("provider-a")
    assert breaker.allow_request("provider-a") is True
    assert breaker.allow_request("provider-a") is False


def test_circuit_breaker_half_open_admits_one_probe():
    from omnifusion.ratelimit.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=0)
    breaker.record_failure("provider-a")

    assert breaker.allow_request("provider-a") is True
    assert breaker.allow_request("provider-a") is False

    breaker.record_success("provider-a")
    assert breaker.allow_request("provider-a") is True


def test_api_key_hmac_keyed(monkeypatch):
    from omnifusion.api.auth import get_key_hash

    monkeypatch.setattr(settings, "omnifusion_secret_key", SecretStr("server-secret-one"))
    first = get_key_hash("client-key")
    bare_sha = hashlib.sha256(b"client-key").hexdigest()

    assert first != bare_sha
    assert first.startswith("hmac-sha256:")

    monkeypatch.setattr(settings, "omnifusion_secret_key", SecretStr("server-secret-two"))
    assert get_key_hash("client-key") != first


def test_session_cookie_secure_default():
    cfg = Settings(
        _env_file=None,
        omnifusion_secret_key=FERNET_KEY,
        omnifusion_admin_password="real-password",
    )
    assert cfg.omnifusion_secure_cookie is True


def test_session_rotated_on_login(isolated_db, monkeypatch):
    from omnifusion.admin import routes as admin_routes
    from omnifusion.main import app

    monkeypatch.setattr(settings, "omnifusion_secure_cookie", False)
    admin_routes._admin_hash = None

    async def seed_old_session():
        await init_db()
        async with get_db_connection() as db:
            await db.execute(
                "INSERT INTO sessions (session_id, username, csrf_token, expires_at) VALUES (?, ?, ?, ?)",
                ("old-session", "admin", "old-csrf", int(time.time()) + 3600),
            )
            await db.commit()

    import anyio

    anyio.run(seed_old_session)
    with TestClient(app) as client:
        client.cookies.set("session_id", "old-session")
        response = client.post(
            "/admin/login",
            data={"username": "admin", "password": "test-password-123"},
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.cookies.get("session_id") != "old-session"

    async def assert_rotated():
        async with get_db_connection() as db:
            old_cursor = await db.execute(
                "SELECT COUNT(*) FROM sessions WHERE session_id='old-session'"
            )
            old_count = (await old_cursor.fetchone())[0]
            all_cursor = await db.execute("SELECT COUNT(*) FROM sessions")
            total_count = (await all_cursor.fetchone())[0]
        assert old_count == 0
        assert total_count == 1

    anyio.run(assert_rotated)


def test_startup_rejects_placeholder_secrets():
    from omnifusion.settings import validate_startup_security

    with pytest.raises(ValueError, match="placeholder"):
        validate_startup_security(
            Settings(
                _env_file=None,
                omnifusion_secret_key="YOUR_FERNET_SECRET_KEY",
                omnifusion_admin_password="real-password",
            )
        )

    with pytest.raises(ValueError, match="placeholder"):
        validate_startup_security(
            Settings(
                _env_file=None,
                omnifusion_secret_key=FERNET_KEY,
                omnifusion_admin_password="YOUR_ADMIN_PASSWORD",
            )
        )


def test_logging_redacts_and_correlates_run_id():
    from omnifusion.logging_config import JSONFormatter, set_run_id
    from omnifusion.secrets.redact import redactor

    secret = "sk-step0-secret-value-1234567890"
    redactor.add_secret(secret)
    set_run_id("run-step0")
    try:
        record = logging.LogRecord(
            name="omnifusion.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="calling provider with %s",
            args=(secret,),
            exc_info=None,
        )
        redactor.filter(record)
        output = JSONFormatter().format(record)
    finally:
        set_run_id(None)
        redactor.known_secrets.discard(secret)

    payload = json.loads(output)
    assert payload["run_id"] == "run-step0"
    assert secret not in output
    assert "[REDACTED]" in output


def test_health_reports_db_and_tasks(isolated_db):
    from omnifusion.main import app

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["db"]["status"] == "ok"
    assert {"heartbeat", "jobs_sweep", "session_sweep", "reservation_sweep", "runs_sweep"}.issubset(
        payload["tasks"]
    )
