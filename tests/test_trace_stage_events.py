"""FusionTrace.stage_events across non-stream, stream, web-grounded, degraded, and
budget-exceeded runs, plus the admin trace-timeline render."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, SecretStr
from unittest.mock import patch

import omnifusion.admin.routes as admin_routes
from omnifusion.api.errors import BudgetExceededError
from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage
from omnifusion.fusion.orchestrator import run_fusion
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.store.runs import get_trace


# ── LiteLLM-shaped mocks ──────────────────────────────────────────────────────


class _Usage(BaseModel):
    prompt_tokens: int = 10
    completion_tokens: int = 20


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content, pt=10, ct=20):
        self.choices = [_Choice(content)]
        self.usage = _Usage(prompt_tokens=pt, completion_tokens=ct)


class _Delta:
    def __init__(self, content):
        self.content = content


class _ChunkChoice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _StreamChunk:
    def __init__(self, content):
        self.content = content
        self.choices = [_ChunkChoice(content)]
        self.usage = None

    def model_dump_json(self):
        return '{"choices": [{"delta": {"content": "%s"}}]}' % self.content


async def _stream(chunks):
    for c in chunks:
        yield _StreamChunk(c)
        await asyncio.sleep(0)


def _side_effect(judge_consensus="agreed"):
    def side_effect(provider_id, model, messages, **kwargs):
        if model in ("panel-a", "panel-b"):
            return _Resp(f"Answer {model}", 10, 10)
        if model == "judge-model":
            return _Resp(
                '{"consensus": "%s", "recommended_final_answer_plan": "plan"}'
                % judge_consensus,
                20,
                20,
            )
        if model == "final-model":
            if kwargs.get("stream"):
                return _stream(["Synthesized", " answer"])
            return _Resp("Synthesized answer", 30, 30)
        raise ValueError(f"unexpected model {model}")

    return side_effect


def _preset(cost_ceiling=1.0, web_enabled=False):
    return Preset(
        name="general",
        strategy="B",
        panel_models=["panel-a", "panel-b"],
        panel=PresetStage(max_tokens=100, timeout=10),
        judge_model="judge-model",
        judge=PresetStage(max_tokens=100, timeout=10),
        final_model="final-model",
        final=PresetStage(max_tokens=200, timeout=20),
        cost_ceiling=cost_ceiling,
        min_panel_success=1,
        web_enabled=web_enabled,
    )


def _req(stream=False):
    return ChatCompletionRequest(
        model="fusion/general",
        messages=[ChatMessage(role="user", content="hello")],
        stream=stream,
        store=True,
    )


@pytest.fixture
def trace_db(tmp_path):
    old_db = settings.db_path
    settings.db_path = str(tmp_path / "stage-events.db")
    try:
        yield
    finally:
        settings.db_path = old_db


@pytest.mark.asyncio
@patch("omnifusion.llm.client.llm_client.acompletion")
async def test_non_stream_stage_events(mock_acompletion, trace_db):
    await init_db()
    mock_acompletion.side_effect = _side_effect()

    await run_fusion("run-ns", _preset(), _req(), "kh")
    trace = await get_trace("run-ns")

    stages = [e.stage for e in trace.stage_events]
    assert stages.count("panel") == 2
    assert "judge" in stages
    assert "synthesis" in stages

    panel = next(e for e in trace.stage_events if e.stage == "panel")
    assert panel.role == "panel"
    assert panel.model in ("panel-a", "panel-b")
    assert panel.status == "ok"
    assert panel.tokens == {"prompt": 10, "completion": 10}

    synth = next(e for e in trace.stage_events if e.stage == "synthesis")
    assert synth.role == "final"
    assert synth.model == "final-model"
    assert synth.status == "ok"


@pytest.mark.asyncio
@patch("omnifusion.llm.client.llm_client.acompletion")
async def test_stream_stage_events(mock_acompletion, trace_db):
    await init_db()
    mock_acompletion.side_effect = _side_effect()

    result = await run_fusion("run-st", _preset(), _req(stream=True), "kh")
    async for _ in result.body_iterator:  # drives the finally → save_trace
        pass

    trace = await get_trace("run-st")
    assert trace.final_answer == "Synthesized answer"
    synth = next(e for e in trace.stage_events if e.stage == "synthesis")
    assert synth.status == "ok"
    assert [e.stage for e in trace.stage_events].count("panel") == 2


@pytest.mark.asyncio
@patch("omnifusion.llm.client.llm_client.acompletion")
async def test_web_grounded_stage_event(mock_acompletion, trace_db, monkeypatch):
    await init_db()
    mock_acompletion.side_effect = _side_effect()

    class _Ctx:
        sources = [{"url": "https://example.com/a", "title": "A"}]
        has_grounding = True
        grounding_text = "GROUNDED CONTEXT"

    async def _fake_gather(run_id, query):
        return _Ctx()

    monkeypatch.setattr(
        "omnifusion.fusion.web_grounding.gather_web_context", _fake_gather
    )

    await run_fusion("run-web", _preset(web_enabled=True), _req(), "kh")
    trace = await get_trace("run-web")

    web = next((e for e in trace.stage_events if e.stage == "web"), None)
    assert web is not None
    assert web.metadata["source_count"] == 1
    assert "https://example.com/a" in web.metadata["sources"]
    # The web event leads the timeline.
    assert trace.stage_events[0].stage == "web"


@pytest.mark.asyncio
@patch("omnifusion.llm.client.llm_client.acompletion")
async def test_degraded_run_marks_judge(mock_acompletion, trace_db):
    await init_db()
    mock_acompletion.side_effect = _side_effect(judge_consensus="panel degraded")

    await run_fusion("run-deg", _preset(), _req(), "kh")
    trace = await get_trace("run-deg")

    assert trace.degraded is True
    judge = next(e for e in trace.stage_events if e.stage == "judge")
    assert judge.status == "degraded"


@pytest.mark.asyncio
@patch("omnifusion.llm.client.llm_client.acompletion")
async def test_budget_exceeded_still_records_trace(mock_acompletion, trace_db):
    await init_db()
    mock_acompletion.side_effect = _side_effect()

    # A 1-microdollar request ceiling can't cover even one panel reservation.
    with pytest.raises(BudgetExceededError):
        await run_fusion("run-budget", _preset(cost_ceiling=0.000001), _req(), "kh")

    trace = await get_trace("run-budget")
    assert trace is not None
    assert trace.degraded is True
    assert isinstance(trace.stage_events, list)


@pytest.mark.asyncio
async def test_old_trace_without_stage_events_still_validates():
    # A trace JSON persisted before stage_events existed must still load.
    from omnifusion.fusion.types import FusionTrace

    legacy = '{"run_id": "old", "preset": "general", "cost_usd": 0.1, "wall_ms": 5, "panel_results": []}'
    trace = FusionTrace.model_validate_json(legacy)
    assert trace.stage_events == []


# ── Admin trace-timeline render ───────────────────────────────────────────────


@pytest.fixture
def admin_client(tmp_path):
    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    old_secure = settings.omnifusion_secure_cookie
    old_pw = settings.omnifusion_admin_password
    settings.db_path = str(tmp_path / "admin-trace.db")
    settings.omnifusion_api_keys = ["live-key"]
    settings.omnifusion_secure_cookie = False
    settings.omnifusion_admin_password = SecretStr("test-password-123")
    admin_routes._admin_hash = None
    try:
        from omnifusion.main import app

        yield TestClient(app)
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys
        settings.omnifusion_secure_cookie = old_secure
        settings.omnifusion_admin_password = old_pw
        admin_routes._admin_hash = None


@pytest.mark.asyncio
async def test_admin_trace_timeline_renders(admin_client):
    await init_db()
    with patch(
        "omnifusion.llm.client.llm_client.acompletion", side_effect=_side_effect()
    ):
        await run_fusion("run-render", _preset(), _req(), "kh")

    client = admin_client
    res = client.post(
        "/admin/login",
        data={"username": "admin", "password": "test-password-123"},
        follow_redirects=False,
    )
    assert res.status_code == 303
    client.cookies.update(res.cookies)

    page = client.get("/admin/runs/run-render/trace")
    assert page.status_code == 200
    # Timeline rendered with stage names and the JSON toggle.
    assert "synthesis" in page.text
    assert "Raw JSON" in page.text
