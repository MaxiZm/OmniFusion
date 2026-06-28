import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage
from omnifusion.fusion.orchestrator import run_fusion
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.store.presets import save_preset
from omnifusion.store.providers import save_provider
from omnifusion.store.runs import get_trace


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason="stop"):
        self.message = _Msg(content)
        self.finish_reason = finish_reason


class _Usage(BaseModel):
    prompt_tokens: int = 2
    completion_tokens: int = 3


class _Resp:
    def __init__(self, content, pt=2, ct=3):
        self.choices = [_Choice(content)]
        self.usage = _Usage(prompt_tokens=pt, completion_tokens=ct)


class _Delta:
    def __init__(self, content):
        self.content = content


class _ChunkChoice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content):
        self.choices = [_ChunkChoice(content)]
        self.usage = None

    def model_dump_json(self):
        return json.dumps({"choices": [{"delta": {"content": self.choices[0].delta.content}}]})


async def _stream(parts):
    for part in parts:
        yield _Chunk(part)
        await asyncio.sleep(0)


def _stage():
    return PresetStage(max_tokens=64, timeout=5)


def _preset(**overrides):
    stage = _stage()
    data = {
        "name": "of",
        "strategy": "B",
        "panel_models": ["panel-a", "panel-b"],
        "panel": stage,
        "judge_model": "judge-a",
        "judge": stage,
        "final_model": "final-a",
        "final": stage,
        "min_panel_success": 1,
    }
    data.update(overrides)
    return Preset(**data)


def _req(**overrides):
    data = {
        "model": "fusion/of",
        "messages": [ChatMessage(role="user", content="compare a and b")],
        "stream": False,
        "store": True,
    }
    data.update(overrides)
    return ChatCompletionRequest(**data)


@pytest.fixture
def db(tmp_path):
    old_db = settings.db_path
    settings.db_path = str(tmp_path / "openfusion.db")
    try:
        yield
    finally:
        if os.path.exists(settings.db_path):
            os.remove(settings.db_path)
        settings.db_path = old_db


@pytest.mark.asyncio
async def test_openfusion_alias_config_and_estimate_endpoint(db):
    from omnifusion.api.model_names import normalize_requested_model
    from omnifusion.main import app

    old_keys = settings.omnifusion_api_keys
    settings.omnifusion_api_keys = ["of-key"]
    try:
        await init_db()
        await save_preset(_preset(name="general"))
        with TestClient(app) as client:
            config = client.get("/v1/config", headers={"Authorization": "Bearer of-key"})
            estimate = client.post(
                "/api/v1/estimate",
                headers={"Authorization": "Bearer of-key"},
                json={
                    "model": "openfusion",
                    "messages": [{"role": "user", "content": "compare options"}],
                },
            )
            playground = client.get("/playground", follow_redirects=False)
    finally:
        settings.omnifusion_api_keys = old_keys

    assert normalize_requested_model("openfusion") == "fusion/general"
    assert config.status_code == 200
    assert config.json()["source"]["commit"].startswith("058035c")
    assert "providers" in config.json()
    assert estimate.status_code == 200
    assert estimate.json()["route"] == "fuse"
    assert playground.status_code in (307, 308)
    assert playground.headers["location"] == "/admin/playground"


@pytest.mark.asyncio
async def test_openfusion_template_requires_registered_models(db):
    from omnifusion.main import app

    old_keys = settings.omnifusion_api_keys
    settings.omnifusion_api_keys = ["template-key"]
    try:
        await init_db()
        await save_preset(_preset(name="general"))
        with TestClient(app) as client:
            response = client.post(
                "/v1/estimate",
                headers={"Authorization": "Bearer template-key"},
                json={
                    "model": "openfusion",
                    "messages": [{"role": "user", "content": "compare options"}],
                    "openfusion": {"preset": "quality"},
                },
            )
    finally:
        settings.omnifusion_api_keys = old_keys

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "openfusion_template_unavailable"


@pytest.mark.asyncio
async def test_plugin_judge_override_requires_registered_model(db, monkeypatch):
    import omnifusion.api.chat as chat_mod
    from omnifusion.main import app

    old_keys = settings.omnifusion_api_keys
    settings.omnifusion_api_keys = ["plugin-key"]

    async def fail_run_fusion(*args, **kwargs):
        raise AssertionError("run_fusion should not be called for invalid plugin model")

    try:
        await init_db()
        await save_preset(_preset(name="general"))
        monkeypatch.setattr(chat_mod, "run_fusion", fail_run_fusion)
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer plugin-key"},
                json={
                    "model": "openfusion",
                    "messages": [{"role": "user", "content": "hello"}],
                    "plugins": {"judge_model": "unregistered-judge"},
                },
            )
    finally:
        settings.omnifusion_api_keys = old_keys

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "plugin_model_not_registered"


@pytest.mark.asyncio
async def test_openrouter_server_web_tool_maps_to_web_grounding(db, monkeypatch):
    import omnifusion.api.chat as chat_mod
    from omnifusion.main import app

    old_keys = settings.omnifusion_api_keys
    settings.omnifusion_api_keys = ["tool-key"]
    seen = {}

    async def fake_run_fusion(run_id, preset, body, key_hash):
        seen["web_enabled"] = preset.web_enabled
        seen["tools"] = body.tools
        return {
            "id": "chatcmpl-of",
            "object": "chat.completion",
            "created": 1,
            "model": body.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    try:
        await init_db()
        await save_preset(_preset(name="general"))
        monkeypatch.setattr(chat_mod, "run_fusion", fake_run_fusion)
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer tool-key"},
                json={
                    "model": "openfusion",
                    "messages": [{"role": "user", "content": "research this"}],
                    "tools": [{"type": "openrouter:web_search"}],
                },
            )
    finally:
        settings.omnifusion_api_keys = old_keys

    assert response.status_code == 200
    assert seen == {"web_enabled": True, "tools": None}


@pytest.mark.asyncio
async def test_router_solo_uses_registered_provider_and_writes_trace(db, monkeypatch):
    import omnifusion.llm.client as client_mod

    await init_db()
    await save_provider("solo", "ollama", "", "http://localhost:1234", None, ["solo-a"])
    preset = _preset(
        router={
            "enabled": True,
            "mode": "never",
            "route_models": [{"model": "solo-a", "tier": "fast"}],
        }
    )

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        assert provider_id == "solo"
        assert model == "solo-a"
        return _Resp("solo answer", 4, 5)

    monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
    response = await run_fusion("of-solo", preset, _req(messages=[ChatMessage(role="user", content="hi")]), "kh")
    trace = await get_trace("of-solo", "kh")

    assert response["choices"][0]["message"]["content"] == "solo answer"
    assert [event.stage for event in trace.stage_events] == ["router", "router/solo"]
    assert trace.metadata["router"]["decision"] == "solo"


@pytest.mark.asyncio
async def test_vote_aggregator_returns_majority_without_judge_or_final(db, monkeypatch):
    import omnifusion.llm.client as client_mod

    await init_db()
    calls = []

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        calls.append(model)
        return _Resp("same answer" if model in {"panel-a", "panel-b"} else "unexpected")

    monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
    preset = _preset(aggregator="vote")
    response = await run_fusion("of-vote", preset, _req(), "kh")
    trace = await get_trace("of-vote", "kh")

    assert response["choices"][0]["message"]["content"] == "same answer"
    assert sorted(calls) == ["panel-a", "panel-b"]
    assert "judge-a" not in calls
    assert "final-a" not in calls
    assert trace.stage_events[-1].stage == "aggregation"
    assert trace.stage_events[-1].metadata["aggregator"] == "vote"


@pytest.mark.asyncio
async def test_ranked_aggregator_uses_short_judge_selection_without_final(db, monkeypatch):
    import omnifusion.llm.client as client_mod

    await init_db()
    calls = []

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        calls.append(model)
        if model == "panel-a":
            return _Resp("answer a")
        if model == "panel-b":
            return _Resp("answer b")
        if model == "judge-a":
            assert kwargs["max_tokens"] <= 128
            return _Resp('{"winner": 2, "reason": "more complete"}')
        raise AssertionError(model)

    monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
    response = await run_fusion("of-ranked", _preset(aggregator="ranked"), _req(), "kh")
    trace = await get_trace("of-ranked", "kh")

    assert response["choices"][0]["message"]["content"] == "answer b"
    assert calls.count("judge-a") == 1
    assert "final-a" not in calls
    assert trace.judge_analysis.consensus == "Ranked aggregator selected ANSWER_2."


@pytest.mark.asyncio
async def test_response_cache_writes_cached_trace(db, monkeypatch):
    import omnifusion.llm.client as client_mod

    await init_db()
    calls = []

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        calls.append(model)
        if model == "judge-a":
            return _Resp('{"consensus": "ok"}')
        if model == "final-a":
            return _Resp("cached answer")
        return _Resp("panel answer")

    monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
    preset = _preset(response_cache={"enabled": True, "ttl_seconds": 60, "max_entries": 8})
    request = _req(messages=[ChatMessage(role="user", content="compare cache unique")])

    first = await run_fusion("of-cache-1", preset, request, "kh")
    second = await run_fusion("of-cache-2", preset, request, "kh")
    trace = await get_trace("of-cache-2", "kh")

    assert first["choices"][0]["message"]["content"] == "cached answer"
    assert second["choices"][0]["message"]["content"] == "cached answer"
    assert calls == ["panel-a", "panel-b", "judge-a", "final-a"]
    assert trace.stage_events[0].stage == "cache"


@pytest.mark.asyncio
async def test_streaming_analysis_event_is_opt_in(db, monkeypatch):
    import omnifusion.llm.client as client_mod

    await init_db()

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        if model == "judge-a":
            return _Resp('{"consensus": "ok", "recommended_final_answer_plan": "merge"}')
        if model == "final-a" and kwargs.get("stream"):
            return _stream(["final ", "answer"])
        return _Resp("panel answer")

    monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
    result = await run_fusion(
        "of-analysis",
        _preset(analysis_emit={"enabled": True}),
        _req(stream=True, stream_options={"include_usage": True}),
        "kh",
    )
    body = ""
    async for chunk in result.body_iterator:
        body += chunk

    assert "event: analysis" in body
    assert "data: [DONE]" in body


def test_openfusion_cli_parser_exposes_commands():
    from omnifusion.openfusion_cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["ask", "--model", "openfusion", "hello"])

    assert args.command == "ask"
    assert args.model == "openfusion"
