"""
Unit tests for per-step fusion-with-tools (fusion/tool_orchestrator.py).

The full agentic loop (panel proposes actions -> judge picks best -> emit tool_call
or fused final answer) is verified end-to-end against a live provider; these tests
pin the pure orchestration logic that turns proposals into OpenAI-shaped output.
"""
import json
import pytest

from omnifusion.fusion.tool_orchestrator import (
    _normalize_tool_calls,
    _describe_proposal,
    _decide_next_step,
    _sanitize_judge_tool_calls,
    _tool_names,
)
from omnifusion.fusion.runtime.response import ResponseShaper
from omnifusion.fusion.runtime.streaming import StreamingAdapter
from omnifusion.fusion.types import Preset, PresetStage


class _FakeFn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    """Mimics a litellm tool_call object exposing model_dump()."""

    def __init__(self, id, name, arguments):
        self._d = {
            "id": id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    def model_dump(self):
        return self._d


def test_normalize_tool_calls_from_objects():
    out = _normalize_tool_calls([_FakeToolCall("c1", "search", '{"q":"x"}')])
    assert out == [
        {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"q":"x"}'}}
    ]


def test_normalize_tool_calls_from_dicts_and_missing_id():
    out = _normalize_tool_calls(
        [{"type": "function", "function": {"name": "f", "arguments": None}}]
    )
    assert out[0]["function"]["name"] == "f"
    assert out[0]["function"]["arguments"] == "{}"  # defaulted
    assert out[0]["id"]  # synthesized


def test_normalize_none():
    assert _normalize_tool_calls(None) is None
    assert _normalize_tool_calls([]) is None


def test_describe_proposal_tool_vs_text():
    tool_p = {"tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]}
    assert "TOOL CALL" in _describe_proposal(0, tool_p)
    assert "get_weather" in _describe_proposal(0, tool_p)

    text_p = {"content": "The answer is 42."}
    assert "FINAL ANSWER" in _describe_proposal(1, text_p)
    assert "42" in _describe_proposal(1, text_p)


@pytest.mark.asyncio
async def test_decide_short_circuits_to_final_when_no_tool_proposals():
    """If no panelist proposed a tool call, the step is final — no judge LLM call."""
    preset = Preset(
        name="p", strategy="B", panel_models=["m"],
        panel=PresetStage(max_tokens=10, timeout=10),
        judge_model="m", judge=PresetStage(max_tokens=10, timeout=10),
        final_model="m", final=PresetStage(max_tokens=10, timeout=10),
    )
    proposals = [{"content": "answer A", "tool_calls": None}, {"content": "answer B", "tool_calls": None}]
    decision = await _decide_next_step("run-x", preset, [{"role": "user", "content": "hi"}], proposals)
    assert decision["decision"] == "final"
    assert decision["best_index"] == 0
    # No judge LLM call happened, so no cost/tokens are attributed.
    assert decision["cost"] == 0.0
    assert decision["prompt_tokens"] == 0
    assert decision["completion_tokens"] == 0


def test_tool_call_response_dict_shape():
    tcs = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    d = ResponseShaper.tool_call_completion(
        model="fusion/draco", tool_calls=tcs, usage={"prompt_tokens": 1}
    )
    assert d["model"] == "fusion/draco"
    assert d["choices"][0]["finish_reason"] == "tool_calls"
    assert d["choices"][0]["message"]["tool_calls"] == tcs
    assert d["choices"][0]["message"]["content"] is None


def test_tool_call_sse_is_valid_openai_stream():
    tcs = [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]
    events = list(StreamingAdapter("fusion/draco").tool_call_sse(tcs))
    blob = "".join(events)
    assert "data: [DONE]" in blob
    # A chunk must carry the tool_call and a terminal finish_reason.
    payloads = [json.loads(e[len("data: "):]) for e in events if e.startswith("data: {")]
    assert any(
        p["choices"][0]["delta"].get("tool_calls")
        and p["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"
        for p in payloads
    )
    assert any(p["choices"][0]["finish_reason"] == "tool_calls" for p in payloads)


def test_tool_call_sse_emits_usage_chunk_when_requested():
    """[P2] A streamed tool-call turn emits a terminal usage chunk when requested."""
    tcs = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    events = list(StreamingAdapter("fusion/x").tool_call_sse(tcs, usage=(11, 4)))
    blob = "".join(events)
    payloads = [json.loads(e[len("data: "):]) for e in events if e.startswith("data: {")]
    usage_payloads = [p for p in payloads if p.get("usage")]
    assert usage_payloads, "no usage chunk emitted"
    assert usage_payloads[-1]["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
    }
    assert blob.rstrip().endswith("data: [DONE]")


def test_tool_call_sse_omits_usage_by_default():
    tcs = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    events = list(StreamingAdapter("fusion/x").tool_call_sse(tcs))
    assert not any('"usage"' in e for e in events)


# --- Judge-authored tool-call sanitizer -------------------------------------

_VALID = {"get_weather", "search"}


def test_tool_names_extracts_function_names_only():
    tools = [
        {"type": "function", "function": {"name": "get_weather"}},
        {"type": "openrouter:web_search"},  # server tool: no function name
        "not-a-dict",
    ]
    assert _tool_names(tools) == {"get_weather"}
    assert _tool_names(None) == set()


def test_sanitize_serializes_dict_arguments_and_synthesizes_id():
    """The judge often emits arguments as a JSON object; we serialize to a string."""
    calls = [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]
    out = _sanitize_judge_tool_calls(calls, _VALID, None)
    assert out is not None and len(out) == 1
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "get_weather"
    assert json.loads(out[0]["function"]["arguments"]) == {"city": "Paris"}
    assert out[0]["id"]  # synthesized when absent


def test_sanitize_keeps_string_arguments_and_existing_id():
    calls = [{"id": "c9", "type": "function", "function": {"name": "search", "arguments": '{"q":"x"}'}}]
    out = _sanitize_judge_tool_calls(calls, _VALID, None)
    assert out[0]["id"] == "c9"
    assert out[0]["function"]["arguments"] == '{"q":"x"}'


def test_sanitize_defaults_missing_arguments():
    calls = [{"type": "function", "function": {"name": "search"}}]
    out = _sanitize_judge_tool_calls(calls, _VALID, None)
    assert out[0]["function"]["arguments"] == "{}"


def test_sanitize_allows_multiple_calls_when_parallel_unset_or_true():
    calls = [
        {"type": "function", "function": {"name": "search", "arguments": "{}"}},
        {"type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
    ]
    assert len(_sanitize_judge_tool_calls(calls, _VALID, None)) == 2
    assert len(_sanitize_judge_tool_calls(calls, _VALID, True)) == 2


def test_sanitize_enforces_single_call_when_parallel_false():
    calls = [
        {"type": "function", "function": {"name": "search", "arguments": '{"q":"a"}'}},
        {"type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
    ]
    out = _sanitize_judge_tool_calls(calls, _VALID, False)
    assert len(out) == 1
    assert out[0]["function"]["name"] == "search"


def test_sanitize_drops_unknown_tool_name():
    calls = [{"type": "function", "function": {"name": "not_a_real_tool", "arguments": "{}"}}]
    assert _sanitize_judge_tool_calls(calls, _VALID, None) is None


def test_sanitize_drops_malformed_entries():
    calls = [
        "not-a-dict",
        {"type": "image", "function": {"name": "search", "arguments": "{}"}},  # wrong type
        {"type": "function"},  # no function block
        {"type": "function", "function": {"arguments": "{}"}},  # no name
    ]
    assert _sanitize_judge_tool_calls(calls, _VALID, None) is None


def test_sanitize_filters_unknown_but_keeps_valid_call():
    calls = [
        {"type": "function", "function": {"name": "ghost", "arguments": "{}"}},
        {"type": "function", "function": {"name": "search", "arguments": '{"q":"y"}'}},
    ]
    out = _sanitize_judge_tool_calls(calls, _VALID, None)
    assert len(out) == 1
    assert out[0]["function"]["name"] == "search"


def test_sanitize_none_and_empty_and_non_list():
    assert _sanitize_judge_tool_calls(None, _VALID, None) is None
    assert _sanitize_judge_tool_calls([], _VALID, None) is None
    assert _sanitize_judge_tool_calls({"not": "a list"}, _VALID, None) is None


# --- Integration: judge-authored calls through run_fusion_with_tools --------

from pydantic import BaseModel


class _MockUsage(BaseModel):
    prompt_tokens: int = 2
    completion_tokens: int = 3


class _MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _MockChoice:
    def __init__(self, message):
        self.message = message
        self.finish_reason = "tool_calls" if message.tool_calls else "stop"


class _MockResponse:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [_MockChoice(_MockMessage(content=content, tool_calls=tool_calls))]
        self.usage = _MockUsage()


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Return the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


async def _seed_two_panel_preset():
    from omnifusion.fusion.types import Preset, PresetStage
    from omnifusion.store.presets import save_preset

    stage = PresetStage(max_tokens=64, timeout=10)
    await save_preset(
        Preset(
            name="duo",
            strategy="B",
            panel_models=["model-a", "model-b"],
            panel=stage,
            judge_model="model-a",
            judge=stage,
            final_model="model-a",
            final=stage,
            cost_ceiling=1.0,
        )
    )


def _panel_call(city):
    return _MockResponse(
        tool_calls=[
            {
                "id": "call_panel",
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"city": city})},
            }
        ]
    )


@pytest.mark.asyncio
async def test_judge_authors_rewritten_tool_call(tmp_path, monkeypatch):
    """Two panel models propose imperfect calls; the judge emits a corrected call
    (arguments authored as a JSON object) and the response carries the judge's call,
    with usage aggregating both panel models + the judge."""
    import omnifusion.llm.client as client_mod
    from omnifusion.main import app
    from fastapi.testclient import TestClient
    from omnifusion.settings import settings
    from omnifusion.store.db import init_db

    old_db, old_keys = settings.db_path, settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "judge_authored.db")
    settings.omnifusion_api_keys = ["k"]

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        prompt = messages[0]["content"] if messages else ""
        if "PROPOSED NEXT STEPS" in prompt:
            # Judge rewrites the call: corrected casing, arguments as a JSON OBJECT
            # (not a string) to exercise the sanitizer's serialization.
            return _MockResponse(
                content=json.dumps(
                    {
                        "decision": "tool",
                        "best_index": 0,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                            }
                        ],
                        "reasoning": "corrected the city casing",
                    }
                )
            )
        if kwargs.get("tools"):
            # Panel proposals are imperfect and differ from the judge's final call.
            return _panel_call("paris" if model == "model-a" else "PARIS")
        return _MockResponse(content="unused")

    try:
        await init_db()
        await _seed_two_panel_preset()
        monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer k"},
                json={
                    "model": "fusion/duo",
                    "messages": [{"role": "user", "content": "Weather in paris?"}],
                    "tools": [_WEATHER_TOOL],
                    "tool_choice": "auto",
                    "store": True,
                },
            )
            run_id = resp.headers.get("X-OmniFusion-Run-Id")
            trace = client.get(
                f"/v1/traces/{run_id}", headers={"Authorization": "Bearer k"}
            ).json()
    finally:
        settings.db_path, settings.omnifusion_api_keys = old_db, old_keys

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    call = payload["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    # The emitted call is the judge's authored one (corrected "Paris"), not either
    # panel proposal ("paris" / "PARIS").
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    # Usage aggregates 2 panel calls + 1 judge call (each 2/3 prompt/completion).
    assert payload["usage"]["prompt_tokens"] == 6
    assert payload["usage"]["completion_tokens"] == 9
    assert "Judge-authored" in trace["judge_analysis"]["consensus"]


@pytest.mark.asyncio
async def test_falls_back_to_panel_proposal_on_unknown_judge_tool(tmp_path, monkeypatch):
    """If the judge authors an unknown tool (or no usable call), the response falls
    back to the selected panel proposal."""
    import omnifusion.llm.client as client_mod
    from omnifusion.main import app
    from fastapi.testclient import TestClient
    from omnifusion.settings import settings
    from omnifusion.store.db import init_db

    old_db, old_keys = settings.db_path, settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "judge_fallback.db")
    settings.omnifusion_api_keys = ["k"]

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        prompt = messages[0]["content"] if messages else ""
        if "PROPOSED NEXT STEPS" in prompt:
            return _MockResponse(
                content=json.dumps(
                    {
                        "decision": "tool",
                        "best_index": 0,
                        "tool_calls": [
                            {"type": "function", "function": {"name": "ghost_tool", "arguments": "{}"}}
                        ],
                        "reasoning": "hallucinated a tool",
                    }
                )
            )
        if kwargs.get("tools"):
            return _panel_call("Paris")
        return _MockResponse(content="unused")

    try:
        await init_db()
        await _seed_two_panel_preset()
        monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer k"},
                json={
                    "model": "fusion/duo",
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                    "tools": [_WEATHER_TOOL],
                    "tool_choice": "auto",
                    "store": True,
                },
            )
            run_id = resp.headers.get("X-OmniFusion-Run-Id")
            trace = client.get(
                f"/v1/traces/{run_id}", headers={"Authorization": "Bearer k"}
            ).json()
    finally:
        settings.db_path, settings.omnifusion_api_keys = old_db, old_keys

    assert resp.status_code == 200
    payload = resp.json()
    call = payload["choices"][0]["message"]["tool_calls"][0]
    # Fell back to the panel proposal, not the hallucinated tool.
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    assert "Selected tool call" in trace["judge_analysis"]["consensus"]


@pytest.mark.asyncio
async def test_parallel_false_emits_single_judge_call(tmp_path, monkeypatch):
    """parallel_tool_calls=False forces a single emitted call even when the judge
    authors several."""
    import omnifusion.llm.client as client_mod
    from omnifusion.main import app
    from fastapi.testclient import TestClient
    from omnifusion.settings import settings
    from omnifusion.store.db import init_db

    old_db, old_keys = settings.db_path, settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "judge_parallel.db")
    settings.omnifusion_api_keys = ["k"]

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        prompt = messages[0]["content"] if messages else ""
        if "PROPOSED NEXT STEPS" in prompt:
            return _MockResponse(
                content=json.dumps(
                    {
                        "decision": "tool",
                        "best_index": 0,
                        "tool_calls": [
                            {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
                            {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Lyon"}}},
                        ],
                        "reasoning": "two cities",
                    }
                )
            )
        if kwargs.get("tools"):
            return _panel_call("Paris")
        return _MockResponse(content="unused")

    try:
        await init_db()
        await _seed_two_panel_preset()
        monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer k"},
                json={
                    "model": "fusion/duo",
                    "messages": [{"role": "user", "content": "Weather?"}],
                    "tools": [_WEATHER_TOOL],
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                },
            )
    finally:
        settings.db_path, settings.omnifusion_api_keys = old_db, old_keys

    assert resp.status_code == 200
    tool_calls = resp.json()["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Paris"}
