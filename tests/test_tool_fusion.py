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
