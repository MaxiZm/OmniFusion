import pytest
import os
import asyncio
from unittest.mock import patch
from omnifusion.fusion.orchestrator import run_fusion
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage
from omnifusion.api.errors import InsufficientPanelError
from omnifusion.store.db import init_db
from omnifusion.settings import settings

from pydantic import BaseModel


# Dummy mock objects matching LiteLLM outputs
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
        self.usage = MockUsage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )


# Stream chunk mocks
class MockDelta:
    def __init__(self, content):
        self.content = content


class MockChoiceChunk:
    def __init__(self, content):
        self.delta = MockDelta(content)


class MockStreamChunk:
    def __init__(self, content):
        self.choices = [MockChoiceChunk(content)]

    def model_dump_json(self):
        return '{"choices": [{"delta": {"content": "' + self.content + '"}}]}'


async def mock_stream_generator(content_list):
    for chunk in content_list:
        yield MockStreamChunk(chunk)
        await asyncio.sleep(0.01)


@pytest.fixture(autouse=True)
def setup_db():
    old_db = settings.db_path
    settings.db_path = "test_orchestrator.db"
    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)
    yield
    if os.path.exists(settings.db_path):
        try:
            os.remove(settings.db_path)
        except Exception:
            pass
    settings.db_path = old_db


@pytest.mark.asyncio
@patch("omnifusion.llm.client.llm_client.acompletion")
async def test_successful_fusion_run(mock_acompletion):
    await init_db()

    # Configure mock responses
    def side_effect(provider_id, model, messages, **kwargs):
        if model == "panel-a":
            return MockResponse("Answer A", 10, 10)
        elif model == "panel-b":
            return MockResponse("Answer B", 10, 10)
        elif model == "judge-model":
            return MockResponse(
                '{"consensus": "agreed", "recommended_final_answer_plan": "plan"}',
                20,
                20,
            )
        elif model == "final-model":
            if kwargs.get("stream"):
                return mock_stream_generator(["Synthesized", " answer"])
            else:
                return MockResponse("Synthesized answer", 30, 30)
        raise ValueError(f"Unknown mock model: {model}")

    mock_acompletion.side_effect = side_effect

    preset = Preset(
        name="general",
        strategy="B",
        panel_models=["panel-a", "panel-b"],
        panel=PresetStage(max_tokens=100, timeout=10),
        judge_model="judge-model",
        judge=PresetStage(max_tokens=100, timeout=10),
        final_model="final-model",
        final=PresetStage(max_tokens=200, timeout=20),
        cost_ceiling=1.0,
        min_panel_success=2,
    )

    # Test non-streaming
    req = ChatCompletionRequest(
        model="fusion/general",
        messages=[ChatMessage(role="user", content="hello")],
        stream=False,
        store=True,
    )

    res = await run_fusion("run-1", preset, req, "keyhash123")
    assert res["choices"][0]["message"]["content"] == "Synthesized answer"
    assert res["usage"]["prompt_tokens"] > 0
    assert res["model"] == "fusion/general"

    # Usage aggregates panel (2×10) + judge (20) + final (30) prompt tokens = 70,
    # and completion 2×10 + 20 + 30 = 70. Proves it is NOT final-only (which would
    # report just 30/30).
    assert res["usage"]["prompt_tokens"] == 70
    assert res["usage"]["completion_tokens"] == 70
    assert res["usage"]["total_tokens"] == 140


@pytest.mark.asyncio
@patch("omnifusion.llm.client.llm_client.acompletion")
async def test_usage_reporting_final_only(mock_acompletion):
    await init_db()

    def side_effect(provider_id, model, messages, **kwargs):
        if model == "panel-a":
            return MockResponse("Answer A", 10, 10)
        elif model == "judge-model":
            return MockResponse(
                '{"consensus": "agreed", "recommended_final_answer_plan": "plan"}', 20, 20
            )
        elif model == "final-model":
            return MockResponse("Synthesized answer", 30, 30)
        raise ValueError(f"Unknown mock model: {model}")

    mock_acompletion.side_effect = side_effect

    preset = Preset(
        name="general",
        strategy="B",
        panel_models=["panel-a"],
        panel=PresetStage(max_tokens=100, timeout=10),
        judge_model="judge-model",
        judge=PresetStage(max_tokens=100, timeout=10),
        final_model="final-model",
        final=PresetStage(max_tokens=200, timeout=20),
        cost_ceiling=1.0,
        min_panel_success=1,
        usage_reporting="final",
    )
    req = ChatCompletionRequest(
        model="fusion/general",
        messages=[ChatMessage(role="user", content="hello")],
        stream=False,
        store=True,
    )

    res = await run_fusion("run-final-usage", preset, req, "keyhash123")
    # usage_reporting="final" reports ONLY the final synthesis call's tokens.
    assert res["usage"]["prompt_tokens"] == 30
    assert res["usage"]["completion_tokens"] == 30


@pytest.mark.asyncio
@patch("omnifusion.llm.client.llm_client.acompletion")
async def test_partial_panel_failure_tolerance(mock_acompletion):
    await init_db()

    # Model A succeeds, Model B throws exception (e.g. Rate limit)
    def side_effect(provider_id, model, messages, **kwargs):
        if model == "panel-a":
            return MockResponse("Answer A", 10, 10)
        elif model == "panel-b":
            raise Exception("Rate limit hit")
        elif model == "judge-model":
            return MockResponse(
                '{"consensus": "agreed", "recommended_final_answer_plan": "plan"}',
                20,
                20,
            )
        elif model == "final-model":
            return MockResponse("Synthesized answer", 30, 30)
        raise ValueError(f"Unknown mock model: {model}")

    mock_acompletion.side_effect = side_effect

    preset = Preset(
        name="general",
        strategy="B",
        panel_models=["panel-a", "panel-b"],
        panel=PresetStage(max_tokens=100, timeout=10),
        judge_model="judge-model",
        judge=PresetStage(max_tokens=100, timeout=10),
        final_model="final-model",
        final=PresetStage(max_tokens=200, timeout=20),
        cost_ceiling=1.0,
        min_panel_success=1,  # Require at least 1 success, so it should succeed!
    )

    req = ChatCompletionRequest(
        model="fusion/general",
        messages=[ChatMessage(role="user", content="hello")],
        stream=False,
        store=True,
    )

    res = await run_fusion("run-2", preset, req, "keyhash123")
    assert res["choices"][0]["message"]["content"] == "Synthesized answer"


@pytest.mark.asyncio
@patch("omnifusion.llm.client.llm_client.acompletion")
async def test_insufficient_panel_success_raises(mock_acompletion):
    await init_db()

    # Both fail
    mock_acompletion.side_effect = Exception("API Connection Error")

    preset = Preset(
        name="general",
        strategy="B",
        panel_models=["panel-a", "panel-b"],
        panel=PresetStage(max_tokens=100, timeout=10),
        judge_model="judge-model",
        judge=PresetStage(max_tokens=100, timeout=10),
        final_model="final-model",
        final=PresetStage(max_tokens=200, timeout=20),
        cost_ceiling=1.0,
        min_panel_success=1,
    )

    req = ChatCompletionRequest(
        model="fusion/general",
        messages=[ChatMessage(role="user", content="hello")],
        stream=False,
        store=True,
    )

    with pytest.raises(InsufficientPanelError):
        await run_fusion("run-3", preset, req, "keyhash123")
