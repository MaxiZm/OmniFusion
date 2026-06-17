"""
Tests for streaming paths and streaming error handling.
Fix #17: Streaming paths previously had zero coverage.
"""
import pytest
import os
import json
from unittest.mock import patch

from omnifusion.store.db import init_db
from omnifusion.store.presets import save_preset
from omnifusion.store.providers import save_provider
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.fusion.orchestrator import run_fusion
from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage
from omnifusion.settings import settings


@pytest.fixture(autouse=True)
def setup_db():
    old_db = settings.db_path
    settings.db_path = "test_streaming.db"
    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)
    yield
    if os.path.exists(settings.db_path):
        try:
            os.remove(settings.db_path)
        except Exception:
            pass
    settings.db_path = old_db


def make_preset(name="stream-preset") -> Preset:
    return Preset(
        name=name,
        strategy="B",
        panel_models=["mock-model"],
        panel=PresetStage(max_tokens=50, timeout=10),
        judge_model="mock-model",
        judge=PresetStage(max_tokens=50, timeout=10),
        final_model="mock-model",
        final=PresetStage(max_tokens=50, timeout=10),
        cost_ceiling=1.0,
        min_panel_success=1,
    )


class MockDelta:
    def __init__(self, content):
        self.content = content


class MockStreamChoice:
    def __init__(self, content):
        self.delta = MockDelta(content)


class MockStreamChunk:
    def __init__(self, content):
        self.choices = [MockStreamChoice(content)]

    def model_dump_json(self):
        return json.dumps({"choices": [{"delta": {"content": self.choices[0].delta.content}}]})


async def _make_async_gen(items, raise_after_n=None):
    """Helper: yield items from a list, optionally raising an error after N items."""
    for i, item in enumerate(items):
        yield item
        if raise_after_n is not None and i + 1 >= raise_after_n:
            raise RuntimeError("Simulated mid-stream error")


class MockMessage:
    def __init__(self, content):
        self.content = content


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)
        self.delta = MockDelta(content)


class MockUsage:
    prompt_tokens = 10
    completion_tokens = 20


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]
        self.usage = MockUsage()
        self.model = "mock-model"

    def model_dump_json(self):
        return json.dumps({"choices": [{"delta": {"content": self.choices[0].message.content}}]})


@pytest.mark.asyncio
async def test_streaming_response_format():
    """Test that fusion streaming returns SSE-formatted chunks and [DONE]."""
    await init_db()
    await save_provider("default", "openai", "test-key", models=["mock-model"])

    preset = make_preset()
    await save_preset(preset)

    chunks = [
        MockStreamChunk("Hello"),
        MockStreamChunk(" world"),
        MockStreamChunk("!"),
    ]

    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count in (1,):  # Panel
            return MockResponse("Panel answer")
        elif call_count == 2:  # Judge
            return MockResponse('{"consensus": "good", "recommended_final_answer_plan": "synthesize"}')
        elif call_count == 3:  # Synth (streaming)
            return _make_async_gen(chunks)
        raise ValueError(f"Unexpected call {call_count}")

    with patch("omnifusion.llm.client.llm_client.acompletion", side_effect=mock_acompletion):
        req = ChatCompletionRequest(
            model="fusion/stream-preset",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
            store=False,
        )
        result = await run_fusion("test-stream-run-1", preset, req, "test-key")

    # run_fusion with stream=True returns a StreamingResponse
    from fastapi.responses import StreamingResponse
    assert isinstance(result, StreamingResponse)
    assert result.media_type == "text/event-stream"

    # Consume the streaming response and collect all SSE events
    events = []
    async for chunk in result.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        events.append(chunk)

    combined = "".join(events)
    # Must contain [DONE]
    assert "data: [DONE]" in combined, f"[DONE] not found in stream output: {combined[:500]}"
    # Must contain SSE data: prefix
    assert "data: " in combined


@pytest.mark.asyncio
async def test_streaming_error_aborts_without_done():
    """Mid-stream errors must abort the stream: the partial content is emitted, but
    NO synthetic error chunk and NO [DONE] terminator are sent, and the generator
    re-raises so the transport closes abnormally. This keeps OpenAI-compatible
    clients from treating a failed stream as cleanly completed.

    We mock run_synthesis to return an async generator that raises after the first
    chunk, exercising the orchestrator's stream_generator error path directly.
    """
    await init_db()
    await save_provider("default", "openai", "test-key", models=["mock-model"])

    preset = make_preset("stream-error-preset")
    await save_preset(preset)

    # Mock generator that yields 1 chunk then raises
    async def erroring_gen():
        yield MockStreamChunk("Partial")
        raise RuntimeError("Simulated mid-stream error")

    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:  # Panel
            return MockResponse("Panel answer")
        elif call_count == 2:  # Judge
            return MockResponse('{"consensus": "good", "recommended_final_answer_plan": "synthesize"}')
        raise ValueError(f"Unexpected call {call_count}")

    # We mock run_synthesis to return our erroring generator directly,
    # bypassing synth.py's cleanup machinery.
    async def mock_run_synthesis(*args, **kwargs):
        return erroring_gen()

    with patch("omnifusion.llm.client.llm_client.acompletion", side_effect=mock_acompletion), \
         patch("omnifusion.fusion.orchestrator.run_synthesis", side_effect=mock_run_synthesis):
        req = ChatCompletionRequest(
            model="fusion/stream-error-preset",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
            store=False,
        )
        result = await run_fusion("test-stream-run-2", preset, req, "test-key")

    from fastapi.responses import StreamingResponse
    assert isinstance(result, StreamingResponse)

    # Consuming the stream must raise (abnormal close), after yielding partial content.
    events = []
    with pytest.raises(RuntimeError, match="Simulated mid-stream error"):
        async for chunk in result.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode()
            events.append(chunk)

    combined = "".join(events)

    # The partial content was delivered...
    assert "Partial" in combined, f"partial content missing: {combined[:500]}"
    # ...but the stream must NOT be cleanly terminated.
    assert "data: [DONE]" not in combined, f"[DONE] must not be sent on error: {combined[:500]}"
    # ...and no synthetic error chunk is fabricated.
    assert "stream_error" not in combined, f"synthetic error chunk should not be emitted: {combined[:500]}"


@pytest.mark.asyncio
async def test_streaming_always_emits_done_on_normal_completion():
    """Regression: [DONE] must be present even on happy path."""
    await init_db()
    await save_provider("default", "openai", "test-key", models=["mock-model"])

    preset = make_preset("stream-done-preset")
    await save_preset(preset)

    chunks = [MockStreamChunk("A"), MockStreamChunk("B")]
    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockResponse("Panel")
        elif call_count == 2:
            return MockResponse('{"consensus": "ok", "recommended_final_answer_plan": "go"}')
        elif call_count == 3:
            return _make_async_gen(chunks)
        raise ValueError(f"Unexpected call {call_count}")

    with patch("omnifusion.llm.client.llm_client.acompletion", side_effect=mock_acompletion):
        req = ChatCompletionRequest(
            model="fusion/stream-done-preset",
            messages=[ChatMessage(role="user", content="Test")],
            stream=True,
            store=False,
        )
        result = await run_fusion("test-stream-run-3", preset, req, "test-key")

    from fastapi.responses import StreamingResponse
    assert isinstance(result, StreamingResponse)

    all_data = ""
    async for chunk in result.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        all_data += chunk

    assert "data: [DONE]" in all_data
    # Content chunks should appear
    assert "data: " in all_data
