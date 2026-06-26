"""M3b exit gate: byte-diff vs the checked-in canonical SSE fixture is empty for a
fixed sequence, and the non-stream dict is identical. This locks the single canonical
SSE/response shape — any drift in shaping breaks the diff and forces a deliberate
fixture update."""
import json
import os
import re
from pathlib import Path

import pytest
from unittest.mock import patch

from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage
from omnifusion.fusion.orchestrator import run_fusion
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.settings import settings
from omnifusion.store.db import init_db

FIXTURE = Path("tests/fixtures/sse/classic_stream.sse")


class _D:
    def __init__(self, c):
        self.content = c


class _C:
    def __init__(self, c):
        self.delta = _D(c)
        self.message = _D(c)
        self.finish_reason = "stop"


class _Chunk:
    def __init__(self, c):
        self.choices = [_C(c)]
        self.usage = None

    def model_dump_json(self):
        return json.dumps(
            {
                "choices": [
                    {"index": 0, "delta": {"content": self.choices[0].delta.content}, "finish_reason": None}
                ]
            }
        )


class _Usage:
    prompt_tokens = 10
    completion_tokens = 20


class _Resp:
    def __init__(self, c):
        self.choices = [_C(c)]
        self.usage = _Usage()
        self.model = "m"


async def _gen(items):
    for item in items:
        yield item


def _preset():
    return Preset(
        name="canonical-sse",
        strategy="B",
        panel_models=["m"],
        panel=PresetStage(max_tokens=50, timeout=10),
        judge_model="m",
        judge=PresetStage(max_tokens=50, timeout=10),
        final_model="m",
        final=PresetStage(max_tokens=50, timeout=10),
        cost_ceiling=1.0,
        min_panel_success=1,
    )


def _mock_factory(stream_chunks):
    state = {"n": 0}

    async def mock(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return _Resp("panel")
        if state["n"] == 2:
            return _Resp('{"consensus":"ok","recommended_final_answer_plan":"go"}')
        if state["n"] == 3:
            return _gen(stream_chunks)
        raise ValueError(state["n"])

    return mock


@pytest.fixture(autouse=True)
def _db(tmp_path):
    old = settings.db_path
    settings.db_path = str(tmp_path / "m3b.db")
    yield
    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)
    settings.db_path = old


@pytest.mark.asyncio
async def test_classic_stream_byte_diff_against_canonical_fixture():
    await init_db()
    chunks = [_Chunk("Hello"), _Chunk(" world"), _Chunk("!")]
    with patch("omnifusion.llm.client.llm_client.acompletion", side_effect=_mock_factory(chunks)):
        req = ChatCompletionRequest(
            model="fusion/canonical-sse",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
            store=False,
        )
        result = await run_fusion("m3b-stream", _preset(), req, "k")

    blob = ""
    async for ch in result.body_iterator:
        blob += ch.decode() if isinstance(ch, bytes) else ch

    expected = FIXTURE.read_text()
    assert blob == expected, f"SSE byte-diff is non-empty:\n--- got ---\n{blob!r}\n--- want ---\n{expected!r}"


@pytest.mark.asyncio
async def test_non_stream_dict_is_identical_modulo_id_and_timestamp():
    await init_db()
    with patch("omnifusion.llm.client.llm_client.acompletion", side_effect=_mock_factory([])):
        req = ChatCompletionRequest(
            model="fusion/canonical-sse",
            messages=[ChatMessage(role="user", content="hi")],
            stream=False,
            store=False,
        )

        # Non-stream final synthesis returns a single response, not a generator.
        async def mock(*args, **kwargs):
            mock.n = getattr(mock, "n", 0) + 1
            if mock.n == 1:
                return _Resp("panel")
            if mock.n == 2:
                return _Resp('{"consensus":"ok","recommended_final_answer_plan":"go"}')
            if mock.n == 3:
                return _Resp("Hello world!")
            raise ValueError(mock.n)

        with patch("omnifusion.llm.client.llm_client.acompletion", side_effect=mock):
            result = await run_fusion("m3b-nonstream", _preset(), req, "k")

    # Normalize the inherently-volatile id/created, then assert exact structure.
    result["id"] = re.sub(r"^chatcmpl-.*$", "chatcmpl-<id>", result["id"])
    result["created"] = "<created>"
    assert result == {
        "id": "chatcmpl-<id>",
        "object": "chat.completion",
        "created": "<created>",
        "model": "fusion/canonical-sse",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello world!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 60, "total_tokens": 90},
    }
