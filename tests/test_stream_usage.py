"""stream_options.include_usage support: a terminal usage chunk is emitted before
[DONE] (backs the matrix stream_usage cells, which previously cited a test that never
exercised include_usage)."""
import json

import pytest
from unittest.mock import patch

from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage, StreamOptions
from omnifusion.fusion.orchestrator import run_fusion
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.settings import settings
from omnifusion.store.db import init_db


class _Delta:
    def __init__(self, content):
        self.content = content


class _StreamChoice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _StreamChunk:
    def __init__(self, content):
        self.choices = [_StreamChoice(content)]
        self.usage = None

    def model_dump_json(self):
        return json.dumps({"choices": [{"delta": {"content": self.choices[0].delta.content}}]})


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)
        self.finish_reason = "stop"


class _Usage:
    prompt_tokens = 10
    completion_tokens = 20


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


async def _gen(items):
    for i in items:
        yield i


def _preset():
    stage = PresetStage(max_tokens=50, timeout=10)
    return Preset(
        name="usage-preset",
        strategy="B",
        panel_models=["mock-model"],
        panel=stage,
        judge_model="mock-model",
        judge=stage,
        final_model="mock-model",
        final=stage,
        cost_ceiling=1.0,
        min_panel_success=1,
    )


@pytest.mark.asyncio
async def test_stream_emits_terminal_usage_chunk_when_include_usage(tmp_path):
    old_db = settings.db_path
    settings.db_path = str(tmp_path / "usage.db")
    try:
        await init_db()
        n = {"c": 0}

        async def mock(*a, **k):
            n["c"] += 1
            if n["c"] == 1:
                return _Resp("panel")
            if n["c"] == 2:
                return _Resp('{"consensus":"ok","recommended_final_answer_plan":"go"}')
            return _gen([_StreamChunk("Hi"), _StreamChunk("!")])

        with patch("omnifusion.llm.client.llm_client.acompletion", side_effect=mock):
            req = ChatCompletionRequest(
                model="fusion/usage-preset",
                messages=[ChatMessage(role="user", content="hi")],
                stream=True,
                stream_options=StreamOptions(include_usage=True),
                store=False,
            )
            result = await run_fusion("usage-run", _preset(), req, "k")

        blob = ""
        async for ch in result.body_iterator:
            blob += ch.decode() if isinstance(ch, bytes) else ch
    finally:
        import os

        if os.path.exists(settings.db_path):
            os.remove(settings.db_path)
        settings.db_path = old_db

    # A terminal usage chunk (empty choices + usage block) precedes [DONE].
    events = [line for line in blob.split("\n\n") if line.startswith("data: ") and line != "data: [DONE]"]
    usage_chunks = [
        json.loads(e[len("data: ") :])
        for e in events
        if '"usage"' in e
    ]
    assert usage_chunks, "no terminal usage chunk emitted"
    usage = usage_chunks[-1]["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert blob.rstrip().endswith("data: [DONE]")


@pytest.mark.asyncio
async def test_stream_without_include_usage_has_no_usage_chunk(tmp_path):
    old_db = settings.db_path
    settings.db_path = str(tmp_path / "no_usage.db")
    try:
        await init_db()
        n = {"c": 0}

        async def mock(*a, **k):
            n["c"] += 1
            if n["c"] == 1:
                return _Resp("panel")
            if n["c"] == 2:
                return _Resp('{"consensus":"ok","recommended_final_answer_plan":"go"}')
            return _gen([_StreamChunk("Hi")])

        with patch("omnifusion.llm.client.llm_client.acompletion", side_effect=mock):
            req = ChatCompletionRequest(
                model="fusion/usage-preset",
                messages=[ChatMessage(role="user", content="hi")],
                stream=True,
                store=False,
            )
            result = await run_fusion("no-usage-run", _preset(), req, "k")
        blob = ""
        async for ch in result.body_iterator:
            blob += ch.decode() if isinstance(ch, bytes) else ch
    finally:
        import os

        if os.path.exists(settings.db_path):
            os.remove(settings.db_path)
        settings.db_path = old_db

    assert '"usage"' not in blob
