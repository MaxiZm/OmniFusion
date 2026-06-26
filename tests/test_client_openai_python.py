"""Real openai-python SDK smoke against the app (pinned openai==2.42.0).

Drives the actual OpenAI SDK over an in-process ASGI transport so the client-contract
matrix's openai-python cells are backed by the real client, not only generic HTTP
contract checks. Uses AsyncOpenAI because httpx.ASGITransport is async-only."""
import json
import os

os.environ.setdefault("OMNIFUSION_ADMIN_PASSWORD", "test-password-123")
os.environ.setdefault(
    "OMNIFUSION_SECRET_KEY", "U1NfdlhjdmJubWwwMTIzNDU2Nzg5MGFiY2RlZmdoaWo="
)

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel
from unittest.mock import patch

from omnifusion.main import app
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.store.presets import save_preset
from omnifusion.store.providers import save_provider
from omnifusion.fusion.types import Preset, PresetStage


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)
        self.finish_reason = "stop"


class _Usage(BaseModel):
    prompt_tokens: int = 5
    completion_tokens: int = 5


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


@pytest.fixture
def sdk_env():
    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = "test_openai_sdk.db"
    settings.omnifusion_api_keys = ["sdk-token"]
    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)
    yield
    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)
    settings.db_path = old_db
    settings.omnifusion_api_keys = old_keys


def _make_client():
    transport = httpx.ASGITransport(app=app)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://omnifusion.test")
    return AsyncOpenAI(
        base_url="http://omnifusion.test/v1", api_key="sdk-token", http_client=http_client
    )


async def _setup():
    await init_db()
    await save_provider(
        "provider-1", "openai", "key-1", None, None, ["panel-a", "judge-model", "final-model"]
    )
    await save_preset(
        Preset(
            name="sdkpreset",
            strategy="B",
            panel_models=["panel-a"],
            panel=PresetStage(max_tokens=10, timeout=10),
            judge_model="judge-model",
            judge=PresetStage(max_tokens=10, timeout=10),
            final_model="final-model",
            final=PresetStage(max_tokens=20, timeout=20),
            cost_ceiling=1.0,
        )
    )


def _side_effect(provider_id, model, messages, **kwargs):
    if model == "panel-a":
        return _Resp("Answer A")
    if model == "judge-model":
        return _Resp('{"consensus": "agreed", "recommended_final_answer_plan": "plan"}')
    if model == "final-model":
        return _Resp("Hello from fusion")
    raise ValueError(model)


@pytest.mark.asyncio
async def test_openai_sdk_models_list(sdk_env):
    await _setup()
    client = _make_client()
    try:
        models = await client.models.list()
        ids = [m.id for m in models.data]
        assert "fusion/sdkpreset" in ids
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_openai_sdk_chat_completion(sdk_env):
    await _setup()
    client = _make_client()
    try:
        with patch("omnifusion.llm.client.llm_client.acompletion", side_effect=_side_effect):
            completion = await client.chat.completions.create(
                model="fusion/sdkpreset",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert completion.choices[0].message.content == "Hello from fusion"
        assert completion.model == "fusion/sdkpreset"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_openai_sdk_streaming(sdk_env):
    await _setup()
    client = _make_client()

    class _SChunk:
        def __init__(self, c):
            self.choices = [type("C", (), {"delta": type("D", (), {"content": c})()})()]
            self.usage = None

        def model_dump_json(self):
            return json.dumps({"choices": [{"delta": {"content": self.choices[0].delta.content}}]})

    async def _gen(items):
        for i in items:
            yield i

    def side_effect(provider_id, model, messages, **kwargs):
        if model == "panel-a":
            return _Resp("a")
        if model == "judge-model":
            return _Resp('{"consensus": "ok", "recommended_final_answer_plan": "p"}')
        return _gen([_SChunk("Hello"), _SChunk(" SDK")])

    try:
        with patch("omnifusion.llm.client.llm_client.acompletion", side_effect=side_effect):
            stream = await client.chat.completions.create(
                model="fusion/sdkpreset",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            text = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text += chunk.choices[0].delta.content
        assert "Hello SDK" in text
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_openai_sdk_error_envelope(sdk_env):
    await _setup()
    client = _make_client()
    from openai import BadRequestError

    try:
        with pytest.raises(BadRequestError):
            # n > 1 is unsupported -> 400 error envelope the SDK surfaces.
            await client.chat.completions.create(
                model="fusion/sdkpreset",
                messages=[{"role": "user", "content": "hi"}],
                n=2,
            )
    finally:
        await client.close()
