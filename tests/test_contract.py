import os

# Configure env variables before importing anything from omnifusion
os.environ["OMNIFUSION_ADMIN_PASSWORD"] = "test-password-123"
# A valid Fernet key for encrypt_key/decrypt_key verification
os.environ["OMNIFUSION_SECRET_KEY"] = "U1NfdlhjdmJubWwwMTIzNDU2Nzg5MGFiY2RlZmdoaWo="

import pytest
import json
from pydantic import BaseModel
from unittest.mock import patch
from fastapi.testclient import TestClient

from omnifusion.main import app
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.store.presets import save_preset
from omnifusion.store.providers import save_provider
from omnifusion.fusion.types import Preset, PresetStage


# Mock response classes
class MockMessage:
    def __init__(self, content):
        self.content = content


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)


class MockUsage(BaseModel):
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


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
        self.content = content
        self.choices = [MockChoiceChunk(content)]

    def model_dump_json(self):
        return json.dumps({"choices": [{"delta": {"content": self.content}}]})


async def mock_stream_generator(content_list):
    for chunk in content_list:
        yield MockStreamChunk(chunk)


@pytest.fixture(autouse=True)
def setup_settings_and_db():
    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    old_whitelist = settings.omnifusion_passthrough_whitelist

    settings.db_path = "test_contract.db"
    settings.omnifusion_api_keys = ["test-token-1", "test-token-2"]
    settings.omnifusion_passthrough_whitelist = ["gpt-4o"]

    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)

    client = TestClient(app)
    yield client

    if os.path.exists(settings.db_path):
        try:
            os.remove(settings.db_path)
        except Exception:
            pass

    settings.db_path = old_db
    settings.omnifusion_api_keys = old_keys
    settings.omnifusion_passthrough_whitelist = old_whitelist


@pytest.mark.asyncio
async def test_bearer_authentication(setup_settings_and_db):
    client = setup_settings_and_db
    await init_db()

    # 1. No auth header -> 401/403
    response = client.get("/v1/models")
    assert response.status_code in (401, 403)

    # 2. Invalid auth header -> 401 Unauthorized
    response = client.get(
        "/v1/models", headers={"Authorization": "Bearer invalid-token"}
    )
    assert response.status_code == 401

    # 3. Valid auth header -> 200 OK
    response = client.get(
        "/v1/models", headers={"Authorization": "Bearer test-token-1"}
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_models_endpoint_listing(setup_settings_and_db):
    client = setup_settings_and_db
    await init_db()

    # Save a preset
    preset = Preset(
        name="testpreset",
        strategy="B",
        panel_models=["panel-a"],
        panel=PresetStage(max_tokens=10, timeout=10),
        judge_model="judge-model",
        judge=PresetStage(max_tokens=10, timeout=10),
        final_model="final-model",
        final=PresetStage(max_tokens=20, timeout=20),
        cost_ceiling=1.0,
    )
    await save_preset(preset)

    headers = {"Authorization": "Bearer test-token-1"}
    response = client.get("/v1/models", headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "data" in data
    model_ids = [m["id"] for m in data["data"]]
    assert "fusion/testpreset" in model_ids
    assert "gpt-4o" in model_ids  # passthrough whitelisted model


@pytest.mark.asyncio
async def test_request_validation_rejection(setup_settings_and_db):
    client = setup_settings_and_db
    await init_db()

    headers = {"Authorization": "Bearer test-token-1"}

    # Rejects legacy `functions` (use `tools` instead). NOTE: `tools`/`tool_choice`
    # are now ACCEPTED — they route to a tool-capable model (agentic clients).
    req_body = {
        "model": "fusion/testpreset",
        "messages": [{"role": "user", "content": "hello"}],
        "functions": [{"name": "test"}],
    }
    response = client.post("/v1/chat/completions", headers=headers, json=req_body)
    assert response.status_code == 400
    err_data = response.json()
    assert "error" in err_data
    assert "functions" in err_data["error"]["message"]

    # Rejects n > 1
    req_body = {
        "model": "fusion/testpreset",
        "messages": [{"role": "user", "content": "hello"}],
        "n": 2,
    }
    response = client.post("/v1/chat/completions", headers=headers, json=req_body)
    assert response.status_code == 400
    err_data = response.json()
    assert "n > 1 is not supported" in err_data["error"]["message"]


@pytest.mark.asyncio
@patch("omnifusion.llm.client.llm_client.acompletion")
async def test_successful_contract_flow(mock_acompletion, setup_settings_and_db):
    client = setup_settings_and_db
    await init_db()

    # Pre-populate provider & preset
    await save_provider(
        "provider-1",
        "openai",
        "key-1",
        None,
        None,
        ["panel-a", "judge-model", "final-model"],
    )
    preset = Preset(
        name="contractpreset",
        strategy="B",
        panel_models=["panel-a"],
        panel=PresetStage(max_tokens=10, timeout=10),
        judge_model="judge-model",
        judge=PresetStage(max_tokens=10, timeout=10),
        final_model="final-model",
        final=PresetStage(max_tokens=20, timeout=20),
        cost_ceiling=1.0,
    )
    await save_preset(preset)

    # Setup mock behaviors
    def side_effect(provider_id, model, messages, **kwargs):
        if model == "panel-a":
            return MockResponse("Answer A", 5, 5)
        elif model == "judge-model":
            return MockResponse(
                '{"consensus": "agreed", "recommended_final_answer_plan": "plan"}', 5, 5
            )
        elif model == "final-model":
            if kwargs.get("stream"):
                return mock_stream_generator(["Hello", " world"])
            return MockResponse("Hello world", 5, 5)
        raise ValueError(f"Unknown mock model: {model}")

    mock_acompletion.side_effect = side_effect

    headers = {"Authorization": "Bearer test-token-1"}

    # 1. Non-streaming completion
    req_body = {
        "model": "fusion/contractpreset",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "store": True,
    }
    response = client.post("/v1/chat/completions", headers=headers, json=req_body)
    assert response.status_code == 200

    # Check headers
    run_id = response.headers.get("X-OmniFusion-Run-Id")
    assert run_id is not None

    # Check body
    res_data = response.json()
    assert res_data["choices"][0]["message"]["content"] == "Hello world"

    # 2. Check Trace endpoint with owner key
    trace_res = client.get(
        f"/v1/traces/{run_id}", headers={"Authorization": "Bearer test-token-1"}
    )
    assert trace_res.status_code == 200
    assert trace_res.json()["run_id"] == run_id

    # 3. Check Trace endpoint with other key (unauthorized / 404)
    trace_res_other = client.get(
        f"/v1/traces/{run_id}", headers={"Authorization": "Bearer test-token-2"}
    )
    assert trace_res_other.status_code == 404

    # 4. Check Trace endpoint with store:false (run another one)
    req_body_no_store = {
        "model": "fusion/contractpreset",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "store": False,
    }
    response_no_store = client.post(
        "/v1/chat/completions", headers=headers, json=req_body_no_store
    )
    run_id_no_store = response_no_store.headers.get("X-OmniFusion-Run-Id")

    trace_res_no_store = client.get(
        f"/v1/traces/{run_id_no_store}",
        headers={"Authorization": "Bearer test-token-1"},
    )
    assert trace_res_no_store.status_code == 404

    # 5. Streaming completion must also carry the run-id header so stream clients
    # can fetch the trace (P1: StreamingResponse bypasses the injected Response).
    req_body_stream = {
        "model": "fusion/contractpreset",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "store": True,
    }
    with client.stream(
        "POST", "/v1/chat/completions", headers=headers, json=req_body_stream
    ) as stream_resp:
        assert stream_resp.status_code == 200
        stream_run_id = stream_resp.headers.get("X-OmniFusion-Run-Id")
        assert stream_run_id is not None
        body = "".join(stream_resp.iter_text())
    assert "data: [DONE]" in body

    # The streamed run's trace is retrievable by run-id from the header.
    stream_trace = client.get(
        f"/v1/traces/{stream_run_id}",
        headers={"Authorization": "Bearer test-token-1"},
    )
    assert stream_trace.status_code == 200
