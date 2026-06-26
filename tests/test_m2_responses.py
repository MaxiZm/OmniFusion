import json

import pytest
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from omnifusion.settings import settings
from omnifusion.store.db import init_db


@pytest.mark.asyncio
async def test_responses_endpoint_maps_text_request_to_chat_shape(tmp_path, monkeypatch):
    import omnifusion.api.chat as chat_mod
    from omnifusion.main import app

    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "m2_responses.db")
    settings.omnifusion_api_keys = ["responses-key"]
    captured = {}

    async def fake_run_fusion(run_id, preset, body, key_hash):
        captured["model"] = body.model
        captured["messages"] = [message.model_dump() for message in body.messages]
        captured["max_tokens"] = body.max_tokens
        captured["metadata"] = body.metadata
        return {
            "id": "chatcmpl-resp",
            "object": "chat.completion",
            "created": 123,
            "model": f"fusion/{preset.name}",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }

    try:
        await init_db()
        monkeypatch.setattr(chat_mod, "run_fusion", fake_run_fusion)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer responses-key"},
                json={
                    "model": "fugu",
                    "instructions": "be concise",
                    "input": "hello",
                    "max_output_tokens": 33,
                    "metadata": {"case": "responses"},
                },
            )
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys

    assert response.status_code == 200
    assert captured["model"] == "fusion/fugu"
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][0]["content"] == "be concise"
    assert captured["messages"][1]["role"] == "user"
    assert captured["messages"][1]["content"] == "hello"
    assert captured["max_tokens"] == 33
    assert captured["metadata"] == {"case": "responses"}

    payload = response.json()
    assert payload["object"] == "response"
    assert payload["created_at"] == 123
    assert payload["model"] == "fusion/fugu"
    assert payload["status"] == "completed"
    assert payload["output"][0]["content"][0] == {
        "type": "output_text",
        "text": "answer",
    }
    assert payload["usage"] == {
        "input_tokens": 5,
        "output_tokens": 7,
        "total_tokens": 12,
    }


@pytest.mark.asyncio
async def test_api_v1_responses_stream_emits_minimal_text_events(tmp_path, monkeypatch):
    import omnifusion.api.chat as chat_mod
    from omnifusion.main import app

    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "m2_responses_stream.db")
    settings.omnifusion_api_keys = ["responses-stream-key"]

    async def fake_run_fusion(run_id, preset, body, key_hash):
        async def chunks():
            for text in ["hel", "lo"]:
                data = {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": f"fusion/{preset.name}",
                    "choices": [{"index": 0, "delta": {"content": text}}],
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    try:
        await init_db()
        monkeypatch.setattr(chat_mod, "run_fusion", fake_run_fusion)
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/responses",
                headers={"Authorization": "Bearer responses-stream-key"},
                json={"model": "fugu", "input": "hello", "stream": True},
            )
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys

    assert response.status_code == 200
    assert "event: response.output_text.delta" in response.text
    assert '"delta": "hel"' in response.text
    assert '"delta": "lo"' in response.text
    assert "event: response.completed" in response.text
    assert "data: [DONE]" not in response.text
