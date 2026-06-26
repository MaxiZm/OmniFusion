import json

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from omnifusion.settings import settings


class MockUsage(BaseModel):
    prompt_tokens: int = 2
    completion_tokens: int = 3


class MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    def __init__(self, message):
        self.message = message
        self.finish_reason = "tool_calls" if message.tool_calls else "stop"


class MockResponse:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [MockChoice(MockMessage(content=content, tool_calls=tool_calls))]
        self.usage = MockUsage()


@pytest.mark.asyncio
async def test_opencode_style_tool_loop_through_fugu_ultra(tmp_path, monkeypatch):
    import omnifusion.llm.client as client_mod
    from omnifusion.main import app

    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "m8_opencode.db")
    settings.omnifusion_api_keys = ["opencode-key"]

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        if kwargs.get("tools"):
            has_tool_result = any(message.get("role") == "tool" for message in messages)
            if has_tool_result:
                return MockResponse(content="The tool result says it is sunny.")
            return MockResponse(
                tool_calls=[
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                        },
                    }
                ]
            )

        prompt = messages[0]["content"]
        if "PROPOSED NEXT STEPS" in prompt:
            return MockResponse(
                content=json.dumps(
                    {
                        "decision": "tool",
                        "best_index": 0,
                        "reasoning": "weather lookup is needed",
                    }
                )
            )
        if "Output valid JSON ONLY" in prompt:
            return MockResponse(
                content=json.dumps(
                    {
                        "consensus": "tool result is enough",
                        "contradictions": "",
                        "synthesis_plan": "answer from the tool result",
                    }
                )
            )
        return MockResponse(content="It is sunny in Paris.")

    weather_tool = {
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

    try:
        monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
        with TestClient(app) as client:
            first = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer opencode-key"},
                json={
                    "model": "fugu-ultra",
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                    "tools": [weather_tool],
                    "tool_choice": "auto",
                    "store": True,
                },
            )
            first_run_id = first.headers.get("X-OmniFusion-Run-Id")
            first_payload = first.json()
            tool_calls = first_payload["choices"][0]["message"]["tool_calls"]

            trace = client.get(
                f"/v1/traces/{first_run_id}",
                headers={"Authorization": "Bearer opencode-key"},
            )

            second = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer opencode-key"},
                json={
                    "model": "fugu-ultra",
                    "messages": [
                        {"role": "user", "content": "Weather in Paris?"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_weather",
                            "content": "sunny",
                        },
                    ],
                    "tools": [weather_tool],
                    "tool_choice": "auto",
                    "store": True,
                },
            )
            second_run_id = second.headers.get("X-OmniFusion-Run-Id")
            second_trace = client.get(
                f"/v1/traces/{second_run_id}",
                headers={"Authorization": "Bearer opencode-key"},
            )
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys

    assert first.status_code == 200
    assert first_run_id
    assert first_payload["model"] == "fusion/fugu-ultra"
    assert first_payload["choices"][0]["finish_reason"] == "tool_calls"
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert trace.status_code == 200

    assert second.status_code == 200
    assert second_run_id
    assert second.json()["model"] == "fusion/fugu-ultra"
    assert second.json()["choices"][0]["message"]["content"] == "It is sunny in Paris."
    assert second_trace.status_code == 200
    assert second_trace.json()["final_answer"] == "It is sunny in Paris."
