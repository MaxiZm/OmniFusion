import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.settings import settings


def valid_preset(**overrides):
    stage = PresetStage(max_tokens=16, timeout=5)
    data = {
        "name": "bounded",
        "strategy": "B",
        "panel_models": ["m1", "m2"],
        "panel": stage,
        "judge_model": "judge",
        "judge": stage,
        "final_model": "final",
        "final": stage,
        "min_panel_success": 1,
    }
    data.update(overrides)
    return Preset(**data)


def test_tool_shapes_are_typed_and_valid_tool_choice_is_preserved():
    req = ChatCompletionRequest(
        model="fusion/general",
        messages=[ChatMessage(role="user", content="weather?")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Read weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )

    assert req.tools[0].type == "function"
    assert req.tools[0].function.name == "get_weather"
    assert req.tool_choice.function.name == "get_weather"


@pytest.mark.parametrize(
    "payload",
    [
        {"tools": [{"type": "web_search", "function": {"name": "search"}}]},
        {"tools": [{"type": "function", "function": {"parameters": {}}}]},
        {"tool_choice": {"type": "function", "function": {}}},
        {
            "messages": [
                {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function"}]}
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "x", "arguments": {"bad": "shape"}},
                        }
                    ],
                }
            ]
        },
    ],
)
def test_malformed_tool_shapes_are_rejected(payload):
    base = {"model": "fusion/general", "messages": [{"role": "user", "content": "hi"}]}
    base.update(payload)

    with pytest.raises(ValidationError):
        ChatCompletionRequest(**base)


def test_request_body_size_cap_returns_413(tmp_path):
    from omnifusion.main import app

    old_db = settings.db_path
    old_limit = settings.omnifusion_max_request_body_bytes
    settings.db_path = str(tmp_path / "m1c.db")
    settings.omnifusion_max_request_body_bytes = 128

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "fusion/general",
                    "messages": [{"role": "user", "content": "x" * 256}],
                },
            )
    finally:
        settings.db_path = old_db
        settings.omnifusion_max_request_body_bytes = old_limit

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_body_too_large"


@pytest.mark.parametrize(
    "overrides",
    [
        {"usage_reporting": "verbose"},
        {"on_final_failure": "fallback"},
        {"panel_models": []},
        {"panel_models": [f"m{i}" for i in range(settings.max_panel + 1)]},
        {"min_panel_success": 3},
        {"cost_ceiling": 0},
        {"cost_ceiling": settings.global_daily_budget_usd + 1},
        {"strategy": "unknown"},
    ],
)
def test_preset_rejects_invalid_enums_and_bounds(overrides):
    with pytest.raises(ValidationError):
        valid_preset(**overrides)


@pytest.mark.parametrize(
    "stage",
    [
        {"max_tokens": 0, "timeout": 5},
        {"max_tokens": settings.omnifusion_max_tokens_limit + 1, "timeout": 5},
        {"max_tokens": 16, "timeout": 0},
        {"max_tokens": 16, "timeout": settings.omnifusion_max_stage_timeout + 1},
    ],
)
def test_preset_stage_rejects_invalid_bounds(stage):
    with pytest.raises(ValidationError):
        PresetStage(**stage)
