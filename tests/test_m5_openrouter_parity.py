import json
import os

import pytest
from pydantic import BaseModel

from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage
from omnifusion.fusion.orchestrator import run_fusion
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.store.runs import get_trace


class MockMessage:
    def __init__(self, content):
        self.content = content


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)
        self.finish_reason = "stop"


class MockUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class MockResponse:
    def __init__(self, content, prompt_tokens, completion_tokens):
        self.choices = [MockChoice(content)]
        self.usage = MockUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


@pytest.mark.asyncio
async def test_m5_parity_flow_preserves_structured_judge_usage_and_trace(
    tmp_path, monkeypatch
):
    import omnifusion.llm.client as client_mod

    old_db = settings.db_path
    settings.db_path = str(tmp_path / "m5_parity.db")
    calls = []

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        if model == "panel-a":
            return MockResponse("panel answer", 3, 4)
        if model == "judge-model":
            assert kwargs["temperature"] == 0
            return MockResponse(
                json.dumps(
                    {
                        "consensus": "shared answer",
                        "contradictions": "conflict to resolve",
                        "partial_coverage": "missing tests",
                        "unique_insights": {"MODEL_A": ["edge case"]},
                        "blind_spots": "deployment risk",
                        "model_strengths": {"MODEL_A": "clear"},
                        "synthesis_plan": "merge and qualify",
                    }
                ),
                5,
                6,
            )
        if model == "final-model":
            final_prompt = messages[0]["content"]
            assert "conflict to resolve" in final_prompt
            assert "missing tests" in final_prompt
            assert "deployment risk" in final_prompt
            assert "merge and qualify" in final_prompt
            return MockResponse("final answer", 7, 8)
        raise AssertionError(f"unexpected model call: {model}")

    try:
        await init_db()
        monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
        stage = PresetStage(max_tokens=128, timeout=5)
        preset = Preset(
            name="general",
            strategy="B",
            panel_models=["panel-a"],
            panel=stage,
            judge_model="judge-model",
            judge=stage,
            final_model="final-model",
            final=stage,
            min_panel_success=1,
        )
        request = ChatCompletionRequest(
            model="fusion/general",
            messages=[ChatMessage(role="user", content="question")],
            stream=False,
            store=True,
        )

        response = await run_fusion("m5-parity-run", preset, request, "keyhash")
        trace = await get_trace("m5-parity-run", "keyhash")
    finally:
        if os.path.exists(settings.db_path):
            os.remove(settings.db_path)
        settings.db_path = old_db

    assert [call["model"] for call in calls] == ["panel-a", "judge-model", "final-model"]
    assert response["model"] == "fusion/general"
    assert response["choices"][0]["message"]["content"] == "final answer"
    assert response["usage"] == {
        "prompt_tokens": 15,
        "completion_tokens": 18,
        "total_tokens": 33,
    }
    assert trace is not None
    assert trace.judge_analysis.consensus == "shared answer"
    assert trace.judge_analysis.contradictions == "conflict to resolve"
    assert trace.judge_analysis.unique_insights == {"MODEL_A": ["edge case"]}
    assert trace.final_answer == "final answer"
