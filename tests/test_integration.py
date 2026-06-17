import pytest
import os
import httpx
from unittest.mock import patch
from pydantic import BaseModel

from omnifusion.store.db import init_db
from omnifusion.store.presets import save_preset
from omnifusion.store.providers import save_provider
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.fusion.orchestrator import run_fusion
from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage


# Mock response classes matching test_orchestrator.py
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


# Mark this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def setup_db():
    from omnifusion.settings import settings

    old_db = settings.db_path
    settings.db_path = "test_integration.db"

    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)

    yield

    if os.path.exists(settings.db_path):
        try:
            os.remove(settings.db_path)
        except Exception:
            pass
    settings.db_path = old_db


async def get_ollama_models(url: str) -> list:
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{url}/api/tags", timeout=1.0)
            if res.status_code == 200:
                data = res.json()
                return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


@pytest.mark.asyncio
async def test_ollama_integration():
    ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model_name = "qwen2.5:0.5b"

    # Check if we should run real or mocked
    run_real = os.getenv("OLLAMA_INTEGRATION_TEST") == "1"

    if run_real:
        # Check if Ollama is running and has model
        models = await get_ollama_models(ollama_url)
        if not models:
            pytest.skip(
                f"Ollama is not running at {ollama_url}. Skipping integration test."
            )
        has_model = any(model_name in m or m.startswith(model_name) for m in models)
        if not has_model:
            pytest.skip(
                f"Ollama does not have model '{model_name}' pulled. Skipping integration test."
            )

        await init_db()
        # 1. Register Ollama provider
        await save_provider(
            provider_id="ollama-local",
            p_type="ollama",
            plain_key="",
            base_url=ollama_url,
            api_key_ref=None,
            models=[model_name],
        )
        # 2. Register preset using Ollama
        preset = Preset(
            name="local-ollama-fusion",
            strategy="B",
            panel_models=[model_name, model_name],
            panel=PresetStage(max_tokens=30, timeout=15),
            judge_model=model_name,
            judge=PresetStage(max_tokens=50, timeout=15),
            final_model=model_name,
            final=PresetStage(max_tokens=50, timeout=15),
            cost_ceiling=1.0,
            min_panel_success=1,
        )
        await save_preset(preset)

        req = ChatCompletionRequest(
            model="fusion/local-ollama-fusion",
            messages=[ChatMessage(role="user", content="What is 2+2?")],
            stream=False,
            store=True,
        )
        res = await run_fusion("test-run-ollama", preset, req, "test-key-hash")
        assert res is not None
        assert len(res["choices"]) > 0
        assert res["usage"]["prompt_tokens"] > 0
    else:
        # Run mocked version
        with patch("omnifusion.llm.client.llm_client.acompletion") as mock_acompletion:
            call_count = 0

            def side_effect(provider_id, model, messages, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count in (1, 2):
                    return MockResponse("Ollama local answer", 10, 10)
                elif call_count == 3:
                    return MockResponse(
                        '{"consensus": "agreed", "recommended_final_answer_plan": "plan"}',
                        20,
                        20,
                    )
                elif call_count == 4:
                    return MockResponse("Synthesized answer", 30, 30)
                raise ValueError(f"Unexpected acompletion call: {call_count}")

            mock_acompletion.side_effect = side_effect

            await init_db()
            await save_provider(
                provider_id="ollama-local",
                p_type="ollama",
                plain_key="",
                base_url=ollama_url,
                api_key_ref=None,
                models=[model_name],
            )
            preset = Preset(
                name="local-ollama-fusion",
                strategy="B",
                panel_models=[model_name, model_name],
                panel=PresetStage(max_tokens=30, timeout=15),
                judge_model=model_name,
                judge=PresetStage(max_tokens=50, timeout=15),
                final_model=model_name,
                final=PresetStage(max_tokens=50, timeout=15),
                cost_ceiling=1.0,
                min_panel_success=1,
            )
            await save_preset(preset)
            req = ChatCompletionRequest(
                model="fusion/local-ollama-fusion",
                messages=[ChatMessage(role="user", content="What is 2+2?")],
                stream=False,
                store=True,
            )
            res = await run_fusion("test-run-ollama", preset, req, "test-key-hash")
            assert res is not None
            assert len(res["choices"]) > 0
            assert res["choices"][0]["message"]["content"] == "Synthesized answer"
            assert res["usage"]["prompt_tokens"] > 0
