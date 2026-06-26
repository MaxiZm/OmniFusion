import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.store.presets import get_preset, save_preset
from omnifusion.store.providers import save_provider


def preset(name="plugins"):
    stage = PresetStage(max_tokens=64, timeout=5)
    return Preset(
        name=name,
        strategy="B",
        panel_models=["stored-panel-a", "stored-panel-b"],
        panel=stage,
        judge_model="stored-judge",
        judge=stage,
        final_model="stored-final",
        final=stage,
    )


def test_chat_request_accepts_bounded_plugins_contract():
    from omnifusion.api.schemas import ChatCompletionRequest

    body = ChatCompletionRequest(
        model="openrouter/fusion",
        messages=[{"role": "user", "content": "hi"}],
        plugins={
            "analysis_models": ["panel-a", "panel-b"],
            "synthesis_model": "final-a",
            "web": True,
            "max_panel": 1,
        },
    )

    assert body.plugins.analysis_models == ["panel-a", "panel-b"]
    assert body.plugins.synthesis_model == "final-a"
    assert body.plugins.web is True
    assert body.plugins.max_panel == 1

    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="openrouter/fusion",
            messages=[{"role": "user", "content": "hi"}],
            plugins={"analysis_models": ["panel-a"], "unknown": True},
        )


def test_search_provider_adapters_are_swappable():
    from omnifusion.tools.search import (
        BraveSearchProvider,
        SearXNGSearchProvider,
        TavilySearchProvider,
    )

    def searx_transport(url, headers, body):
        assert "format=json" in url
        assert body is None
        return {
            "results": [
                {
                    "title": "SearX result",
                    "url": "https://example.com/searx",
                    "content": "from searx",
                }
            ]
        }

    def tavily_transport(url, headers, body):
        assert url == "https://api.tavily.com/search"
        assert headers["authorization"] == "Bearer tavily-key"
        assert json.loads(body.decode("utf-8"))["query"] == "query"
        return {
            "results": [
                {
                    "title": "Tavily result",
                    "url": "https://example.com/tavily",
                    "content": "from tavily",
                }
            ]
        }

    def brave_transport(url, headers, body):
        assert "q=query" in url
        assert headers["x-subscription-token"] == "brave-key"
        assert body is None
        return {
            "web": {
                "results": [
                    {
                        "title": "Brave result",
                        "url": "https://example.com/brave",
                        "description": "from brave",
                    }
                ]
            }
        }

    assert SearXNGSearchProvider(
        base_url="https://search.example",
        transport=searx_transport,
    ).search("query", max_results=1)[0].source == "searxng"
    assert TavilySearchProvider(
        api_key="tavily-key",
        transport=tavily_transport,
    ).search("query", max_results=1)[0].url == "https://example.com/tavily"
    assert BraveSearchProvider(
        api_key="brave-key",
        transport=brave_transport,
    ).search("query", max_results=1)[0].snippet == "from brave"


@pytest.mark.asyncio
async def test_plugins_override_preset_for_request_without_persisting(tmp_path, monkeypatch):
    import omnifusion.api.chat as chat_mod
    from omnifusion.main import app

    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "m5_plugins_override.db")
    settings.omnifusion_api_keys = ["plugins-key"]

    captured = {}

    async def fake_run_fusion(run_id, runtime_preset, body, key_hash):
        captured["panel_models"] = list(runtime_preset.panel_models)
        captured["final_model"] = runtime_preset.final_model
        captured["plugins"] = body.plugins.model_dump(exclude_none=True)
        return {
            "id": "chatcmpl-plugins",
            "object": "chat.completion",
            "created": 1,
            "model": f"fusion/{runtime_preset.name}",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    try:
        await init_db()
        await save_provider(
            "default",
            "openai",
            "test-key",
            models=["panel-a", "panel-b", "final-b"],
        )
        await save_preset(preset("plugins"))
        monkeypatch.setattr(chat_mod, "run_fusion", fake_run_fusion)

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer plugins-key"},
                json={
                    "model": "fusion/plugins",
                    "messages": [{"role": "user", "content": "hello"}],
                    "plugins": {
                        "analysis_models": ["panel-a", "panel-b"],
                        "synthesis_model": "final-b",
                        "max_panel": 1,
                        "web": True,
                    },
                },
            )

        stored = await get_preset("plugins")
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys

    assert response.status_code == 200
    assert captured["panel_models"] == ["panel-a"]
    assert captured["final_model"] == "final-b"
    assert captured["plugins"] == {
        "analysis_models": ["panel-a", "panel-b"],
        "synthesis_model": "final-b",
        "web": True,
        "max_panel": 1,
    }
    assert stored.panel_models == ["stored-panel-a", "stored-panel-b"]
    assert stored.final_model == "stored-final"


@pytest.mark.asyncio
async def test_plugins_analysis_models_must_be_registered(tmp_path, monkeypatch):
    import omnifusion.api.chat as chat_mod
    from omnifusion.main import app

    old_db = settings.db_path
    old_keys = settings.omnifusion_api_keys
    settings.db_path = str(tmp_path / "m5_plugins_registered.db")
    settings.omnifusion_api_keys = ["plugins-key"]

    async def fail_run_fusion(*args, **kwargs):
        raise AssertionError("run_fusion should not be called for unregistered plugins")

    try:
        await init_db()
        await save_provider("default", "openai", "test-key", models=["registered-model"])
        await save_preset(preset("strict"))
        monkeypatch.setattr(chat_mod, "run_fusion", fail_run_fusion)

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer plugins-key"},
                json={
                    "model": "fusion/strict",
                    "messages": [{"role": "user", "content": "hello"}],
                    "plugins": {"analysis_models": ["unregistered-model"]},
                },
            )
    finally:
        settings.db_path = old_db
        settings.omnifusion_api_keys = old_keys

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "plugin_model_not_registered"


@pytest.mark.asyncio
async def test_recursive_fusion_models_are_blocked_before_litellm(monkeypatch):
    import litellm
    from omnifusion.api.errors import OmniFusionError
    from omnifusion.llm.client import llm_client

    async def fail_acompletion(**kwargs):
        raise AssertionError("recursive fusion guard should run before LiteLLM")

    monkeypatch.setattr(litellm, "acompletion", fail_acompletion)

    with pytest.raises(OmniFusionError) as exc:
        await llm_client.acompletion(
            provider_id="default",
            model="openrouter/fusion",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert exc.value.status_code == 400
    assert exc.value.code == "recursive_fusion_model"


@pytest.mark.asyncio
async def test_plugins_override_routes_to_resolved_provider(tmp_path, monkeypatch):
    """[P2] A plugin model registered under a non-default provider must be routed to
    that provider in the models pool, not rewritten to 'default'."""
    import omnifusion.fusion.plugins as plugins_mod
    from omnifusion.api.schemas import FusionPlugins
    from omnifusion.fusion.types import Preset, PresetStage

    async def fake_resolve(model):
        return {"id": "prov-custom"} if model == "special-model" else None

    monkeypatch.setattr(plugins_mod, "resolve_registered_provider_for_model", fake_resolve)

    stage = PresetStage(max_tokens=16, timeout=5)
    preset = Preset(
        name="p",
        strategy="B",
        panel_models=["panel-a"],
        panel=stage,
        judge_model="judge-a",
        judge=stage,
        final_model="final-a",
        final=stage,
    )
    updated = await plugins_mod.apply_plugins_override(
        preset, FusionPlugins(analysis_models=["special-model"])
    )
    panel_entries = [m for m in updated.models if m.role == "panel"]
    assert panel_entries[0].provider_id == "prov-custom"
    # The pool entry is honored by provider_id_for too.
    assert updated.provider_id_for("special-model", "panel") == "prov-custom"
