"""M5 server-side web grounding ("web on") wired into the fusion panel."""
import os

import pytest
from pydantic import BaseModel

from omnifusion.api.schemas import ChatCompletionRequest, ChatMessage, FusionPlugins
from omnifusion.fusion import web_grounding as wg
from omnifusion.fusion.orchestrator import run_fusion
from omnifusion.fusion.plugins import apply_plugins_override
from omnifusion.fusion.types import Preset, PresetStage
from omnifusion.settings import settings
from omnifusion.store.db import init_db
from omnifusion.store.runs import get_trace
from omnifusion.tools.search import SearchResult
from omnifusion.tools.web import WebFetchResult
from omnifusion.budget.ledger import initialize_request_budget


class _FakeSearch:
    def __init__(self, results):
        self._results = results
        self.queries = []

    def search(self, query, max_results=5):
        self.queries.append((query, max_results))
        return self._results[:max_results]


class _FakeFetcher:
    def __init__(self, by_url):
        self._by_url = by_url
        self.fetched = []

    def fetch(self, url):
        self.fetched.append(url)
        if url not in self._by_url:
            raise ValueError("blocked")
        return self._by_url[url]


def _fetch_result(url, excerpt):
    return WebFetchResult(
        url=url,
        final_url=url,
        mime_type="text/html",
        content_hash="sha256:deadbeef",
        excerpt=excerpt,
        truncated=False,
        fenced_content="(fenced)",
        trace_metadata={"url": url, "excerpt": excerpt},
    )


def test_latest_user_text_handles_dicts_objects_and_parts():
    assert wg.latest_user_text([{"role": "user", "content": "hello"}]) == "hello"
    assert (
        wg.latest_user_text(
            [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "mid"},
                {"role": "user", "content": "newest"},
            ]
        )
        == "newest"
    )
    parts = [{"role": "user", "content": [{"type": "text", "text": "part-a"}]}]
    assert wg.latest_user_text(parts) == "part-a"


def test_inject_grounding_after_leading_system_message():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    out = wg.inject_grounding(messages, "GROUNDING")
    assert out[0]["content"] == "sys"
    assert out[1] == {"role": "system", "content": "GROUNDING"}
    assert out[2]["content"] == "q"
    # Original list is not mutated.
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_gather_web_context_fences_attributes_and_bounds_persistence(tmp_path):
    old_db = settings.db_path
    settings.db_path = str(tmp_path / "wg.db")
    try:
        await init_db()
        await initialize_request_budget("wg-run", None)

        search = _FakeSearch(
            [
                SearchResult(title="T1", url="https://a.example/1", snippet="snip-a", source="fake"),
                SearchResult(title="T2", url="https://b.example/2", snippet="snip-b", source="fake"),
            ]
        )
        fetcher = _FakeFetcher(
            {"https://a.example/1": _fetch_result("https://a.example/1", "FETCHED-EXCERPT-A")}
        )

        ctx = await wg.gather_web_context(
            "wg-run",
            "what is x",
            search_provider=search,
            fetcher=fetcher,
            max_results=2,
            fetch_top=1,
        )
    finally:
        if os.path.exists(settings.db_path):
            os.remove(settings.db_path)
        settings.db_path = old_db

    assert ctx.has_grounding
    # Untrusted-data framing + nonce fences present.
    assert "UNTRUSTED" in ctx.grounding_text
    assert "START OF WEB_SOURCE 1" in ctx.grounding_text
    # Fetched excerpt used for the top result; snippet used for the rest.
    assert "FETCHED-EXCERPT-A" in ctx.grounding_text
    assert "snip-b" in ctx.grounding_text
    assert search.queries == [("what is x", 2)]
    assert fetcher.fetched == ["https://a.example/1"]

    # Invariant 6: only bounded metadata is retained, never a full page body.
    fetched_source = next(s for s in ctx.sources if s.get("fetched"))
    assert fetched_source["content_hash"] == "sha256:deadbeef"
    assert fetched_source["excerpt"] == "FETCHED-EXCERPT-A"
    assert "body" not in fetched_source
    assert "fenced_content" not in fetched_source


@pytest.mark.asyncio
async def test_gather_web_context_degrades_when_search_fails(tmp_path):
    old_db = settings.db_path
    settings.db_path = str(tmp_path / "wg2.db")

    class _Boom:
        def search(self, query, max_results=5):
            raise RuntimeError("provider down")

    try:
        await init_db()
        await initialize_request_budget("wg-run-2", None)
        ctx = await wg.gather_web_context(
            "wg-run-2", "q", search_provider=_Boom(), fetcher=_FakeFetcher({})
        )
    finally:
        if os.path.exists(settings.db_path):
            os.remove(settings.db_path)
        settings.db_path = old_db

    assert not ctx.has_grounding
    assert ctx.sources and ctx.sources[0]["error"] == "provider down"


@pytest.mark.asyncio
async def test_plugins_web_overrides_preset_web_enabled():
    stage = PresetStage(max_tokens=64, timeout=5)
    preset = Preset(
        name="general",
        strategy="B",
        panel_models=["panel-a"],
        panel=stage,
        judge_model="judge-a",
        judge=stage,
        final_model="final-a",
        final=stage,
        web_enabled=False,
    )
    enabled = await apply_plugins_override(preset, FusionPlugins(web=True))
    assert enabled.web_enabled is True

    disabled = await apply_plugins_override(
        preset.model_copy(update={"web_enabled": True}), FusionPlugins(web=False)
    )
    assert disabled.web_enabled is False

    # When plugins omit `web`, the preset value is preserved.
    untouched = await apply_plugins_override(
        preset.model_copy(update={"web_enabled": True}), FusionPlugins(max_panel=1)
    )
    assert untouched.web_enabled is True


class _MockMessage:
    def __init__(self, content):
        self.content = content


class _MockChoice:
    def __init__(self, content):
        self.message = _MockMessage(content)
        self.finish_reason = "stop"


class _MockUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class _MockResponse:
    def __init__(self, content, prompt_tokens=1, completion_tokens=1):
        self.choices = [_MockChoice(content)]
        self.usage = _MockUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


@pytest.mark.asyncio
async def test_web_enabled_run_injects_grounding_into_panel_and_traces_sources(
    tmp_path, monkeypatch
):
    import omnifusion.llm.client as client_mod

    old_db = settings.db_path
    settings.db_path = str(tmp_path / "wg_e2e.db")

    panel_seen = {}

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        if model == "panel-a":
            panel_seen["messages"] = messages
            return _MockResponse("panel answer")
        if model == "judge-a":
            return _MockResponse("{\"consensus\": \"ok\"}")
        if model == "final-a":
            return _MockResponse("final answer")
        raise AssertionError(f"unexpected model {model}")

    # Inject fake web tools into the grounding module.
    search = _FakeSearch(
        [SearchResult(title="Doc", url="https://docs.example/x", snippet="SNIP", source="fake")]
    )
    fetcher = _FakeFetcher(
        {"https://docs.example/x": _fetch_result("https://docs.example/x", "GROUNDED-CONTENT")}
    )
    monkeypatch.setattr(wg, "build_search_provider", lambda *a, **k: search)
    monkeypatch.setattr(wg, "WebFetcher", lambda *a, **k: fetcher)

    try:
        await init_db()
        monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
        stage = PresetStage(max_tokens=64, timeout=5)
        preset = Preset(
            name="general",
            strategy="B",
            panel_models=["panel-a"],
            panel=stage,
            judge_model="judge-a",
            judge=stage,
            final_model="final-a",
            final=stage,
            web_enabled=True,
        )
        request = ChatCompletionRequest(
            model="fusion/general",
            messages=[ChatMessage(role="user", content="explain x")],
            stream=False,
            store=True,
        )
        response = await run_fusion("wg-e2e", preset, request, "keyhash")
        trace = await get_trace("wg-e2e", "keyhash")
    finally:
        if os.path.exists(settings.db_path):
            os.remove(settings.db_path)
        settings.db_path = old_db

    # The panel saw the fenced, grounded web content as a system message.
    panel_system = [m for m in panel_seen["messages"] if m.get("role") == "system"]
    assert any("GROUNDED-CONTENT" in (m.get("content") or "") for m in panel_system)
    assert any("UNTRUSTED" in (m.get("content") or "") for m in panel_system)

    assert response["choices"][0]["message"]["content"] == "final answer"
    # Trace carries bounded web-source attribution.
    assert trace is not None
    web_sources = trace.metadata.get("web_sources")
    assert web_sources and web_sources[0]["url"] == "https://docs.example/x"
    assert web_sources[0]["content_hash"] == "sha256:deadbeef"


@pytest.mark.asyncio
async def test_web_enabled_tool_request_injects_grounding_into_tool_panel_and_trace(
    tmp_path, monkeypatch
):
    import omnifusion.llm.client as client_mod

    old_db = settings.db_path
    settings.db_path = str(tmp_path / "wg_tool_e2e.db")

    panel_seen = {}

    async def fake_acompletion(provider_id, model, messages, **kwargs):
        if model == "panel-a":
            panel_seen["messages"] = messages
            return _MockResponse("panel answer from grounded web")
        if model == "judge-a":
            return _MockResponse("{\"consensus\": \"ok\"}")
        if model == "final-a":
            return _MockResponse("final answer")
        raise AssertionError(f"unexpected model {model}")

    search = _FakeSearch(
        [SearchResult(title="Doc", url="https://docs.example/tool", snippet="SNIP", source="fake")]
    )
    fetcher = _FakeFetcher(
        {"https://docs.example/tool": _fetch_result("https://docs.example/tool", "TOOL-GROUNDED-CONTENT")}
    )
    monkeypatch.setattr(wg, "build_search_provider", lambda *a, **k: search)
    monkeypatch.setattr(wg, "WebFetcher", lambda *a, **k: fetcher)

    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }

    try:
        await init_db()
        monkeypatch.setattr(client_mod.llm_client, "acompletion", fake_acompletion)
        stage = PresetStage(max_tokens=64, timeout=5)
        preset = Preset(
            name="tool-web",
            strategy="B",
            panel_models=["panel-a"],
            panel=stage,
            judge_model="judge-a",
            judge=stage,
            final_model="final-a",
            final=stage,
            web_enabled=True,
        )
        request = ChatCompletionRequest(
            model="fusion/tool-web",
            messages=[ChatMessage(role="user", content="explain x")],
            tools=[weather_tool],
            tool_choice="auto",
            stream=False,
            store=True,
        )
        response = await run_fusion("wg-tool-e2e", preset, request, "keyhash")
        trace = await get_trace("wg-tool-e2e", "keyhash")
    finally:
        if os.path.exists(settings.db_path):
            os.remove(settings.db_path)
        settings.db_path = old_db

    panel_system = [m for m in panel_seen["messages"] if m.get("role") == "system"]
    assert any("TOOL-GROUNDED-CONTENT" in (m.get("content") or "") for m in panel_system)
    assert any("UNTRUSTED" in (m.get("content") or "") for m in panel_system)

    assert response["choices"][0]["message"]["content"] == "final answer"
    assert trace is not None
    web_sources = trace.metadata.get("web_sources")
    assert web_sources and web_sources[0]["url"] == "https://docs.example/tool"
    assert web_sources[0]["content_hash"] == "sha256:deadbeef"
