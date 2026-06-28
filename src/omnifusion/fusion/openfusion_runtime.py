from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import Counter
from typing import Any

from fastapi.responses import StreamingResponse

from omnifusion.api.errors import InsufficientPanelError, OmniFusionError
from omnifusion.api.normalize import generation_passthrough_kwargs
from omnifusion.api.schemas import ChatCompletionRequest, ToolDefinition
from omnifusion.api.sse import wants_usage
from omnifusion.budget.ledger import initialize_request_budget
from omnifusion.fusion.judge import extract_json_from_text, run_judge
from omnifusion.fusion.panel import run_panel, run_panelist
from omnifusion.fusion.runtime.executor import BudgetedExecutor
from omnifusion.fusion.runtime.response import ResponseShaper
from omnifusion.fusion.runtime.streaming import StreamingAdapter, normalize_finish_reason
from omnifusion.fusion.synth import run_synthesis
from omnifusion.fusion.types import (
    FusionTrace,
    JudgeAnalysis,
    PanelResult,
    Preset,
    StageEvent,
    build_stage_events,
    trace_metadata_for_preset,
)
from omnifusion.providers.pricing import estimate_call_cost
from omnifusion.store.providers import resolve_registered_provider_for_model
from omnifusion.store.runs import save_trace


OPENROUTER_SERVER_TOOL_TYPES = {"openrouter:web_search", "openrouter:web_fetch"}

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def server_web_tools_requested(tools: list[ToolDefinition] | None) -> bool:
    if not tools:
        return False
    return all(tool.type in OPENROUTER_SERVER_TOOL_TYPES for tool in tools)


def mixed_server_and_function_tools(tools: list[ToolDefinition] | None) -> bool:
    if not tools:
        return False
    has_server = any(tool.type in OPENROUTER_SERVER_TOOL_TYPES for tool in tools)
    has_other = any(tool.type not in OPENROUTER_SERVER_TOOL_TYPES for tool in tools)
    return has_server and has_other


def needs_hybrid_runtime(preset: Preset) -> bool:
    return any(
        (
            preset.fusion_mode != "panel",
            preset.aggregator != "judge",
            preset.router.enabled,
            preset.analysis_emit.enabled,
            preset.response_cache.enabled,
        )
    )


def _message_dicts(request: ChatCompletionRequest) -> list[dict[str, Any]]:
    return [message.model_dump(exclude_none=True) for message in request.messages]


def _response_text(response: Any) -> str:
    if not getattr(response, "choices", None):
        return ""
    return getattr(response.choices[0].message, "content", "") or ""


def _usage_tokens(usage: Any) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return int(usage.get("prompt_tokens", 0) or 0), int(
            usage.get("completion_tokens", 0) or 0
        )
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(
        getattr(usage, "completion_tokens", 0) or 0
    )


def _response_usage(response: Any) -> tuple[int, int]:
    return _usage_tokens(getattr(response, "usage", None))


def _panel_usage(panel_results: list[PanelResult]) -> tuple[int, int]:
    prompt = completion = 0
    for result in panel_results:
        p, c = _usage_tokens(getattr(result, "usage", None))
        prompt += p
        completion += c
    return prompt, completion


def _ok_panels(panel_results: list[PanelResult]) -> list[PanelResult]:
    return [result for result in panel_results if result.status == "ok" and result.content]


def _latest_user_text(request: ChatCompletionRequest) -> str:
    from omnifusion.fusion.web_grounding import latest_user_text

    return latest_user_text(request.messages)


def _cache_key(preset: Preset, request: ChatCompletionRequest) -> str:
    payload = {
        "messages": _message_dicts(request),
        "model": request.model,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "fusion": {
            "mode": preset.fusion_mode,
            "aggregator": preset.aggregator,
            "panel_models": preset.panel_models,
            "judge_model": preset.judge_model,
            "final_model": preset.final_model,
            "web_enabled": preset.web_enabled,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cache_get(preset: Preset, request: ChatCompletionRequest) -> dict[str, Any] | None:
    if not preset.response_cache.enabled:
        return None
    entry = _CACHE.get(_cache_key(preset, request))
    if entry is None:
        return None
    expires_at, payload = entry
    if expires_at <= time.time():
        _CACHE.pop(_cache_key(preset, request), None)
        return None
    return payload


def _cache_set(preset: Preset, request: ChatCompletionRequest, payload: dict[str, Any]) -> None:
    if not preset.response_cache.enabled:
        return
    if len(_CACHE) >= preset.response_cache.max_entries:
        oldest_key = min(_CACHE.items(), key=lambda item: item[1][0])[0]
        _CACHE.pop(oldest_key, None)
    _CACHE[_cache_key(preset, request)] = (
        time.time() + preset.response_cache.ttl_seconds,
        payload,
    )


def _synthetic_stream(
    *,
    model: str,
    content: str,
    usage: dict[str, int],
    include_usage: bool,
):
    adapter = StreamingAdapter(model)
    created = int(time.time())
    response_id = f"chatcmpl-{hashlib.sha1(f'{model}:{created}:{content}'.encode()).hexdigest()[:24]}"

    def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    async def gen():
        yield adapter.raw_chunk_sse(chunk({"role": "assistant"}))
        if content:
            yield adapter.raw_chunk_sse(chunk({"content": content}))
        yield adapter.raw_chunk_sse(chunk({}, "stop"))
        if include_usage:
            yield adapter.usage_sse(
                int(usage.get("prompt_tokens", 0) or 0),
                int(usage.get("completion_tokens", 0) or 0),
            )
        yield adapter.done_sse()

    return StreamingResponse(gen(), media_type="text/event-stream")


async def _save_cache_trace(
    run_id: str,
    preset: Preset,
    request: ChatCompletionRequest,
    key_hash: str,
    payload: dict[str, Any],
):
    content = payload["choices"][0]["message"].get("content")
    trace = FusionTrace(
        run_id=run_id,
        preset=preset.name,
        cost_usd=0.0,
        wall_ms=0,
        degraded=False,
        panel_results=[],
        judge_analysis=None,
        final_answer=content,
        stage_events=[
            StageEvent(
                stage="cache",
                role="final",
                status="ok",
                metadata={"cache": "hit", "source": "openfusion_response_cache"},
            )
        ],
        metadata={**trace_metadata_for_preset(preset), "response_cache": {"hit": True}},
    )
    await save_trace(trace, request.store, key_hash)


def _heuristic_route(request: ChatCompletionRequest, preset: Preset) -> str:
    router = preset.router
    if router.mode == "always":
        return "fuse"
    if router.mode == "never":
        return "solo"
    text = _latest_user_text(request)
    lowered = text.lower()
    if "```" in text:
        return "fuse"
    if any(keyword.lower() in lowered for keyword in router.fuse_keywords):
        return "fuse"
    if len(text) >= router.min_chars:
        return "fuse"
    return "solo"


def _prompt_tier(text: str) -> str:
    lowered = text.lower()
    strong_keywords = (
        "analyze",
        "analyse",
        "compare",
        "evaluate",
        "design",
        "research",
        "explain why",
        "trade-off",
        "tradeoff",
        "prove",
        "debug",
        "architecture",
    )
    if "```" in text or len(text) >= 600 or any(k in lowered for k in strong_keywords):
        return "strong"
    if len(text) >= 200:
        return "balanced"
    return "fast"


def _select_route_model(request: ChatCompletionRequest, preset: Preset) -> str:
    route_models = preset.router.route_models
    if not route_models:
        return preset.final_model
    want = _prompt_tier(_latest_user_text(request))
    order = {
        "fast": ("fast", "balanced", "strong"),
        "balanced": ("balanced", "strong", "fast"),
        "strong": ("strong", "balanced", "fast"),
    }[want]
    for tier in order:
        for candidate in route_models:
            if candidate.tier == tier:
                return candidate.model
    return route_models[0].model


async def _route_decision(
    run_id: str,
    preset: Preset,
    request: ChatCompletionRequest,
    executor: BudgetedExecutor,
) -> tuple[str, str | None, StageEvent | None]:
    if not preset.router.enabled:
        return "fuse", None, None

    if preset.router.fuse_only_with_tools and not preset.web_enabled:
        return (
            "solo",
            _select_route_model(request, preset),
            StageEvent(
                stage="router",
                status="ok",
                metadata={"decision": "solo", "reason": "fuse_only_with_tools"},
            ),
        )

    if preset.router.mode == "model" and preset.router.classifier_model:
        messages = [
            {
                "role": "system",
                "content": (
                    "Reply with exactly FUSE or SOLO. FUSE for hard, open-ended, "
                    "research, design, debugging, or analytical requests. SOLO for "
                    "simple factual or trivial requests."
                ),
            },
            {"role": "user", "content": _latest_user_text(request)[:4000]},
        ]
        try:
            response = await executor.call(
                "router",
                provider_id=preset.router.classifier_provider_id
                or preset.provider_id_for(preset.router.classifier_model, "judge"),
                model=preset.router.classifier_model,
                messages=messages,
                max_tokens=preset.router.classifier_max_tokens,
                timeout=preset.judge.timeout,
                temperature=0,
            )
            text = _response_text(response).upper()
            decision = "solo" if "SOLO" in text and "FUSE" not in text else "fuse"
            prompt_tokens, completion_tokens = _response_usage(response)
            cost = float(getattr(response, "_omnifusion_cost_usd", 0.0) or 0.0)
            return (
                decision,
                _select_route_model(request, preset) if decision == "solo" else None,
                StageEvent(
                    stage="router",
                    role="judge",
                    provider_id=preset.router.classifier_provider_id
                    or preset.provider_id_for(preset.router.classifier_model, "judge"),
                    model=preset.router.classifier_model,
                    status="ok",
                    tokens={"prompt": prompt_tokens, "completion": completion_tokens},
                    cost_usd=cost,
                    metadata={"decision": decision, "mode": "model"},
                ),
            )
        except Exception:
            decision = _heuristic_route(request, preset)
            return (
                decision,
                _select_route_model(request, preset) if decision == "solo" else None,
                StageEvent(
                    stage="router",
                    status="degraded",
                    metadata={"decision": decision, "mode": "heuristic_fallback"},
                ),
            )

    decision = _heuristic_route(request, preset)
    return (
        decision,
        _select_route_model(request, preset) if decision == "solo" else None,
        StageEvent(
            stage="router",
            status="ok",
            metadata={"decision": decision, "mode": preset.router.mode},
        ),
    )


async def _run_solo(
    run_id: str,
    preset: Preset,
    request: ChatCompletionRequest,
    key_hash: str,
    model: str,
    route_event: StageEvent | None,
):
    provider = await resolve_registered_provider_for_model(model)
    if provider is None:
        raise OmniFusionError(
            f"Router SOLO model '{model}' does not resolve to a registered provider.",
            status_code=400,
            type_="invalid_request_error",
            code="router_model_not_registered",
        )

    start = time.time()
    executor = BudgetedExecutor(run_id)
    messages = _message_dicts(request)
    max_tokens = min(
        request.max_tokens or preset.final.max_tokens,
        preset.final.max_tokens,
    )
    kwargs = {
        "timeout": preset.final.timeout,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stop": request.stop,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs.update(generation_passthrough_kwargs(request))

    if request.stream:
        stream = await executor.stream(
            "router/solo",
            provider_id=provider["id"],
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )
        adapter = StreamingAdapter(request.model)

        async def gen():
            completion_text = ""
            usage = None
            stream_error = None
            try:
                async for chunk in stream:
                    if getattr(chunk, "choices", None):
                        delta = getattr(chunk.choices[0], "delta", None)
                        content = getattr(delta, "content", None)
                        if content:
                            completion_text += content
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                    yield adapter.chunk_sse(chunk)
                if wants_usage(request):
                    prompt_tokens, completion_tokens = _usage_tokens(usage)
                    yield adapter.usage_sse(prompt_tokens, completion_tokens)
                yield adapter.done_sse()
            except Exception as exc:
                stream_error = exc
            finally:
                wall_ms = int((time.time() - start) * 1000)
                prompt_tokens, completion_tokens = _usage_tokens(usage)
                trace = FusionTrace(
                    run_id=run_id,
                    preset=preset.name,
                    cost_usd=float(getattr(stream, "cost_usd", 0.0) or 0.0),
                    wall_ms=wall_ms,
                    degraded=stream_error is not None,
                    panel_results=[],
                    judge_analysis=None,
                    final_answer=completion_text,
                    stage_events=[
                        *([route_event] if route_event else []),
                        StageEvent(
                            stage="router/solo",
                            role="final",
                            provider_id=provider["id"],
                            model=model,
                            status="error" if stream_error else "ok",
                            tokens={
                                "prompt": prompt_tokens,
                                "completion": completion_tokens,
                            },
                            cost_usd=float(getattr(stream, "cost_usd", 0.0) or 0.0),
                            wall_ms=wall_ms,
                        ),
                    ],
                    metadata={
                        **trace_metadata_for_preset(preset),
                        "router": {"decision": "solo", "model": model},
                    },
                )
                await save_trace(trace, request.store, key_hash)
                if stream_error is not None:
                    raise stream_error

        return StreamingResponse(gen(), media_type="text/event-stream")

    response = await executor.call(
        "router/solo",
        provider_id=provider["id"],
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        **kwargs,
    )
    content = _response_text(response)
    prompt_tokens, completion_tokens = _response_usage(response)
    cost = float(getattr(response, "_omnifusion_cost_usd", 0.0) or 0.0)
    wall_ms = int((time.time() - start) * 1000)
    trace = FusionTrace(
        run_id=run_id,
        preset=preset.name,
        cost_usd=cost,
        wall_ms=wall_ms,
        degraded=False,
        panel_results=[],
        judge_analysis=None,
        final_answer=content,
        stage_events=[
            *([route_event] if route_event else []),
            StageEvent(
                stage="router/solo",
                role="final",
                provider_id=provider["id"],
                model=model,
                status="ok",
                tokens={"prompt": prompt_tokens, "completion": completion_tokens},
                cost_usd=cost,
                wall_ms=wall_ms,
            ),
        ],
        metadata={
            **trace_metadata_for_preset(preset),
            "router": {"decision": "solo", "model": model},
        },
    )
    await save_trace(trace, request.store, key_hash)
    return ResponseShaper.chat_completion(
        model=request.model,
        content=content,
        usage=ResponseShaper.usage_block(prompt_tokens, completion_tokens),
        finish_reason=normalize_finish_reason(
            getattr(response.choices[0], "finish_reason", "stop")
        ),
    )


async def _run_panel_for_mode(
    run_id: str,
    preset: Preset,
    messages: list[Any],
    request: ChatCompletionRequest,
) -> list[PanelResult]:
    extra = generation_passthrough_kwargs(request)
    if preset.fusion_mode == "panel":
        return await run_panel(
            run_id,
            preset,
            messages,
            min_success=preset.min_panel_success,
            extra_kwargs=extra,
        )

    if preset.fusion_mode == "self_fusion":
        model = preset.panel_models[0]
        temperatures = preset.self_fusion.temperature_spread or [request.temperature or 0.7]
        tasks = []
        for index in range(preset.self_fusion.n):
            kwargs = dict(extra)
            kwargs["temperature"] = temperatures[index % len(temperatures)]
            if preset.self_fusion.seed_offset and request.seed is not None:
                kwargs["seed"] = request.seed + index
            tasks.append(run_panelist(run_id, model, preset, messages, kwargs))
        results = await asyncio.gather(*tasks)
        ok_count = len(_ok_panels(results))
        if ok_count < min(preset.min_panel_success, preset.self_fusion.n):
            raise InsufficientPanelError(
                f"Self-fusion only got {ok_count} successes, needed {preset.min_panel_success}"
            )
        return list(results)

    panel_results = await run_panel(
        run_id,
        preset,
        messages,
        min_success=preset.min_panel_success,
        extra_kwargs=extra,
    )
    for round_index in range(preset.debate.rounds):
        ok = _ok_panels(panel_results)
        debate_context = "\n\n".join(
            f"{idx + 1}. {result.model}: {result.content}" for idx, result in enumerate(ok)
        )
        debate_messages = [
            {
                "role": "system",
                "content": (
                    "Revise your answer after reading the other panel drafts. "
                    "Keep what is correct, fix mistakes, and answer directly.\n\n"
                    f"Round {round_index + 1} drafts:\n{debate_context}"
                ),
            },
            *messages,
        ]
        panel_results = await run_panel(
            run_id,
            preset,
            debate_messages,
            min_success=preset.min_panel_success,
            extra_kwargs=extra,
        )
    return panel_results


def _aggregation_events(
    preset: Preset,
    panel_results: list[PanelResult],
    judge_analysis: JudgeAnalysis | None,
    final_answer: str,
    route_event: StageEvent | None,
    *,
    aggregator: str,
) -> list[StageEvent]:
    events = build_stage_events(
        preset,
        panel_results,
        judge_analysis,
        None,
        synth_cost=0.0,
    )
    return [
        *([route_event] if route_event else []),
        *events,
        StageEvent(
            stage="aggregation",
            role="final",
            status="ok",
            metadata={"aggregator": aggregator, "final_chars": len(final_answer)},
        ),
    ]


async def _aggregate_vote(
    run_id: str,
    preset: Preset,
    request: ChatCompletionRequest,
    key_hash: str,
    panel_results: list[PanelResult],
    route_event: StageEvent | None,
    start: float,
):
    ok = _ok_panels(panel_results)
    if not ok:
        raise InsufficientPanelError("Vote aggregator found no successful panel answers")
    counts = Counter((result.content or "").strip() for result in ok)
    final_answer = counts.most_common(1)[0][0]
    prompt_tokens, completion_tokens = _panel_usage(panel_results)
    usage = ResponseShaper.usage_block(prompt_tokens, completion_tokens)
    wall_ms = int((time.time() - start) * 1000)
    trace = FusionTrace(
        run_id=run_id,
        preset=preset.name,
        cost_usd=sum(result.cost_usd for result in panel_results),
        wall_ms=wall_ms,
        degraded=False,
        panel_results=panel_results,
        judge_analysis=None,
        final_answer=final_answer,
        stage_events=_aggregation_events(
            preset,
            panel_results,
            None,
            final_answer,
            route_event,
            aggregator="vote",
        ),
        metadata={**trace_metadata_for_preset(preset), "aggregator": "vote"},
    )
    await save_trace(trace, request.store, key_hash)
    payload = ResponseShaper.chat_completion(
        model=request.model,
        content=final_answer,
        usage=usage,
    )
    _cache_set(preset, request, payload)
    if request.stream:
        return _synthetic_stream(
            model=request.model,
            content=final_answer,
            usage=usage,
            include_usage=wants_usage(request),
        )
    return payload


async def _aggregate_ranked(
    run_id: str,
    preset: Preset,
    request: ChatCompletionRequest,
    key_hash: str,
    panel_results: list[PanelResult],
    route_event: StageEvent | None,
    start: float,
):
    ok = _ok_panels(panel_results)
    if not ok:
        raise InsufficientPanelError("Ranked aggregator found no successful panel answers")

    answers = "\n\n".join(
        f"ANSWER_{index + 1}:\n{result.content}" for index, result in enumerate(ok)
    )
    prompt = (
        "Pick the single best answer to the user's request. Return JSON with "
        '{"winner": <1-based index>, "reason": "short reason"}.\n\n'
        f"{answers}"
    )
    response = await BudgetedExecutor(run_id).call(
        "ranked_judge",
        provider_id=preset.provider_id_for(preset.judge_model, "judge"),
        model=preset.judge_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=min(preset.judge.max_tokens, 128),
        timeout=preset.judge.timeout,
        temperature=0,
    )
    judge_text = _response_text(response)
    try:
        data = extract_json_from_text(judge_text)
        winner = int(data.get("winner", 1))
        reason = str(data.get("reason", ""))
    except Exception:
        match = re.search(r"\b([1-9][0-9]*)\b", judge_text)
        winner = int(match.group(1)) if match else 1
        reason = judge_text[:500]
    winner = max(1, min(winner, len(ok)))
    final_answer = ok[winner - 1].content or ""
    judge_prompt, judge_completion = _response_usage(response)
    judge_cost = float(getattr(response, "_omnifusion_cost_usd", 0.0) or 0.0)
    judge_analysis = JudgeAnalysis(
        consensus=f"Ranked aggregator selected ANSWER_{winner}.",
        recommended_final_answer_plan=reason,
        cost_usd=judge_cost,
        prompt_tokens=judge_prompt,
        completion_tokens=judge_completion,
    )
    panel_prompt, panel_completion = _panel_usage(panel_results)
    usage = ResponseShaper.usage_block(
        panel_prompt + judge_prompt,
        panel_completion + judge_completion,
    )
    wall_ms = int((time.time() - start) * 1000)
    trace = FusionTrace(
        run_id=run_id,
        preset=preset.name,
        cost_usd=sum(result.cost_usd for result in panel_results) + judge_cost,
        wall_ms=wall_ms,
        degraded=False,
        panel_results=panel_results,
        judge_analysis=judge_analysis,
        final_answer=final_answer,
        stage_events=_aggregation_events(
            preset,
            panel_results,
            judge_analysis,
            final_answer,
            route_event,
            aggregator="ranked",
        ),
        metadata={
            **trace_metadata_for_preset(preset),
            "aggregator": "ranked",
            "winner": winner,
        },
    )
    await save_trace(trace, request.store, key_hash)
    payload = ResponseShaper.chat_completion(
        model=request.model,
        content=final_answer,
        usage=usage,
    )
    _cache_set(preset, request, payload)
    if request.stream:
        return _synthetic_stream(
            model=request.model,
            content=final_answer,
            usage=usage,
            include_usage=wants_usage(request),
        )
    return payload


async def execute_openfusion_hybrid(
    run_id: str,
    preset: Preset,
    request: ChatCompletionRequest,
    key_hash: str,
):
    start = time.time()
    ceiling_micro_usd = (
        int(preset.cost_ceiling * 1_000_000)
        if preset.cost_ceiling is not None
        else None
    )
    await initialize_request_budget(run_id, ceiling_micro_usd)

    cached = _cache_get(preset, request)
    if cached is not None:
        await _save_cache_trace(run_id, preset, request, key_hash, cached)
        if request.stream:
            content = cached["choices"][0]["message"].get("content") or ""
            return _synthetic_stream(
                model=request.model,
                content=content,
                usage=cached.get("usage") or {},
                include_usage=wants_usage(request),
            )
        return cached

    executor = BudgetedExecutor(run_id)
    route_decision, route_model, route_event = await _route_decision(
        run_id, preset, request, executor
    )
    if route_decision == "solo":
        return await _run_solo(
            run_id,
            preset,
            request,
            key_hash,
            route_model or preset.final_model,
            route_event,
        )

    web_sources: list[dict[str, Any]] = []
    panel_messages: list[Any] = request.messages
    if preset.web_enabled:
        from omnifusion.fusion.web_grounding import (
            gather_web_context,
            inject_grounding,
            latest_user_text,
        )

        web_context = await gather_web_context(run_id, latest_user_text(request.messages))
        web_sources = web_context.sources
        if web_context.has_grounding:
            panel_messages = inject_grounding(request.messages, web_context.grounding_text)

    panel_results: list[PanelResult] = []
    judge_analysis: JudgeAnalysis | None = None
    degraded = False
    try:
        panel_results = await _run_panel_for_mode(run_id, preset, panel_messages, request)

        if preset.aggregator == "vote":
            return await _aggregate_vote(
                run_id,
                preset,
                request,
                key_hash,
                panel_results,
                route_event,
                start,
            )
        if preset.aggregator == "ranked":
            return await _aggregate_ranked(
                run_id,
                preset,
                request,
                key_hash,
                panel_results,
                route_event,
                start,
            )

        judge_analysis = await run_judge(run_id, preset, request.messages, panel_results)
        consensus_lower = judge_analysis.consensus.lower()
        degraded = any(
            marker in consensus_lower
            for marker in ("degraded", "failed", "parse failure", "failed to execute")
        )

        try:
            final_result = await run_synthesis(
                run_id, preset, request, panel_results, judge_analysis, {}
            )
        except Exception:
            if preset.on_final_failure == "best_panel" and not request.stream:
                ok = _ok_panels(panel_results)
                if ok:
                    best_panel = max(ok, key=lambda result: len(result.content or ""))
                    wall_ms = int((time.time() - start) * 1000)
                    total_cost = sum(result.cost_usd for result in panel_results) + (
                        judge_analysis.cost_usd if judge_analysis else 0.0
                    )
                    trace = FusionTrace(
                        run_id=run_id,
                        preset=preset.name,
                        cost_usd=total_cost,
                        wall_ms=wall_ms,
                        degraded=True,
                        panel_results=panel_results,
                        judge_analysis=judge_analysis,
                        final_answer=best_panel.content,
                        stage_events=[
                            *([route_event] if route_event else []),
                            *build_stage_events(
                                preset,
                                panel_results,
                                judge_analysis,
                                best_panel.content,
                                synth_cost=0.0,
                                web_sources=web_sources,
                                degraded=True,
                            ),
                        ],
                        metadata={
                            **trace_metadata_for_preset(preset),
                            "web_sources": web_sources,
                        },
                    )
                    await save_trace(trace, request.store, key_hash)
                    return ResponseShaper.chat_completion(
                        model=request.model,
                        content=best_panel.content,
                        usage=ResponseShaper.usage_block(*_panel_usage(panel_results)),
                    )
            raise

        if request.stream:
            first_chunk = await final_result.__anext__()
            adapter = StreamingAdapter(request.model)

            async def gen():
                completion_text = ""
                stream_error = None
                synth_usage = None
                try:
                    if preset.analysis_emit.enabled and judge_analysis is not None:
                        data = judge_analysis.model_dump()
                        yield f"event: analysis\ndata: {json.dumps(data)}\n\n"
                    if getattr(first_chunk, "choices", None):
                        delta = getattr(first_chunk.choices[0], "delta", None)
                        content = getattr(delta, "content", None)
                        if content:
                            completion_text += content
                    if getattr(first_chunk, "usage", None):
                        synth_usage = first_chunk.usage
                    yield adapter.chunk_sse(first_chunk)
                    async for chunk in final_result:
                        if getattr(chunk, "choices", None):
                            delta = getattr(chunk.choices[0], "delta", None)
                            content = getattr(delta, "content", None)
                            if content:
                                completion_text += content
                        if getattr(chunk, "usage", None):
                            synth_usage = chunk.usage
                        yield adapter.chunk_sse(chunk)
                    if wants_usage(request):
                        final_prompt, final_completion = _usage_tokens(synth_usage)
                        panel_prompt, panel_completion = _panel_usage(panel_results)
                        judge_prompt = int(getattr(judge_analysis, "prompt_tokens", 0) or 0)
                        judge_completion = int(
                            getattr(judge_analysis, "completion_tokens", 0) or 0
                        )
                        yield adapter.usage_sse(
                            final_prompt + panel_prompt + judge_prompt,
                            final_completion + panel_completion + judge_completion,
                        )
                    yield adapter.done_sse()
                except Exception as exc:
                    stream_error = exc
                finally:
                    synth_cost = float(getattr(final_result, "cost_usd", 0.0) or 0.0)
                    if synth_cost == 0:
                        synth_cost = float(
                            getattr(final_result, "_omnifusion_cost_usd", 0.0) or 0.0
                        )
                    wall_ms = int((time.time() - start) * 1000)
                    total_cost = (
                        sum(result.cost_usd for result in panel_results)
                        + (judge_analysis.cost_usd if judge_analysis else 0.0)
                        + synth_cost
                    )
                    trace = FusionTrace(
                        run_id=run_id,
                        preset=preset.name,
                        cost_usd=total_cost,
                        wall_ms=wall_ms,
                        degraded=degraded or stream_error is not None,
                        panel_results=panel_results,
                        judge_analysis=judge_analysis,
                        final_answer=completion_text,
                        stage_events=[
                            *([route_event] if route_event else []),
                            *build_stage_events(
                                preset,
                                panel_results,
                                judge_analysis,
                                completion_text if stream_error is None else None,
                                synth_cost=synth_cost,
                                synth_usage=synth_usage,
                                web_sources=web_sources,
                                degraded=degraded or stream_error is not None,
                            ),
                        ],
                        metadata={
                            **trace_metadata_for_preset(preset),
                            "web_sources": web_sources,
                            "aggregator": "judge",
                        },
                    )
                    await save_trace(trace, request.store, key_hash)
                    if stream_error is not None:
                        raise stream_error

            return StreamingResponse(gen(), media_type="text/event-stream")

        content = _response_text(final_result)
        synth_cost = float(getattr(final_result, "_omnifusion_cost_usd", 0.0) or 0.0)
        total_cost = (
            sum(result.cost_usd for result in panel_results)
            + (judge_analysis.cost_usd if judge_analysis else 0.0)
            + synth_cost
        )
        wall_ms = int((time.time() - start) * 1000)
        trace = FusionTrace(
            run_id=run_id,
            preset=preset.name,
            cost_usd=total_cost,
            wall_ms=wall_ms,
            degraded=degraded,
            panel_results=panel_results,
            judge_analysis=judge_analysis,
            final_answer=content,
            stage_events=[
                *([route_event] if route_event else []),
                *build_stage_events(
                    preset,
                    panel_results,
                    judge_analysis,
                    content,
                    synth_cost=synth_cost,
                    synth_usage=getattr(final_result, "usage", None),
                    web_sources=web_sources,
                    degraded=degraded,
                ),
            ],
            metadata={
                **trace_metadata_for_preset(preset),
                "web_sources": web_sources,
                "aggregator": "judge",
            },
        )
        await save_trace(trace, request.store, key_hash)
        payload = ResponseShaper.chat_completion(
            model=request.model,
            content=content,
            usage=ResponseShaper.aggregate_usage(
                preset, panel_results, judge_analysis, final_result
            ),
            finish_reason=normalize_finish_reason(
                getattr(final_result.choices[0], "finish_reason", "stop")
            ),
        )
        _cache_set(preset, request, payload)
        return payload
    except Exception as exc:
        wall_ms = int((time.time() - start) * 1000)
        trace = FusionTrace(
            run_id=run_id,
            preset=preset.name,
            cost_usd=sum(result.cost_usd for result in panel_results),
            wall_ms=wall_ms,
            degraded=True,
            panel_results=panel_results,
            judge_analysis=judge_analysis,
            final_answer=None,
            stage_events=[
                *([route_event] if route_event else []),
                *build_stage_events(
                    preset,
                    panel_results,
                    judge_analysis,
                    None,
                    synth_cost=0.0,
                    web_sources=web_sources,
                    degraded=True,
                ),
            ],
            metadata={
                **trace_metadata_for_preset(preset),
                "web_sources": web_sources,
                "error": type(exc).__name__,
            },
        )
        await save_trace(trace, request.store, key_hash)
        raise


def estimate_openfusion_request(preset: Preset, request: ChatCompletionRequest) -> dict[str, Any]:
    messages = _message_dicts(request)
    max_tokens = request.max_tokens or preset.final.max_tokens
    stages = []

    if preset.router.enabled:
        decision = _heuristic_route(request, preset)
        stages.append({"stage": "router", "decision": decision, "cost_usd": 0.0})
        if decision == "solo":
            model = _select_route_model(request, preset)
            cost = estimate_call_cost(model, messages, max_tokens)
            return {
                "model": request.model,
                "preset": preset.name,
                "route": "solo",
                "estimated_cost_usd": cost,
                "stages": [*stages, {"stage": "router/solo", "model": model, "cost_usd": cost}],
            }

    panel_cost = 0.0
    panel_count = len(preset.panel_models)
    if preset.fusion_mode == "self_fusion":
        panel_count = preset.self_fusion.n
    elif preset.fusion_mode == "debate":
        panel_count = len(preset.panel_models) * (1 + preset.debate.rounds)
    panel_model = preset.panel_models[0]
    for index in range(panel_count):
        model = preset.panel_models[index % len(preset.panel_models)] if preset.panel_models else panel_model
        cost = estimate_call_cost(model, messages, preset.panel.max_tokens)
        panel_cost += cost
    stages.append({"stage": "panel", "calls": panel_count, "cost_usd": panel_cost})

    judge_cost = 0.0
    final_cost = 0.0
    if preset.aggregator in {"judge", "ranked"}:
        judge_cost = estimate_call_cost(preset.judge_model, messages, preset.judge.max_tokens)
        stages.append({"stage": "judge", "model": preset.judge_model, "cost_usd": judge_cost})
    if preset.aggregator == "judge":
        final_cost = estimate_call_cost(preset.final_model, messages, max_tokens)
        stages.append({"stage": "synthesis", "model": preset.final_model, "cost_usd": final_cost})

    return {
        "model": request.model,
        "preset": preset.name,
        "route": "fuse",
        "estimated_cost_usd": panel_cost + judge_cost + final_cost,
        "stages": stages,
    }
