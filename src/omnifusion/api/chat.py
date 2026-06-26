from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse
from .schemas import ChatCompletionRequest
from .auth import verify_api_key
from .errors import OmniFusionError
from .model_names import normalize_requested_model
from ..fusion.orchestrator import run_fusion
from ..fusion.tool_orchestrator import run_fusion_with_tools
from ..store.presets import get_or_create_compat_placeholder_preset, get_preset
from ..settings import settings
from ..llm.client import llm_client
import uuid
import time
import asyncio
import logging
from ..budget.ledger import initialize_request_budget, reserve_budget, reconcile_budget
from ..store.runs import save_trace
from ..fusion.types import FusionTrace
from .sse import wants_usage, usage_chunk_sse
from ..logging_config import set_run_id
from ..providers.pricing import (
    estimate_call_cost,
    calculate_actual_cost,
    estimate_tokens,
    get_model_cost_estimate,
    usd_to_micro,
)

logger = logging.getLogger("omnifusion.api.chat")

router = APIRouter()

# Fix #11: Per-API-key inbound concurrency semaphores.
# Dictionary of key_hash -> asyncio.Semaphore
# Prevents a single key from fan-outing unbounded concurrent fusion requests.
_key_semaphores: dict = {}
_key_semaphores_lock = asyncio.Lock()


async def _acquire_key_slot(key_hash: str) -> asyncio.Semaphore:
    """Acquire a concurrency slot for this API key. Returns the semaphore."""
    async with _key_semaphores_lock:
        if key_hash not in _key_semaphores:
            _key_semaphores[key_hash] = asyncio.Semaphore(
                settings.omnifusion_max_concurrent_per_key
            )
    sem = _key_semaphores[key_hash]
    await asyncio.wait_for(sem.acquire(), timeout=10.0)
    return sem


def _defer_slot_release_to_stream(result, sem: asyncio.Semaphore) -> bool:
    """If `result` is a streamed response, hold the concurrency slot until the
    stream body is fully consumed (or aborted) rather than releasing it the moment
    the handler returns.

    Without this, a streamed fusion/passthrough request releases its per-key slot
    immediately on return — so a client could hold open many long-lived streams
    that no longer count against the inbound concurrency cap, defeating the
    spend/DoS guard. Returns True when ownership of the slot has been transferred
    to the stream (caller must NOT release it in its own finally).
    """
    if sem is None or not isinstance(result, StreamingResponse):
        return False

    original_iterator = result.body_iterator

    async def _release_after_stream():
        try:
            async for chunk in original_iterator:
                yield chunk
        finally:
            sem.release()

    result.body_iterator = _release_after_stream()
    return True


async def _single_model_completion(
    run_id: str,
    model: str,
    body: ChatCompletionRequest,
    key_hash: str,
    sem,
    label: str,
    extra_kwargs: dict = None,
):
    """Single-model OpenAI-compatible completion with budget reservation + trace.

    Used for (a) passthrough-whitelisted models and (b) tool-calling requests to a
    fusion preset: the council cannot emit tool_calls, so when `tools` are present we
    route to one tool-capable model and return its response (tool_calls included)
    verbatim. Returns (result, stream_owns_sem) — the caller returns `result` and uses
    the flag to decide whether to release the per-key slot in its own finally.
    """
    start_time = time.time()
    # exclude_none so tool/assistant messages serialize to clean OpenAI shape
    # (no null tool_calls/tool_call_id/name leaking onto normal messages).
    dict_messages = [m.model_dump(exclude_none=True) for m in body.messages]

    ceiling_micro_usd = int(getattr(settings, "request_budget_usd", 10.0) * 1_000_000)
    await initialize_request_budget(run_id, ceiling_micro_usd)

    max_tokens = body.max_tokens if body.max_tokens is not None else 1024
    cost_usd = estimate_call_cost(model, dict_messages, max_tokens)
    reserve_micro_usd = max(1, int(cost_usd * 1_000_000))
    reservation_id = await reserve_budget(run_id, label, reserve_micro_usd)

    call_kwargs = {
        "provider_id": "default",
        "model": model,
        "messages": dict_messages,
        "stream": body.stream,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "max_tokens": body.max_tokens,
        "stop": body.stop,
    }
    if extra_kwargs:
        call_kwargs.update(extra_kwargs)
    call_kwargs = {k: v for k, v in call_kwargs.items() if v is not None}

    reconciled = False
    try:
        try:
            response_obj = await asyncio.wait_for(
                llm_client.acompletion(**call_kwargs),
                timeout=settings.omnifusion_wall_timeout,
            )
        except asyncio.TimeoutError:
            raise OmniFusionError(
                f"Request exceeded wall timeout of {settings.omnifusion_wall_timeout}s",
                status_code=504,
                type_="server_error",
                code="wall_timeout",
            )

        if body.stream:
            first_chunk = await response_obj.__anext__()

            async def stream_gen():
                completion_text = ""
                stream_usage = None
                prompt_tokens = 0
                completion_tokens = 0
                try:
                    if first_chunk.choices and len(first_chunk.choices) > 0:
                        delta = first_chunk.choices[0].delta
                        if delta and getattr(delta, "content", None):
                            completion_text += delta.content
                    if getattr(first_chunk, "usage", None):
                        stream_usage = first_chunk.usage
                    yield f"data: {first_chunk.model_dump_json()}\n\n"

                    async for chunk in response_obj:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if delta and getattr(delta, "content", None):
                                completion_text += delta.content
                        if getattr(chunk, "usage", None):
                            stream_usage = chunk.usage
                        yield f"data: {chunk.model_dump_json()}\n\n"

                    # Prefer provider-reported usage from the terminal chunk; only
                    # fall back to the char//4 heuristic when the stream omits usage.
                    if stream_usage is not None:
                        prompt_tokens = int(getattr(stream_usage, "prompt_tokens", 0) or 0)
                        completion_tokens = int(getattr(stream_usage, "completion_tokens", 0) or 0)
                    else:
                        prompt_tokens = estimate_tokens(model, dict_messages)
                        completion_tokens = max(1, len(completion_text) // 4)

                    if wants_usage(body):
                        yield usage_chunk_sse(model, prompt_tokens, completion_tokens)
                    yield "data: [DONE]\n\n"
                finally:
                    async def _cleanup():
                        pt = prompt_tokens or estimate_tokens(model, dict_messages)
                        ct = completion_tokens or max(1, len(completion_text) // 4)
                        actual = get_model_cost_estimate(model, pt, ct)
                        await reconcile_budget(reservation_id, usd_to_micro(actual))
                        wall_ms = int((time.time() - start_time) * 1000)
                        trace = FusionTrace(
                            run_id=run_id,
                            preset=label,
                            cost_usd=actual,
                            wall_ms=wall_ms,
                            degraded=False,
                            panel_results=[],
                            judge_analysis=None,
                            final_answer=completion_text,
                        )
                        await save_trace(trace, body.store, key_hash)
                    await asyncio.shield(_cleanup())

            reconciled = True
            resp = StreamingResponse(stream_gen(), media_type="text/event-stream")
            resp.headers["X-OmniFusion-Run-Id"] = run_id
            owns = _defer_slot_release_to_stream(resp, sem)
            return resp, owns
        else:
            actual = calculate_actual_cost(response_obj, model)

            async def _cleanup():
                await reconcile_budget(reservation_id, usd_to_micro(actual))
                wall_ms = int((time.time() - start_time) * 1000)
                content = (
                    response_obj.choices[0].message.content
                    if response_obj.choices
                    else None
                )
                trace = FusionTrace(
                    run_id=run_id,
                    preset=label,
                    cost_usd=actual,
                    wall_ms=wall_ms,
                    degraded=False,
                    panel_results=[],
                    judge_analysis=None,
                    final_answer=content,
                )
                await save_trace(trace, body.store, key_hash)
            await asyncio.shield(_cleanup())
            reconciled = True
            return response_obj, False
    finally:
        if not reconciled:
            async def _cleanup():
                await reconcile_budget(reservation_id, 0)
                wall_ms = int((time.time() - start_time) * 1000)
                trace = FusionTrace(
                    run_id=run_id,
                    preset=label,
                    cost_usd=0.0,
                    wall_ms=wall_ms,
                    degraded=True,
                    panel_results=[],
                    judge_analysis=None,
                    final_answer=None,
                )
                await save_trace(trace, body.store, key_hash)
            await asyncio.shield(_cleanup())


@router.post("/chat/completions")
async def create_chat_completion(
    request: Request,
    body: ChatCompletionRequest,
    response: Response,
    key_hash: str = Depends(verify_api_key),
):
    run_id = str(uuid.uuid4())
    request.state.run_id = run_id
    response.headers["X-OmniFusion-Run-Id"] = run_id
    set_run_id(run_id)

    normalized_model = normalize_requested_model(body.model)
    if normalized_model != body.model:
        body = body.model_copy(update={"model": normalized_model})

    # Fix #11: Acquire per-key concurrency slot
    sem = None
    stream_owns_sem = False
    try:
        sem = await _acquire_key_slot(key_hash)
    except asyncio.TimeoutError:
        raise OmniFusionError(
            "Too many concurrent requests for this API key. Try again shortly.",
            status_code=429,
            type_="rate_limit_error",
            code="too_many_requests",
        )

    try:
        if body.model.startswith("fusion/"):
            preset_name = body.model[len("fusion/"):]
            preset = await get_preset(preset_name)
            if not preset:
                preset = await get_or_create_compat_placeholder_preset(preset_name)
            if not preset:
                raise OmniFusionError(f"Preset {preset_name} not found", status_code=404)

            # Tool-calling requests: fuse the NEXT ACTION at every agentic step —
            # the panel proposes actions, the judge picks the best, we return it.
            # (See fusion/tool_orchestrator.py.) Keeps the council's benefit for
            # agentic/Draco-style tasks instead of bypassing fusion.
            if body.tools:
                try:
                    result = await asyncio.wait_for(
                        run_fusion_with_tools(run_id, preset, body, key_hash),
                        timeout=settings.omnifusion_wall_timeout,
                    )
                except asyncio.TimeoutError:
                    raise OmniFusionError(
                        f"Request exceeded wall timeout of {settings.omnifusion_wall_timeout}s",
                        status_code=504,
                        type_="server_error",
                        code="wall_timeout",
                    )
                if isinstance(result, StreamingResponse):
                    result.headers["X-OmniFusion-Run-Id"] = run_id
                stream_owns_sem = _defer_slot_release_to_stream(result, sem)
                return result

            # Fix #13: Wrap run_fusion with wall_timeout from settings
            try:
                result = await asyncio.wait_for(
                    run_fusion(run_id, preset, body, key_hash),
                    timeout=settings.omnifusion_wall_timeout,
                )
            except asyncio.TimeoutError:
                raise OmniFusionError(
                    f"Request exceeded wall timeout of {settings.omnifusion_wall_timeout}s",
                    status_code=504,
                    type_="server_error",
                    code="wall_timeout",
                )
            # Streaming responses bypass the injected `response` object, so set the
            # run-id header directly on them; otherwise stream clients can't read it.
            if isinstance(result, StreamingResponse):
                result.headers["X-OmniFusion-Run-Id"] = run_id
            # Hold the per-key slot until a streamed body is fully consumed.
            stream_owns_sem = _defer_slot_release_to_stream(result, sem)
            return result

        else:
            # Check passthrough whitelist
            if body.model in settings.omnifusion_passthrough_whitelist:
                extra = None
                if body.tools:
                    body_dict = body.model_dump(exclude_none=True)
                    extra = {
                        "tools": body_dict.get("tools"),
                        "tool_choice": body_dict.get("tool_choice"),
                    }
                result, stream_owns_sem = await _single_model_completion(
                    run_id,
                    body.model,
                    body,
                    key_hash,
                    sem,
                    label=f"passthrough/{body.model}",
                    extra_kwargs=extra,
                )
                return result
            else:
                raise OmniFusionError(
                    f"Model {body.model} is not supported or whitelisted.", status_code=404
                )
    finally:
        # Fix #11: Release the concurrency slot — unless ownership was handed off to
        # a streaming response, which releases it when its body finishes/aborts.
        if sem is not None and not stream_owns_sem:
            sem.release()
