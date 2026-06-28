import time
import logging
from fastapi.responses import StreamingResponse
from .types import Preset, FusionTrace, trace_metadata_for_preset, build_stage_events
from .panel import run_panel
from .judge import run_judge
from .synth import run_synthesis
from ..api.schemas import ChatCompletionRequest
from ..api.normalize import generation_passthrough_kwargs
from ..api.sse import wants_usage
from ..budget.ledger import initialize_request_budget
from ..store.runs import save_trace
from .runtime.response import ResponseShaper
from .runtime.streaming import StreamingAdapter, normalize_finish_reason

logger = logging.getLogger("omnifusion.orchestrator")


def _read_usage(usage) -> tuple:
    """Thin alias for the canonical usage reader on the shaper."""
    return ResponseShaper.read_usage(usage)


def _final_result_cost(final_result) -> float:
    return float(
        getattr(
            final_result,
            "_omnifusion_cost_usd",
            getattr(final_result, "cost_usd", 0.0),
        )
        or 0.0
    )


def _trace_metadata(preset, web_sources) -> dict:
    """Preset trace metadata, plus bounded web-source attribution when web grounding
    ran (Invariant 6: URL/title/hash/excerpt only — never the full page)."""
    metadata = trace_metadata_for_preset(preset)
    if web_sources:
        metadata["web_sources"] = web_sources
    return metadata


async def run_fusion(
    run_id: str, preset: Preset, request: ChatCompletionRequest, key_hash: str
):
    from .runtime.registry import execute_strategy

    return await execute_strategy(run_id, preset, request, key_hash)


async def run_fusion_classic(
    run_id: str, preset: Preset, request: ChatCompletionRequest, key_hash: str
):
    start_time = time.time()

    # 1. Initialize Request and Global budgets
    ceiling_micro_usd = (
        int(preset.cost_ceiling * 1_000_000)
        if preset.cost_ceiling is not None
        else None
    )
    await initialize_request_budget(run_id, ceiling_micro_usd)

    panel_results = []
    judge_analysis = None
    degraded = False
    web_sources = []

    try:
        # 1b. Server-side web grounding ("web on"). Opt-in per preset / plugins.web.
        # Untrusted, fenced, attributed; each web call is its own budget stage.
        panel_messages = request.messages
        if getattr(preset, "web_enabled", False):
            from .web_grounding import gather_web_context, inject_grounding, latest_user_text

            web_context = await gather_web_context(
                run_id, latest_user_text(request.messages)
            )
            web_sources = web_context.sources
            if web_context.has_grounding:
                panel_messages = inject_grounding(
                    request.messages, web_context.grounding_text
                )

        # 2. Run Panel (on the web-grounded messages when enabled). Forward caller
        # generation params (seed/penalties/service_tier) so they take effect.
        panel_results = await run_panel(
            run_id,
            preset,
            panel_messages,
            min_success=preset.min_panel_success
            if hasattr(preset, "min_panel_success")
            else 1,
            extra_kwargs=generation_passthrough_kwargs(request),
        )

        # 3. Run Judge
        judge_analysis = await run_judge(
            run_id, preset, request.messages, panel_results
        )
        # Fix (medium): use structural check instead of naive substring match for degraded.
        # "Degraded" if judge explicitly notes failure/degradation in its consensus.
        consensus_lower = judge_analysis.consensus.lower()
        if (
            "degraded" in consensus_lower
            or "failed" in consensus_lower
            or "parse failure" in consensus_lower
            or "failed to execute" in consensus_lower
        ):
            degraded = True

        # 4. Synthesis
        try:
            final_result = await run_synthesis(
                run_id, preset, request, panel_results, judge_analysis, {}
            )
        except Exception as e:
            on_fail = getattr(preset, "on_final_failure", "error")
            if on_fail == "best_panel" and not request.stream:
                # Fallback to best panelist answer
                ok_panels = [r for r in panel_results if r.status == "ok"]
                if ok_panels:
                    best_panel = max(ok_panels, key=lambda x: len(x.content or ""))
                    degraded = True

                    final_response_dict = ResponseShaper.chat_completion(
                        model=request.model,
                        content=best_panel.content,
                        usage={
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                        finish_reason="stop",
                    )

                    wall_ms = int((time.time() - start_time) * 1000)
                    panel_cost = sum(r.cost_usd for r in panel_results)
                    judge_cost = judge_analysis.cost_usd if judge_analysis else 0.0
                    total_cost = panel_cost + judge_cost
                    trace = FusionTrace(
                        run_id=run_id,
                        preset=preset.name,
                        cost_usd=total_cost,
                        wall_ms=wall_ms,
                        degraded=True,
                        panel_results=panel_results,
                        judge_analysis=judge_analysis,
                        final_answer=best_panel.content,
                        stage_events=build_stage_events(
                            preset,
                            panel_results,
                            judge_analysis,
                            best_panel.content,
                            synth_cost=0.0,
                            web_sources=web_sources,
                            degraded=True,
                        ),
                        metadata=_trace_metadata(preset, web_sources),
                    )
                    await save_trace(trace, request.store, key_hash)
                    return final_response_dict
            raise e

        if request.stream:
            # Streaming commits 200 ONLY after final synthesis yields its first token
            # Get the first chunk
            first_chunk = await final_result.__anext__()

            _fusion_model = f"fusion/{preset.name}"
            stream_adapter = StreamingAdapter(_fusion_model)

            async def stream_generator():
                completion_text = ""
                stream_error = None
                synth_usage = None
                try:
                    if first_chunk.choices and len(first_chunk.choices) > 0:
                        delta = first_chunk.choices[0].delta
                        if delta and delta.content:
                            completion_text += delta.content
                    if getattr(first_chunk, "usage", None):
                        synth_usage = first_chunk.usage
                    yield stream_adapter.chunk_sse(first_chunk)

                    async for chunk in final_result:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                completion_text += delta.content
                        if getattr(chunk, "usage", None):
                            synth_usage = chunk.usage
                        yield stream_adapter.chunk_sse(chunk)

                    # Clean completion: optionally emit an aggregate usage chunk, then [DONE].
                    if wants_usage(request):
                        s_pt, s_ct = _read_usage(synth_usage)
                        agg_pt, agg_ct = s_pt, s_ct
                        for r in panel_results:
                            p, c = _read_usage(getattr(r, "usage", None))
                            agg_pt += p
                            agg_ct += c
                        if judge_analysis is not None:
                            agg_pt += int(getattr(judge_analysis, "prompt_tokens", 0) or 0)
                            agg_ct += int(getattr(judge_analysis, "completion_tokens", 0) or 0)
                        yield stream_adapter.usage_sse(agg_pt, agg_ct)
                    yield stream_adapter.done_sse()

                except Exception as exc:
                    # Mid-stream failure (HTTP 200 already committed). Per the failure
                    # policy we do NOT emit a synthetic error chunk and do NOT emit
                    # [DONE] — an OpenAI-compatible client must see an aborted stream,
                    # not a cleanly-terminated one. We record the failure in the trace
                    # and re-raise so the transport closes the connection abnormally.
                    stream_error = exc
                    logger.error(
                        f"Streaming synthesis error for run {run_id}: {exc}", exc_info=True
                    )
                finally:
                    wall_ms = int((time.time() - start_time) * 1000)
                    panel_cost = sum(r.cost_usd for r in panel_results)
                    judge_cost = judge_analysis.cost_usd if judge_analysis else 0.0
                    synth_cost = _final_result_cost(final_result)
                    total_cost = panel_cost + judge_cost + synth_cost
                    _stream_degraded = degraded or stream_error is not None
                    trace = FusionTrace(
                        run_id=run_id,
                        preset=preset.name,
                        cost_usd=total_cost,
                        wall_ms=wall_ms,
                        degraded=_stream_degraded,
                        panel_results=panel_results,
                        judge_analysis=judge_analysis,
                        final_answer=completion_text,
                        stage_events=build_stage_events(
                            preset,
                            panel_results,
                            judge_analysis,
                            completion_text if stream_error is None else None,
                            synth_cost=synth_cost,
                            synth_usage=synth_usage,
                            web_sources=web_sources,
                            degraded=_stream_degraded,
                        ),
                        metadata=_trace_metadata(preset, web_sources),
                    )
                    await save_trace(trace, request.store, key_hash)

                if stream_error is not None:
                    # Abort the response so the client does not treat it as complete.
                    raise stream_error

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            content = final_result.choices[0].message.content
            panel_cost = sum(r.cost_usd for r in panel_results)
            judge_cost = judge_analysis.cost_usd if judge_analysis else 0.0
            synth_cost = _final_result_cost(final_result)
            total_cost = panel_cost + judge_cost + synth_cost

            wall_ms = int((time.time() - start_time) * 1000)

            trace = FusionTrace(
                run_id=run_id,
                preset=preset.name,
                cost_usd=total_cost,
                wall_ms=wall_ms,
                degraded=degraded,
                panel_results=panel_results,
                judge_analysis=judge_analysis,
                final_answer=content,
                stage_events=build_stage_events(
                    preset,
                    panel_results,
                    judge_analysis,
                    content,
                    synth_cost=synth_cost,
                    synth_usage=getattr(final_result, "usage", None),
                    web_sources=web_sources,
                    degraded=degraded,
                ),
                metadata=_trace_metadata(preset, web_sources),
            )
            await save_trace(trace, request.store, key_hash)

            # Return an OpenAI-compatible response with fusion/<preset> as the model.
            # Usage aggregates panel + judge + final by default (preset.usage_reporting).
            usage = ResponseShaper.aggregate_usage(
                preset, panel_results, judge_analysis, final_result
            )
            finish_reason = normalize_finish_reason(
                getattr(final_result.choices[0], "finish_reason", "stop")
            )
            return ResponseShaper.chat_completion(
                model=f"fusion/{preset.name}",
                content=content,
                usage=usage,
                finish_reason=finish_reason,
            )

    except Exception as outer_err:
        wall_ms = int((time.time() - start_time) * 1000)
        panel_cost = sum(r.cost_usd for r in panel_results) if panel_results else 0.0
        judge_cost = judge_analysis.cost_usd if judge_analysis else 0.0
        total_cost = panel_cost + judge_cost
        trace = FusionTrace(
            run_id=run_id,
            preset=preset.name,
            cost_usd=total_cost,
            wall_ms=wall_ms,
            degraded=True,
            panel_results=panel_results,
            judge_analysis=judge_analysis,
            final_answer=None,
            stage_events=build_stage_events(
                preset,
                panel_results,
                judge_analysis,
                None,
                synth_cost=0.0,
                web_sources=web_sources,
                degraded=True,
            ),
            metadata=_trace_metadata(preset, web_sources),
        )
        await save_trace(trace, request.store, key_hash)
        raise outer_err
