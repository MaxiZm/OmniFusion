"""
Per-step fusion with tool calling.

A fusion council can't be applied to the *final* answer when the task is agentic —
the work IS the sequence of tool calls. So instead of fusing the final text, we fuse
the NEXT ACTION at every step of the agentic loop:

    1. Panel  — every model independently proposes the next step (a tool call, or a
                final answer), in parallel, given the full conversation + the tools.
    2. Judge  — looks at all proposed actions and picks the single best next action.
    3. Emit   — return that action to the client:
                  * tool call  -> finish_reason="tool_calls"; client executes, loops back
                  * final      -> run the classic panel→judge→synthesis on the text
                                  proposals to produce the fused final answer.

This preserves the fusion benefit (N diverse models + a judge at every decision point)
while remaining a drop-in OpenAI tool-calling endpoint, which is what agentic
benchmarks (e.g. Draco: deep research + complex orchestration) require.

NOTE: panel models must support function calling. Modern models — including DeepSeek
V4 pro/flash (tools work in both thinking and non-thinking modes) — do. A model that
genuinely doesn't support tools errors on the call and is dropped from the panel.
"""
import asyncio
import json
import time
import uuid
import logging
from typing import Optional

from fastapi.responses import StreamingResponse

from .types import Preset, PanelResult, JudgeAnalysis, trace_metadata_for_preset
from ..llm.client import llm_client
from ..budget.ledger import (
    initialize_request_budget,
    reserve_budget,
    reconcile_budget,
)
from ..providers.pricing import calculate_actual_cost, estimate_call_cost, usd_to_micro
from ..api.errors import InsufficientPanelError
from ..api.schemas import ChatCompletionRequest
from ..api.sse import wants_usage, usage_chunk_sse
from .judge import run_judge, extract_json_from_text
from .synth import run_synthesis
from .types import FusionTrace
from ..store.runs import save_trace

logger = logging.getLogger("omnifusion.tool_orchestrator")


def _final_result_cost(final_result) -> float:
    return float(
        getattr(
            final_result,
            "_omnifusion_cost_usd",
            getattr(final_result, "cost_usd", 0.0),
        )
        or 0.0
    )


def _disable_thinking_kwargs(model: str) -> dict:
    """DeepSeek V4 defaults to *thinking* mode, which returns reasoning_content that
    must be passed back on every subsequent turn of a multi-turn tool conversation.
    A standard OpenAI client (OpenCode, etc.) doesn't preserve that DeepSeek-specific
    field, so the second agentic step gets rejected with "reasoning_content ... must
    be passed back". Disabling thinking on the agentic path makes the tool loop work.
    Returns {} for non-DeepSeek models (the field would be an unknown param to them).
    """
    if "deepseek" in model.lower():
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def _normalize_tool_calls(tool_calls) -> Optional[list]:
    """Convert litellm tool_calls (objects or dicts) to plain OpenAI-shaped dicts."""
    if not tool_calls:
        return None
    out = []
    for tc in tool_calls:
        if hasattr(tc, "model_dump"):
            d = tc.model_dump()
        elif isinstance(tc, dict):
            d = tc
        else:
            continue
        fn = d.get("function") or {}
        name = fn.get("name")
        if not name:
            # Skip malformed tool calls with no function name: they are an invalid
            # OpenAI message shape and would crash the trace builder's name join.
            continue
        out.append(
            {
                "id": d.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": fn.get("arguments") or "{}",
                },
            }
        )
    return out or None


def _usage_tokens(usage) -> tuple:
    """Extract (prompt_tokens, completion_tokens) from a usage object/dict (0 if absent)."""
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


async def _panel_propose(run_id, preset, dict_messages, tools, tool_choice):
    """Each panel model proposes its next action (tool call or text), in parallel."""

    async def one(model):
        cost = estimate_call_cost(model, dict_messages, preset.panel.max_tokens)
        reservation_id = await reserve_budget(
            run_id, f"tool-panel/{model}", max(1, int(cost * 1_000_000))
        )
        actual = 0.0
        try:
            resp = await llm_client.acompletion(
                provider_id="default",
                model=model,
                messages=dict_messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=preset.panel.max_tokens,
                timeout=preset.panel.timeout,
                **_disable_thinking_kwargs(model),
            )
            actual = calculate_actual_cost(resp, model)
            msg = resp.choices[0].message
            pt, ct = _usage_tokens(getattr(resp, "usage", None))
            return {
                "model": model,
                "ok": True,
                "tool_calls": _normalize_tool_calls(getattr(msg, "tool_calls", None)),
                "content": getattr(msg, "content", None),
                "cost": actual,
                "prompt_tokens": pt,
                "completion_tokens": ct,
            }
        except Exception as e:
            logger.warning(f"tool-panel {model} failed to propose: {e}")
            return {"model": model, "ok": False, "error": str(e), "cost": 0.0}
        finally:
            await asyncio.shield(reconcile_budget(reservation_id, usd_to_micro(actual)))

    tasks = [one(m) for m in preset.panel_models[:8]]
    return await asyncio.gather(*tasks)


def _describe_proposal(i: int, p: dict) -> str:
    if p.get("tool_calls"):
        calls = "; ".join(
            f"{tc['function']['name']}({tc['function']['arguments']})"
            for tc in p["tool_calls"]
        )
        return f"[Agent {i}] proposes TOOL CALL: {calls}"
    text = (p.get("content") or "").strip()
    return f"[Agent {i}] proposes FINAL ANSWER: {text[:600]}"


async def _decide_next_step(run_id, preset, dict_messages, ok_proposals) -> dict:
    """Judge selects the single best next action across the panel's proposals.

    Returns {"decision": "tool"|"final", "best_index": int}.
    """
    # If nobody proposed a tool call, the step is necessarily a final answer.
    any_tool = any(p.get("tool_calls") for p in ok_proposals)
    if not any_tool:
        return {"decision": "final", "best_index": 0}

    descriptions = "\n".join(
        _describe_proposal(i, p) for i, p in enumerate(ok_proposals)
    )
    # Use the last user/tool turn as task context.
    last_ctx = ""
    for m in reversed(dict_messages):
        if m.get("role") in ("user", "tool") and m.get("content"):
            last_ctx = str(m.get("content"))[:1500]
            break

    judge_prompt = (
        "You are the coordinator of a panel of agents solving a task that may require "
        "tools (functions). Below is the latest task context and each agent's proposed "
        "NEXT step. Choose the single best next step that most correctly advances the "
        "task toward a complete, accurate solution.\n\n"
        "Rules:\n"
        "- If a tool call is the best next step, pick the agent whose tool call (name + "
        "arguments) is most correct and useful. Prefer well-formed, relevant calls.\n"
        "- If the task is already solved and the agents should now produce a final "
        "answer, choose decision \"final\".\n\n"
        f"TASK CONTEXT:\n{last_ctx}\n\n"
        f"PROPOSED NEXT STEPS:\n{descriptions}\n\n"
        'Output ONLY JSON: {"decision":"tool"|"final","best_index":<agent number>,'
        '"reasoning":"<one sentence>"}'
    )

    messages = [{"role": "user", "content": judge_prompt}]
    cost = estimate_call_cost(preset.judge_model, messages, preset.judge.max_tokens)
    reservation_id = await reserve_budget(
        run_id, "tool-judge", max(1, int(cost * 1_000_000))
    )
    actual = 0.0
    try:
        # Request JSON mode; if a model rejects it, retry without (the extractor is
        # robust either way). filter_params drops it for providers that don't support it.
        kwargs = {
            "timeout": preset.judge.timeout,
            "max_tokens": preset.judge.max_tokens,
            "response_format": {"type": "json_object"},
            **_disable_thinking_kwargs(preset.judge_model),
        }
        try:
            resp = await llm_client.acompletion(
                provider_id="default", model=preset.judge_model, messages=messages, **kwargs
            )
        except Exception:
            kwargs.pop("response_format", None)
            resp = await llm_client.acompletion(
                provider_id="default", model=preset.judge_model, messages=messages, **kwargs
            )
        actual = calculate_actual_cost(resp, preset.judge_model)
        data = extract_json_from_text(resp.choices[0].message.content)
        decision = data.get("decision", "tool")
        best_index = int(data.get("best_index", 0))
    except Exception as e:
        logger.warning(f"tool-judge failed, defaulting to first tool proposal: {e}")
        decision, best_index = "tool", 0
    finally:
        await asyncio.shield(reconcile_budget(reservation_id, usd_to_micro(actual)))

    # Validate index and that the chosen proposal actually has a tool call.
    if best_index < 0 or best_index >= len(ok_proposals):
        best_index = 0
    if decision == "tool" and not ok_proposals[best_index].get("tool_calls"):
        # Judge picked a non-tool proposal but said "tool"; find any tool proposal.
        for i, p in enumerate(ok_proposals):
            if p.get("tool_calls"):
                best_index = i
                break
        else:
            decision = "final"
    return {"decision": decision, "best_index": best_index}


def _usage_block(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _tool_call_response_dict(preset, tool_calls, usage) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"fusion/{preset.name}",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": usage,
    }


def _tool_call_sse(preset, tool_calls):
    cid = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    model = f"fusion/{preset.name}"

    def chunk(delta, finish=None):
        return {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    yield f"data: {json.dumps(chunk({'role': 'assistant', 'content': None}))}\n\n"
    delta_tcs = [
        {
            "index": i,
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            },
        }
        for i, tc in enumerate(tool_calls)
    ]
    yield f"data: {json.dumps(chunk({'tool_calls': delta_tcs}))}\n\n"
    yield f"data: {json.dumps(chunk({}, finish='tool_calls'))}\n\n"
    yield "data: [DONE]\n\n"


async def run_fusion_with_tools(run_id, preset: Preset, body: ChatCompletionRequest, key_hash: str):
    """Per-step fusion for tool-calling requests. Returns a dict (non-stream) or a
    StreamingResponse, exactly like run_fusion."""
    start_time = time.time()
    ceiling_micro_usd = (
        int(preset.cost_ceiling * 1_000_000) if preset.cost_ceiling is not None else None
    )
    await initialize_request_budget(run_id, ceiling_micro_usd)

    dict_messages = [m.model_dump(exclude_none=True) for m in body.messages]
    body_dict = body.model_dump(exclude_none=True)
    tools = body_dict.get("tools")
    tool_choice = body_dict.get("tool_choice", "auto")

    # 1. Panel proposes next actions (parallel).
    proposals = await _panel_propose(run_id, preset, dict_messages, tools, tool_choice)
    ok = [p for p in proposals if p.get("ok")]
    if not ok:
        raise InsufficientPanelError(
            "No panel model could propose a next step (do the panel models support tools?)"
        )

    panel_cost = sum(p.get("cost", 0.0) for p in proposals)

    # 2. Judge selects the best next action.
    decision = await _decide_next_step(run_id, preset, dict_messages, ok)

    # Aggregate panel token usage so responses report real (non-zero) usage.
    panel_pt = sum(p.get("prompt_tokens", 0) for p in proposals)
    panel_ct = sum(p.get("completion_tokens", 0) for p in proposals)

    # 3a. Tool step: return the chosen tool call(s); client executes and loops back.
    if decision["decision"] == "tool":
        chosen = ok[decision["best_index"]]
        tool_calls = chosen["tool_calls"]
        # Build a trace summarizing the step (coerce names defensively).
        panel_results = [
            PanelResult(
                model=p["model"],
                status="ok" if p.get("ok") else "error",
                content=(
                    "tool: "
                    + "; ".join(
                        str(tc["function"].get("name") or "?")
                        for tc in (p.get("tool_calls") or [])
                    )
                    if p.get("tool_calls")
                    else (p.get("content") or "")
                ),
                cost_usd=p.get("cost", 0.0),
            )
            for p in proposals
        ]
        trace = FusionTrace(
            run_id=run_id,
            preset=preset.name,
            cost_usd=panel_cost,
            wall_ms=int((time.time() - start_time) * 1000),
            degraded=False,
            panel_results=panel_results,
            judge_analysis=JudgeAnalysis(
                consensus=f"Selected tool call from {chosen['model']}",
            ),
            final_answer=None,
            metadata=trace_metadata_for_preset(preset),
        )
        await save_trace(trace, body.store, key_hash)

        if body.stream:
            return StreamingResponse(
                _tool_call_sse(preset, tool_calls), media_type="text/event-stream"
            )
        return _tool_call_response_dict(
            preset, tool_calls, _usage_block(panel_pt, panel_ct)
        )

    # 3b. Final step: fuse the text proposals into the final answer via the classic
    # panel→judge→synthesis path. Build PanelResults from the text proposals.
    panel_results = [
        PanelResult(
            model=p["model"],
            status="ok",
            content=p.get("content") or "",
            cost_usd=p.get("cost", 0.0),
        )
        for p in ok
        if (p.get("content") or "").strip()
    ]
    if not panel_results:
        # Edge case: judge said final but no text content exists. Ask the final model
        # directly for the answer based on the conversation.
        panel_results = [
            PanelResult(model=p["model"], status="ok", content="", cost_usd=p.get("cost", 0.0))
            for p in ok
        ]

    judge_analysis = await run_judge(run_id, preset, body.messages, panel_results)

    # Synthesis runs without tools, so it must not re-send assistant tool_calls /
    # tool-result messages (models reject tool messages when no tools are defined).
    # But we must NOT lose the tool-derived context: fold a compact transcript of the
    # assistant tool calls + tool results into an appended note so the synthesis is
    # grounded in what the tools actually returned.
    from ..api.schemas import ChatMessage

    sanitized = [
        m
        for m in body.messages
        if m.role in ("user", "system") or (m.role == "assistant" and m.content)
    ]
    tool_notes = []
    for m in body.messages:
        if m.role == "assistant" and m.tool_calls:
            names = ", ".join(
                str((tc.get("function") or {}).get("name") or "?")
                for tc in m.tool_calls
                if isinstance(tc, dict)
            )
            if names:
                tool_notes.append(f"- called tool(s): {names}")
        elif m.role == "tool" and m.content:
            tool_notes.append(f"- tool result: {m.content}")
    if tool_notes:
        sanitized.append(
            ChatMessage(
                role="user",
                content="Context from tool calls already executed:\n" + "\n".join(tool_notes),
            )
        )
    if not sanitized:
        sanitized = [m for m in body.messages if m.role == "user"]
    synth_body = body.model_copy(
        update={"messages": sanitized, "tools": None, "tool_choice": None}
    )

    final_result = await run_synthesis(
        run_id, preset, synth_body, panel_results, judge_analysis, {}
    )

    judge_cost = judge_analysis.cost_usd if judge_analysis else 0.0

    if body.stream:
        first_chunk = await final_result.__anext__()
        _fusion_model = f"fusion/{preset.name}"

        def _sse(chunk):
            data = json.loads(chunk.model_dump_json())
            data["model"] = _fusion_model
            return json.dumps(data)

        async def gen():
            completion_text = ""
            err = None
            synth_usage = None
            try:
                if first_chunk.choices and first_chunk.choices[0].delta and first_chunk.choices[0].delta.content:
                    completion_text += first_chunk.choices[0].delta.content
                if getattr(first_chunk, "usage", None):
                    synth_usage = first_chunk.usage
                yield f"data: {_sse(first_chunk)}\n\n"
                async for ch in final_result:
                    if ch.choices and ch.choices[0].delta and ch.choices[0].delta.content:
                        completion_text += ch.choices[0].delta.content
                    if getattr(ch, "usage", None):
                        synth_usage = ch.usage
                    yield f"data: {_sse(ch)}\n\n"
                if wants_usage(body):
                    s_pt, s_ct = _usage_tokens(synth_usage)
                    jp = int(getattr(judge_analysis, "prompt_tokens", 0) or 0) if judge_analysis else 0
                    jc = int(getattr(judge_analysis, "completion_tokens", 0) or 0) if judge_analysis else 0
                    yield usage_chunk_sse(_fusion_model, panel_pt + jp + s_pt, panel_ct + jc + s_ct)
                yield "data: [DONE]\n\n"
            except Exception as exc:
                err = exc
                logger.error(f"tool-fusion final stream error {run_id}: {exc}", exc_info=True)
            finally:
                synth_cost = _final_result_cost(final_result)
                trace = FusionTrace(
                    run_id=run_id, preset=preset.name,
                    cost_usd=panel_cost + judge_cost + synth_cost,
                    wall_ms=int((time.time() - start_time) * 1000),
                    degraded=err is not None, panel_results=panel_results,
                    judge_analysis=judge_analysis, final_answer=completion_text,
                    metadata=trace_metadata_for_preset(preset),
                )
                await save_trace(trace, body.store, key_hash)
            if err is not None:
                raise err

        return StreamingResponse(gen(), media_type="text/event-stream")

    content = final_result.choices[0].message.content
    synth_cost = _final_result_cost(final_result)
    trace = FusionTrace(
        run_id=run_id, preset=preset.name,
        cost_usd=panel_cost + judge_cost + synth_cost,
        wall_ms=int((time.time() - start_time) * 1000),
        degraded=False, panel_results=panel_results,
        judge_analysis=judge_analysis, final_answer=content,
        metadata=trace_metadata_for_preset(preset),
    )
    await save_trace(trace, body.store, key_hash)
    # Aggregate usage across panel + judge + final synthesis (no longer hardcoded 0).
    judge_pt = int(getattr(judge_analysis, "prompt_tokens", 0) or 0) if judge_analysis else 0
    judge_ct = int(getattr(judge_analysis, "completion_tokens", 0) or 0) if judge_analysis else 0
    synth_pt, synth_ct = _usage_tokens(getattr(final_result, "usage", None))
    usage = _usage_block(panel_pt + judge_pt + synth_pt, panel_ct + judge_ct + synth_ct)
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"fusion/{preset.name}",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": usage,
    }
