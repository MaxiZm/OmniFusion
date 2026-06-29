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
from ..budget.ledger import initialize_request_budget
from ..api.errors import InsufficientPanelError
from ..api.schemas import ChatCompletionRequest
from ..api.normalize import generation_passthrough_kwargs
from ..api.sse import wants_usage
from .judge import run_judge, extract_json_from_text
from .synth import run_synthesis
from .runtime.executor import BudgetedExecutor
from .runtime.response import ResponseShaper
from .runtime.streaming import StreamingAdapter
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


def _trace_metadata(preset, web_sources) -> dict:
    metadata = trace_metadata_for_preset(preset)
    if web_sources:
        metadata["web_sources"] = web_sources
    return metadata


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


def _tool_names(tools) -> set:
    """Collect the function names declared in the request's `tools` array.

    `tools` here is the model_dump'd request list, i.e. dicts shaped like
    {"type": "function", "function": {"name": ...}}. Only function tools have a
    name the judge may legitimately emit; server tools (openrouter:*) do not.
    """
    names = set()
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        name = fn.get("name")
        if name:
            names.add(name)
    return names


def _sanitize_judge_tool_calls(judge_calls, valid_names, parallel_tool_calls) -> Optional[list]:
    """Validate + normalize judge-authored tool calls into OpenAI-shaped dicts.

    The judge may rewrite the panel's proposals into the final tool call(s). We must
    not trust that output blindly, so this:
      - keeps only function-type calls whose name is a declared request tool;
      - serializes dict/list arguments to a JSON string (judges often emit an object);
      - defaults missing/None arguments to "{}";
      - synthesizes a call id when absent;
      - drops anything malformed;
      - returns at most the first valid call when parallel_tool_calls is False.

    Returns a non-empty list of OpenAI-shaped tool_call dicts, or None if nothing
    usable remains (caller then falls back to the selected panel proposal).
    """
    if not isinstance(judge_calls, list) or not judge_calls:
        return None
    out = []
    for tc in judge_calls:
        if not isinstance(tc, dict):
            continue
        # Accept only function calls. A missing type defaults to "function" since
        # that is the only call shape OpenAI tool calling defines.
        if tc.get("type", "function") != "function":
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name or name not in valid_names:
            continue
        args = fn.get("arguments")
        if isinstance(args, (dict, list)):
            args = json.dumps(args)
        elif args is None:
            args = "{}"
        elif not isinstance(args, str):
            # Numbers/bools/etc. — coerce to a JSON string defensively.
            args = json.dumps(args)
        out.append(
            {
                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )
    if not out:
        return None
    if parallel_tool_calls is False:
        return out[:1]
    return out


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


async def _panel_propose(run_id, preset, dict_messages, tools, tool_choice, gen_kwargs=None):
    """Each panel model proposes its next action (tool call or text), in parallel."""

    executor = BudgetedExecutor(run_id)
    gen_kwargs = gen_kwargs or {}

    async def one(model):
        # Reserve/reconcile is owned by the executor (M3a single-shield invariant):
        # tool orchestration must not run a second model-call reconciliation path.
        try:
            resp = await executor.call(
                f"tool-panel/{model}",
                provider_id=preset.provider_id_for(model, "panel"),
                model=model,
                messages=dict_messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=preset.panel.max_tokens,
                timeout=preset.panel.timeout,
                **gen_kwargs,
                **_disable_thinking_kwargs(model),
            )
            actual = getattr(resp, "_omnifusion_cost_usd", 0.0)
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


async def _decide_next_step(run_id, preset, dict_messages, ok_proposals, tools=None) -> dict:
    """Judge selects the best next action and authors the final tool call(s).

    Returns {"decision", "best_index", "tool_calls", "cost", "prompt_tokens",
    "completion_tokens"} so the coordinating judge call's usage/cost is aggregated,
    not dropped. "tool_calls" is the judge's raw authored call list (or None for a
    legacy best_index-only response); the caller sanitizes it against the request
    tools and falls back to the selected panel proposal when it is unusable.
    """
    # If nobody proposed a tool call, the step is necessarily a final answer.
    any_tool = any(p.get("tool_calls") for p in ok_proposals)
    if not any_tool:
        return {
            "decision": "final",
            "best_index": 0,
            "tool_calls": None,
            "cost": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    descriptions = "\n".join(
        _describe_proposal(i, p) for i, p in enumerate(ok_proposals)
    )
    # Use the last user/tool turn as task context.
    last_ctx = ""
    for m in reversed(dict_messages):
        if m.get("role") in ("user", "tool") and m.get("content"):
            last_ctx = str(m.get("content"))[:1500]
            break

    tool_names = sorted(_tool_names(tools))
    available_tools = ", ".join(tool_names) if tool_names else "(see agent proposals)"

    judge_prompt = (
        "You are the coordinator of a panel of agents solving a task that may require "
        "tools (functions). Below is the latest task context and each agent's proposed "
        "NEXT step. Choose the single best next step that most correctly advances the "
        "task toward a complete, accurate solution.\n\n"
        "Rules:\n"
        "- If a tool call is the best next step, set decision \"tool\" and AUTHOR the "
        "final tool call(s) yourself in \"tool_calls\": start from the best agent's "
        "proposal and CORRECT the name or arguments where they are wrong, incomplete, "
        "or could be improved. Use only tool names from AVAILABLE TOOLS. Each call's "
        "\"arguments\" must be a JSON object encoded as a string.\n"
        "- Set \"best_index\" to the agent whose proposal you based your call on.\n"
        "- If the task is already solved and the agents should now produce a final "
        "answer, choose decision \"final\".\n\n"
        f"AVAILABLE TOOLS: {available_tools}\n\n"
        f"TASK CONTEXT:\n{last_ctx}\n\n"
        f"PROPOSED NEXT STEPS:\n{descriptions}\n\n"
        'Output ONLY JSON: {"decision":"tool"|"final","best_index":<agent number>,'
        '"tool_calls":[{"type":"function","function":{"name":"<tool>",'
        '"arguments":"<json-object-string>"}}],"reasoning":"<one sentence>"}'
    )

    messages = [{"role": "user", "content": judge_prompt}]
    executor = BudgetedExecutor(run_id)
    judge_provider = preset.provider_id_for(preset.judge_model, "judge")
    cost = 0.0
    judge_pt = 0
    judge_ct = 0
    judge_tool_calls = None
    try:
        # Request JSON mode; if a model rejects it, retry without (the extractor is
        # robust either way). filter_params drops it for providers that don't support it.
        # Reserve/reconcile is owned by the executor (M3a single-shield invariant).
        kwargs = {
            "timeout": preset.judge.timeout,
            "response_format": {"type": "json_object"},
            **_disable_thinking_kwargs(preset.judge_model),
        }
        try:
            resp = await executor.call(
                "tool-judge",
                provider_id=judge_provider,
                model=preset.judge_model,
                messages=messages,
                max_tokens=preset.judge.max_tokens,
                **kwargs,
            )
        except Exception:
            kwargs.pop("response_format", None)
            resp = await executor.call(
                "tool-judge",
                provider_id=judge_provider,
                model=preset.judge_model,
                messages=messages,
                max_tokens=preset.judge.max_tokens,
                **kwargs,
            )
        cost = getattr(resp, "_omnifusion_cost_usd", 0.0)
        judge_pt, judge_ct = _usage_tokens(getattr(resp, "usage", None))
        data = extract_json_from_text(resp.choices[0].message.content)
        decision = data.get("decision", "tool")
        best_index = int(data.get("best_index", 0))
        # Raw judge-authored calls (sanitized by the caller). Legacy responses with
        # only best_index leave this None, which triggers the panel-proposal fallback.
        raw_calls = data.get("tool_calls")
        if isinstance(raw_calls, list):
            judge_tool_calls = raw_calls
    except Exception as e:
        logger.warning(f"tool-judge failed, defaulting to first tool proposal: {e}")
        decision, best_index = "tool", 0

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
    return {
        "decision": decision,
        "best_index": best_index,
        "tool_calls": judge_tool_calls,
        "cost": cost,
        "prompt_tokens": judge_pt,
        "completion_tokens": judge_ct,
    }


def _usage_block(prompt_tokens: int, completion_tokens: int) -> dict:
    return ResponseShaper.usage_block(prompt_tokens, completion_tokens)


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
    web_sources = []

    if getattr(preset, "web_enabled", False):
        from .web_grounding import gather_web_context, inject_grounding, latest_user_text

        web_context = await gather_web_context(run_id, latest_user_text(dict_messages))
        web_sources = web_context.sources
        if web_context.has_grounding:
            dict_messages = inject_grounding(dict_messages, web_context.grounding_text)

    # 1. Panel proposes next actions (parallel). Forward caller generation params.
    gen_kwargs = generation_passthrough_kwargs(body, include_tool_params=True)
    proposals = await _panel_propose(
        run_id, preset, dict_messages, tools, tool_choice, gen_kwargs
    )
    ok = [p for p in proposals if p.get("ok")]
    if not ok:
        raise InsufficientPanelError(
            "No panel model could propose a next step (do the panel models support tools?)"
        )

    panel_cost = sum(p.get("cost", 0.0) for p in proposals)

    # 2. Judge selects the best next action. Its cost/tokens are aggregated below so
    # the coordinating tool-judge call is not dropped from the trace/usage.
    decision = await _decide_next_step(run_id, preset, dict_messages, ok, tools)
    step_judge_cost = decision.get("cost", 0.0)

    # Aggregate panel + tool-judge token usage so responses report real usage.
    panel_pt = sum(p.get("prompt_tokens", 0) for p in proposals) + decision.get("prompt_tokens", 0)
    panel_ct = sum(p.get("completion_tokens", 0) for p in proposals) + decision.get("completion_tokens", 0)

    # 3a. Tool step: emit the judge-authored tool call(s); client executes and loops
    # back. The judge rewrites the panel proposals into a corrected final call; if its
    # output is unusable (unknown tool, malformed, or legacy best_index-only), fall
    # back to the selected panel proposal so the loop never stalls.
    if decision["decision"] == "tool":
        chosen = ok[decision["best_index"]]
        judge_authored = _sanitize_judge_tool_calls(
            decision.get("tool_calls"),
            _tool_names(tools),
            getattr(body, "parallel_tool_calls", None),
        )
        if judge_authored:
            tool_calls = judge_authored
            consensus = f"Judge-authored tool call (based on {chosen['model']}'s proposal)"
        else:
            tool_calls = chosen["tool_calls"]
            consensus = f"Selected tool call from {chosen['model']}"
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
            cost_usd=panel_cost + step_judge_cost,
            wall_ms=int((time.time() - start_time) * 1000),
            degraded=False,
            panel_results=panel_results,
            judge_analysis=JudgeAnalysis(
                consensus=consensus,
            ),
            final_answer=None,
            metadata=_trace_metadata(preset, web_sources),
        )
        await save_trace(trace, body.store, key_hash)

        adapter = StreamingAdapter(f"fusion/{preset.name}")
        if body.stream:
            # Emit a terminal usage chunk when the client opted in (e.g. /v1/responses
            # always does), so a tool-call turn still reports usage.
            stream_usage = (panel_pt, panel_ct) if wants_usage(body) else None
            return StreamingResponse(
                adapter.tool_call_sse(tool_calls, usage=stream_usage),
                media_type="text/event-stream",
            )
        return ResponseShaper.tool_call_completion(
            model=f"fusion/{preset.name}",
            tool_calls=tool_calls,
            usage=_usage_block(panel_pt, panel_ct),
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
            # body.messages tool_calls are typed ToolCall objects (M1c), not dicts;
            # normalize through the shared helper so multi-turn tool context is not
            # silently dropped from the synthesis prompt.
            normalized = _normalize_tool_calls(m.tool_calls) or []
            names = ", ".join(
                str(tc["function"].get("name") or "?") for tc in normalized
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
        adapter = StreamingAdapter(_fusion_model)

        async def gen():
            completion_text = ""
            err = None
            synth_usage = None
            try:
                if first_chunk.choices and first_chunk.choices[0].delta and first_chunk.choices[0].delta.content:
                    completion_text += first_chunk.choices[0].delta.content
                if getattr(first_chunk, "usage", None):
                    synth_usage = first_chunk.usage
                yield adapter.chunk_sse(first_chunk)
                async for ch in final_result:
                    if ch.choices and ch.choices[0].delta and ch.choices[0].delta.content:
                        completion_text += ch.choices[0].delta.content
                    if getattr(ch, "usage", None):
                        synth_usage = ch.usage
                    yield adapter.chunk_sse(ch)
                if wants_usage(body):
                    s_pt, s_ct = _usage_tokens(synth_usage)
                    jp = int(getattr(judge_analysis, "prompt_tokens", 0) or 0) if judge_analysis else 0
                    jc = int(getattr(judge_analysis, "completion_tokens", 0) or 0) if judge_analysis else 0
                    yield adapter.usage_sse(panel_pt + jp + s_pt, panel_ct + jc + s_ct)
                yield adapter.done_sse()
            except Exception as exc:
                err = exc
                logger.error(f"tool-fusion final stream error {run_id}: {exc}", exc_info=True)
            finally:
                synth_cost = _final_result_cost(final_result)
                trace = FusionTrace(
                    run_id=run_id, preset=preset.name,
                    cost_usd=panel_cost + step_judge_cost + judge_cost + synth_cost,
                    wall_ms=int((time.time() - start_time) * 1000),
                    degraded=err is not None, panel_results=panel_results,
                    judge_analysis=judge_analysis, final_answer=completion_text,
                    metadata=_trace_metadata(preset, web_sources),
                )
                await save_trace(trace, body.store, key_hash)
            if err is not None:
                raise err

        return StreamingResponse(gen(), media_type="text/event-stream")

    content = final_result.choices[0].message.content
    synth_cost = _final_result_cost(final_result)
    trace = FusionTrace(
        run_id=run_id, preset=preset.name,
        cost_usd=panel_cost + step_judge_cost + judge_cost + synth_cost,
        wall_ms=int((time.time() - start_time) * 1000),
        degraded=False, panel_results=panel_results,
        judge_analysis=judge_analysis, final_answer=content,
        metadata=_trace_metadata(preset, web_sources),
    )
    await save_trace(trace, body.store, key_hash)
    # Aggregate usage across panel + judge + final synthesis (no longer hardcoded 0).
    judge_pt = int(getattr(judge_analysis, "prompt_tokens", 0) or 0) if judge_analysis else 0
    judge_ct = int(getattr(judge_analysis, "completion_tokens", 0) or 0) if judge_analysis else 0
    synth_pt, synth_ct = _usage_tokens(getattr(final_result, "usage", None))
    usage = _usage_block(panel_pt + judge_pt + synth_pt, panel_ct + judge_ct + synth_ct)
    return ResponseShaper.chat_completion(
        model=f"fusion/{preset.name}",
        content=content,
        usage=usage,
        finish_reason="stop",
    )
