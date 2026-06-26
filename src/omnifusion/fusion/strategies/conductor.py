from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from omnifusion.api.errors import InsufficientPanelError, OmniFusionError
from omnifusion.api.schemas import ChatCompletionRequest
from omnifusion.budget.ledger import initialize_request_budget
from omnifusion.fusion.judge import extract_json_from_text
from omnifusion.fusion.runtime.artifacts import ArtifactGraph
from omnifusion.fusion.runtime.executor import BudgetedExecutor
from omnifusion.fusion.runtime.response import ResponseShaper
from omnifusion.fusion.runtime.streaming import normalize_finish_reason
from omnifusion.fusion.runtime.context import RunContext
from omnifusion.fusion.runtime.strategy import FusionStrategy, StrategyResult
from omnifusion.fusion.runtime.bandit import select_panel_models
from omnifusion.fusion.types import (
    FusionTrace,
    JudgeAnalysis,
    PanelResult,
    Preset,
    trace_metadata_for_preset,
)
from omnifusion.settings import settings
from omnifusion.store.runs import save_trace


class ConductorStrategy(FusionStrategy):
    key = "conductor"

    async def execute(self, ctx: RunContext) -> StrategyResult:
        payload = await execute_conductor(
            ctx.run_id,
            ctx.preset,
            ctx.request,
            ctx.key_hash,
            artifacts=ctx.artifacts,
        )
        return StrategyResult(payload=payload)


def _message_dicts(request: ChatCompletionRequest) -> list[dict[str, Any]]:
    return [message.model_dump(exclude_none=True) for message in request.messages]


def _response_text(response) -> str:
    if not getattr(response, "choices", None):
        return ""
    return getattr(response.choices[0].message, "content", "") or ""


def _usage_tokens(response) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _finish_reason(response) -> str:
    if not getattr(response, "choices", None):
        return "stop"
    return normalize_finish_reason(getattr(response.choices[0], "finish_reason", "stop"))


def _render_artifacts(
    *,
    plan: str,
    worker_results: list[PanelResult],
    verifier_text: str = "",
    repair_texts: list[str] | None = None,
) -> str:
    repairs = repair_texts or []
    payload = {
        "plan": plan,
        "workers": [
            {"model": result.model, "status": result.status, "content": result.content}
            for result in worker_results
        ],
        "verifier": verifier_text,
        "repairs": repairs,
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _judge_analysis_from_verifier(content: str, usage_tokens: tuple[int, int]) -> tuple[JudgeAnalysis, dict]:
    try:
        data = extract_json_from_text(content)
    except ValueError:
        data = {"consensus": content}

    prompt_tokens, completion_tokens = usage_tokens
    analysis = JudgeAnalysis(
        consensus=data.get("consensus", ""),
        disagreements=data.get("disagreements", data.get("contradictions", "")),
        contradictions=data.get("contradictions", data.get("disagreements", "")),
        partial_coverage=data.get("partial_coverage", ""),
        unique_insights=data.get("unique_insights", {}),
        blind_spots=data.get("blind_spots", ""),
        model_strengths=data.get("model_strengths", {}),
        synthesis_plan=data.get(
            "synthesis_plan",
            data.get("recommended_final_answer_plan", ""),
        ),
        recommended_final_answer_plan=data.get(
            "recommended_final_answer_plan",
            data.get("synthesis_plan", ""),
        ),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return analysis, data


async def execute_conductor(
    run_id: str,
    preset: Preset,
    request: ChatCompletionRequest,
    key_hash: str,
    artifacts: ArtifactGraph | None = None,
):
    artifacts = artifacts if artifacts is not None else ArtifactGraph()
    if request.stream:
        raise OmniFusionError(
            "conductor strategy streaming is not supported yet.",
            status_code=400,
            type_="invalid_request_error",
            code="unsupported_conductor_stream",
        )
    start_time = time.time()
    ceiling_micro_usd = (
        int(preset.cost_ceiling * 1_000_000)
        if preset.cost_ceiling is not None
        else None
    )
    await initialize_request_budget(run_id, ceiling_micro_usd)

    executor = BudgetedExecutor(run_id)
    base_messages = _message_dicts(request)
    usage_prompt = 0
    usage_completion = 0
    total_cost = 0.0
    stage_names: list[str] = []

    async def call_stage(stage: str, *, model: str, messages: list[dict[str, Any]], max_tokens: int, timeout: int):
        nonlocal usage_prompt, usage_completion, total_cost
        stage_names.append(stage)
        response = await executor.call(
            stage,
            provider_id=preset.provider_id_for(model),
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        prompt_tokens, completion_tokens = _usage_tokens(response)
        usage_prompt += prompt_tokens
        usage_completion += completion_tokens
        total_cost += float(getattr(response, "_omnifusion_cost_usd", 0.0) or 0.0)
        return response

    plan_response = await call_stage(
        "plan",
        model=preset.judge_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Plan the answer before worker execution. This is an "
                    "experimental transparent conductor approximation."
                ),
            },
            *base_messages,
        ],
        max_tokens=preset.judge.max_tokens,
        timeout=preset.judge.timeout,
    )
    plan = _response_text(plan_response)

    async def run_worker(model: str) -> PanelResult:
        try:
            response = await call_stage(
                f"worker/{model}",
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Follow this conductor plan:\n\n{plan}",
                    },
                    *base_messages,
                ],
                max_tokens=preset.panel.max_tokens,
                timeout=preset.panel.timeout,
            )
            return PanelResult(
                model=model,
                status="ok",
                content=_response_text(response),
                usage=getattr(response, "usage", None),
                cost_usd=float(getattr(response, "_omnifusion_cost_usd", 0.0) or 0.0),
            )
        except Exception:
            return PanelResult(model=model, status="error")

    worker_results = await asyncio.gather(
        *[run_worker(model) for model in select_panel_models(preset, max_count=8)]
    )
    ok_workers = [result for result in worker_results if result.status == "ok"]
    if len(ok_workers) < preset.min_panel_success:
        raise InsufficientPanelError(
            f"Conductor workers got {len(ok_workers)} successes, needed {preset.min_panel_success}"
        )

    verifier_response = await call_stage(
        "verify",
        model=preset.judge_model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Verify the plan and worker drafts. Return JSON with "
                    "consensus, contradictions, synthesis_plan, needs_repair, "
                    "and repair_instructions.\n\n"
                    + _render_artifacts(plan=plan, worker_results=worker_results)
                ),
            }
        ],
        max_tokens=preset.judge.max_tokens,
        timeout=preset.judge.timeout,
    )
    verifier_text = _response_text(verifier_response)
    judge_analysis, verifier_data = _judge_analysis_from_verifier(
        verifier_text,
        _usage_tokens(verifier_response),
    )

    repair_texts: list[str] = []
    needs_repair = bool(verifier_data.get("needs_repair", False))
    repair_instructions = str(verifier_data.get("repair_instructions", "") or "")
    max_repairs = max(0, int(settings.omnifusion_conductor_max_repairs))
    for repair_index in range(max_repairs):
        if not needs_repair:
            break
        repair_response = await call_stage(
            f"repair/{repair_index + 1}",
            model=preset.final_model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Repair the draft using these verifier instructions:\n\n"
                        f"{repair_instructions}\n\n"
                        + _render_artifacts(
                            plan=plan,
                            worker_results=worker_results,
                            verifier_text=verifier_text,
                            repair_texts=repair_texts,
                        )
                    ),
                }
            ],
            max_tokens=preset.final.max_tokens,
            timeout=preset.final.timeout,
        )
        repair_texts.append(_response_text(repair_response))
        needs_repair = False

    merge_response = await call_stage(
        "merge",
        model=preset.final_model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Merge the verified worker drafts into the final answer.\n\n"
                    + _render_artifacts(
                        plan=plan,
                        worker_results=worker_results,
                        verifier_text=verifier_text,
                        repair_texts=repair_texts,
                    )
                ),
            },
            *base_messages,
        ],
        max_tokens=preset.final.max_tokens,
        timeout=preset.final.timeout,
    )
    final_answer = _response_text(merge_response)
    wall_ms = int((time.time() - start_time) * 1000)

    # Record the conductor's stage graph as bounded artifacts so the trace shows
    # plan/workers/verifier/repairs/merger (M6) without persisting full bodies.
    artifacts.add("plan_chars", len(plan))
    artifacts.add(
        "workers",
        [{"model": r.model, "status": r.status} for r in worker_results],
    )
    artifacts.add("verifier_requested_repair", bool(verifier_data.get("needs_repair", False)))
    artifacts.add("repair_count", len(repair_texts))
    artifacts.add("final_chars", len(final_answer))

    metadata = trace_metadata_for_preset(preset)
    metadata["conductor"] = {
        "experimental": True,
        "ablation_required": True,
        "stages": stage_names,
        "repair_count": len(repair_texts),
        "verifier_requested_repair": bool(verifier_data.get("needs_repair", False)),
    }
    metadata["artifacts"] = artifacts.to_trace_metadata()
    trace = FusionTrace(
        run_id=run_id,
        preset=preset.name,
        cost_usd=total_cost,
        wall_ms=wall_ms,
        degraded=False,
        panel_results=worker_results,
        judge_analysis=judge_analysis,
        final_answer=final_answer,
        metadata=metadata,
    )
    await save_trace(trace, request.store, key_hash)

    return ResponseShaper.chat_completion(
        model=f"fusion/{preset.name}",
        content=final_answer,
        usage={
            "prompt_tokens": usage_prompt,
            "completion_tokens": usage_completion,
            "total_tokens": usage_prompt + usage_completion,
        },
        finish_reason=_finish_reason(merge_response),
    )
