import asyncio
import json
import logging
import time
from typing import Optional, Dict

from ..store.presets import get_preset

logger = logging.getLogger("omnifusion.admin.jobs")


class PlaygroundJob:
    def __init__(self, run_id: str, task: asyncio.Task):
        self.run_id = run_id
        self.task = task
        self.queue = asyncio.Queue()
        self.created_at = time.time()
        self.last_connected = time.time()
        self.is_connected = True
        self.cancelled = False
        self.completed = False
        self.finished_at = None  # set once terminal; used by sweep() to prune

    async def abort(self):
        self.cancelled = True
        self.finished_at = time.time()
        if not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass


class JobRegistry:
    def __init__(self, max_jobs: int = 100):
        self.jobs: Dict[str, PlaygroundJob] = {}
        self.max_jobs = max_jobs
        self._lock = asyncio.Lock()

    async def register(self, run_id: str, task: asyncio.Task) -> PlaygroundJob:
        async with self._lock:
            # Enforce max job count by evicting a FINISHED or abandoned job. Never
            # cancel a live, connected, still-running job to make room — refuse the
            # new run instead (only reachable at an absurd 100 concurrent live jobs).
            if len(self.jobs) >= self.max_jobs:
                now = time.time()
                removable = None
                oldest = None
                for k, j in self.jobs.items():
                    if j.completed or j.cancelled or (now - j.last_connected > 300):
                        if oldest is None or j.created_at < oldest:
                            oldest = j.created_at
                            removable = k
                if removable is not None:
                    evicted = self.jobs.pop(removable)
                    if not evicted.task.done():
                        evicted.task.cancel()
                else:
                    raise RuntimeError(
                        "Too many concurrent playground runs; try again shortly."
                    )

            job = PlaygroundJob(run_id, task)
            self.jobs[run_id] = job
            return job

    async def get(self, run_id: str) -> Optional[PlaygroundJob]:
        async with self._lock:
            return self.jobs.get(run_id)

    async def cancel(self, run_id: str):
        job = await self.get(run_id)
        if job:
            await job.abort()

    async def sweep(self):
        """Cancels disconnected runs past their grace period AND prunes finished jobs
        so the registry can't grow unbounded."""
        async with self._lock:
            now = time.time()
            to_delete = []
            for k, j in list(self.jobs.items()):
                if j.completed or j.cancelled:
                    # Stamp finish time lazily, then prune ~60s later (lets any
                    # still-attached SSE reader drain the final events first).
                    if j.finished_at is None:
                        j.finished_at = now
                    elif now - j.finished_at > 60:
                        to_delete.append(k)
                elif not j.is_connected:
                    if now - j.last_connected > 15:  # 15 seconds grace period
                        j.cancelled = True
                        j.finished_at = now
                        j.task.cancel()
            for k in to_delete:
                del self.jobs[k]


job_registry = JobRegistry()


async def run_playground_job(run_id: str, preset_name: str, prompt: str, key_hash: str):
    job = await job_registry.get(run_id)
    if not job:
        return

    try:
        preset = await get_preset(preset_name)
        if not preset:
            await job.queue.put(
                {"event": "error", "data": f"Preset {preset_name} not found"}
            )
            job.completed = True
            return

        # 1. Initialize budget
        ceiling_micro_usd = (
            int(preset.cost_ceiling * 1_000_000)
            if preset.cost_ceiling is not None
            else None
        )
        from ..budget.ledger import initialize_request_budget

        await initialize_request_budget(run_id, ceiling_micro_usd)

        # 2. Run Panel
        await job.queue.put(
            {
                "event": "panel_started",
                "data": json.dumps({"models": preset.panel_models}),
            }
        )
        messages = [{"role": "user", "content": prompt}]

        from ..fusion.panel import run_panelist

        # Create explicit tasks so we retain handles and can cancel any that are
        # still in flight if the job is cancelled or the disconnect grace expires.
        # Otherwise asyncio.as_completed schedules child tasks we can't reach, and
        # they keep running (and spending) after the parent task is cancelled.
        panel_tasks = [
            asyncio.create_task(run_panelist(run_id, model, preset, messages))
            for model in preset.panel_models[:8]
        ]

        panel_results = []
        try:
            for fut in asyncio.as_completed(panel_tasks):
                try:
                    res = await fut
                    panel_results.append(res)
                    await job.queue.put(
                        {
                            "event": "panel_result",
                            "data": json.dumps(
                                {
                                    "model": res.model,
                                    "status": res.status,
                                    "content": res.content,
                                    "cost_usd": res.cost_usd,
                                }
                            ),
                        }
                    )
                except Exception as e:
                    from ..secrets.redact import redactor

                    await job.queue.put(
                        {
                            "event": "panel_result",
                            "data": json.dumps(
                                {
                                    "model": "unknown",
                                    "status": "error",
                                    "content": redactor.redact(str(e)),
                                    "cost_usd": 0.0,
                                }
                            ),
                        }
                    )
        finally:
            # Cancel any panelists still running (e.g. on cancellation mid-fan-out).
            # Each panelist reconciles its budget reservation in a shielded finally,
            # so cancelling here stops further spend without leaking reservations.
            for t in panel_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*panel_tasks, return_exceptions=True)

        ok_count = sum(1 for r in panel_results if r.status == "ok")
        min_success = getattr(preset, "min_panel_success", 1)
        if ok_count < min_success:
            await job.queue.put(
                {
                    "event": "error",
                    "data": f"Panel only got {ok_count} successes, needed {min_success}",
                }
            )
            job.completed = True
            return

        # 3. Run Judge
        await job.queue.put(
            {
                "event": "judge_started",
                "data": json.dumps({"model": preset.judge_model}),
            }
        )
        from ..fusion.judge import run_judge

        judge_analysis = await run_judge(run_id, preset, messages, panel_results)
        await job.queue.put(
            {
                "event": "judge_result",
                "data": json.dumps(
                    {
                        "consensus": judge_analysis.consensus,
                        "disagreements": judge_analysis.disagreements,
                        "strongest_points_by_model": judge_analysis.strongest_points_by_model,
                        "missing_information": judge_analysis.missing_information,
                        "likely_errors": judge_analysis.likely_errors,
                        "recommended_final_answer_plan": judge_analysis.recommended_final_answer_plan,
                    }
                ),
            }
        )

        # 4. Run Synth (with streaming)
        await job.queue.put(
            {
                "event": "final_started",
                "data": json.dumps({"model": preset.final_model}),
            }
        )

        from ..api.schemas import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            model=f"fusion/{preset_name}",
            messages=[ChatMessage(role="user", content=prompt)],
            stream=True,
            store=True,
        )

        from ..fusion.synth import run_synthesis

        context = {}
        final_stream = await run_synthesis(
            run_id, preset, req, panel_results, judge_analysis, context
        )

        final_text = ""
        async for chunk in final_stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    final_text += delta.content
                    # Fix (medium): JSON-encode delta content to prevent SSE framing
                    # corruption from newlines in the model's output.
                    await job.queue.put(
                        {
                            "event": "final_delta",
                            "data": json.dumps({"content": delta.content}),
                        }
                    )

        # Persist the trace so the run appears in the admin Runs history. The req
        # was built with store=True; run_synthesis only reconciles cost, so the
        # trace must be saved explicitly here (mirrors orchestrator.run_fusion).
        from ..fusion.types import FusionTrace
        from ..store.runs import save_trace

        panel_cost = sum(r.cost_usd for r in panel_results)
        judge_cost = judge_analysis.cost_usd if judge_analysis else 0.0
        # M3a removed context["cost_usd"]; the streamed synthesis exposes its
        # reconciled cost on the BudgetedStream after consumption (mirrors
        # orchestrator._final_result_cost), so read it from there instead of the
        # now-never-written context dict.
        synth_cost = getattr(final_stream, "cost_usd", 0.0)
        total_cost = panel_cost + judge_cost + synth_cost
        wall_ms = int((time.time() - job.created_at) * 1000)
        trace = FusionTrace(
            run_id=run_id,
            preset=preset.name,
            cost_usd=total_cost,
            wall_ms=wall_ms,
            degraded=False,
            panel_results=panel_results,
            judge_analysis=judge_analysis,
            final_answer=final_text,
        )
        await save_trace(trace, True, key_hash)

        await job.queue.put({"event": "done", "data": "[DONE]"})
        job.completed = True

    except asyncio.CancelledError:
        job.cancelled = True
        try:
            await job.queue.put({"event": "cancelled", "data": "Job was cancelled."})
        except Exception:
            pass
    except Exception as e:
        try:
            from ..secrets.redact import redactor

            await job.queue.put(
                {"event": "error", "data": f"Execution error: {redactor.redact(str(e))}"}
            )
        except Exception:
            pass
        job.completed = True
        job.finished_at = time.time()
