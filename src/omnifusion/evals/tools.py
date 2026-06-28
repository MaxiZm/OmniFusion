"""Tool-calling micro-bench — the thin internal runner M1b reserves for the case
where no off-the-shelf client fits.

Aider (M1b's canonical driver) owns the *coding-edit* loop, but it does not isolate
the one decision OmniFusion actually fuses on the agentic path: *which tool to call
next*. This minimal runner exercises that decision directly against OmniFusion's
OpenAI-compatible endpoint — one user prompt + a set of tool definitions in, the
selected `tool_call` out — and scores name (and optional argument) correctness.

Like the coding smoke, the output is raw pass/fail + cost + wall-time with NO
confidence intervals (CIs are theater at micro-bench N). Tier A `--mock` mode keeps
CI deterministic and never touches a provider; real mode is a Tier C probe.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path("evals/coding/aider_config.json")
DEFAULT_TOOL_TASKS = Path("evals/coding/tool_tasks.json")
DEFAULT_RUNS_DIR = Path("evals/coding/runs")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _selected_tool(message: dict[str, Any]) -> tuple[str | None, str]:
    """Return (tool_name, raw_arguments) for the first tool call, or (None, "")."""
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return None, ""
    function = tool_calls[0].get("function") or {}
    return function.get("name"), function.get("arguments") or ""


def score_task(task: dict[str, Any], selected_tool: str | None, arguments: str) -> dict[str, Any]:
    """Pure scoring: tool name must match; every expected arg-substring must appear."""
    expected_tool = task.get("expected_tool")
    name_ok = selected_tool == expected_tool
    arg_checks = []
    args_ok = True
    for needle in task.get("expected_arguments_contains", []):
        ok = needle in arguments
        arg_checks.append({"contains": needle, "passed": ok})
        args_ok = args_ok and ok
    return {
        "passed": bool(name_ok and args_ok),
        "name_ok": name_ok,
        "selected_tool": selected_tool,
        "expected_tool": expected_tool,
        "argument_checks": arg_checks,
    }


def run_mock_task(task: dict[str, Any]) -> dict[str, Any]:
    """Tier A contract mode: simulate the selected tool from the task fixture.

    This validates the runner/scoring contract, not model quality, so it is never
    benchmark evidence.
    """
    selected = task.get("mock_tool")
    arguments = task.get("mock_arguments", "")
    scored = score_task(task, selected, arguments)
    return {
        "id": task["id"],
        "category": task.get("category", "tool-selection"),
        "passed": scored["passed"],
        "cost_usd": 0.0,
        "wall_time_s": 0.0,
        "driver": "mock-contract",
        "scoring": scored,
    }


def _post_chat(config: dict[str, Any], task: dict[str, Any], api_key: str, timeout_s: int):
    base_url = config["base_url"].rstrip("/")
    body = json.dumps(
        {
            "model": config["model"],
            "messages": [{"role": "user", "content": task["prompt"]}],
            "tools": task["tools"],
            "tool_choice": task.get("tool_choice", "auto"),
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        run_id = response.headers.get("x-omnifusion-run-id")
        payload = json.loads(response.read().decode("utf-8"))
    return payload, run_id


def _trace_cost(config: dict[str, Any], run_id: str | None, api_key: str) -> float:
    """Best-effort cost lookup via the trace API (the body carries no cost field)."""
    if not run_id:
        return 0.0
    base_url = config["base_url"].rstrip("/")
    request = urllib.request.Request(
        f"{base_url}/traces/{run_id}",
        headers={"authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            trace = json.loads(response.read().decode("utf-8"))
        return float(trace.get("cost_usd", 0.0) or 0.0)
    except (urllib.error.URLError, ValueError, KeyError):
        return 0.0


def run_real_task(config: dict[str, Any], task: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    import os

    api_key = os.environ.get(config["api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"{config['api_key_env']} is required for non-mock tool evals")

    start = time.perf_counter()
    payload, run_id = _post_chat(config, task, api_key, timeout_s)
    wall_time_s = time.perf_counter() - start

    message = (payload.get("choices") or [{}])[0].get("message") or {}
    selected_tool, arguments = _selected_tool(message)
    scored = score_task(task, selected_tool, arguments)

    return {
        "id": task["id"],
        "category": task.get("category", "tool-selection"),
        "passed": scored["passed"],
        "cost_usd": round(_trace_cost(config, run_id, api_key), 6),
        "wall_time_s": round(wall_time_s, 3),
        "driver": "omnifusion-tool-runner",
        "run_id": run_id,
        "scoring": scored,
    }


def build_payload(config: dict[str, Any], task_results: list[dict[str, Any]], mock: bool) -> dict[str, Any]:
    total = len(task_results)
    passed = sum(1 for task in task_results if task["passed"])
    total_cost = sum(float(task["cost_usd"]) for task in task_results)
    total_wall = sum(float(task["wall_time_s"]) for task in task_results)
    return {
        "suite": "tool-smoke",
        "tier": "smoke",
        "driver": "mock-contract" if mock else "omnifusion-tool-runner",
        "model": config["model"],
        "base_url_env": config["base_url_env"],
        "api_key_env": config["api_key_env"],
        "tasks": task_results,
        "raw": {
            "passed": passed,
            "total": total,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "total_cost_usd": round(total_cost, 6),
            "total_wall_time_s": round(total_wall, 3),
        },
        "provenance": {
            "config": str(DEFAULT_CONFIG),
            "generated_by": "python -m omnifusion.evals.tools",
        },
    }


def run_suite(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    tasks = load_json(args.tasks)
    if len(tasks) > 20:
        raise RuntimeError("tool-smoke must contain at most 20 tasks")

    task_results = []
    for task in tasks:
        if args.mock:
            task_results.append(run_mock_task(task))
        else:
            task_results.append(run_real_task(config, task, args.timeout_s))

    payload = build_payload(config, task_results, args.mock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    raw = payload["raw"]
    print(
        f"tool-smoke: {raw['passed']}/{raw['total']} passed, "
        f"${raw['total_cost_usd']:.6f}, {raw['total_wall_time_s']:.3f}s"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OmniFusion tool-calling micro-bench.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TOOL_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUNS_DIR / "tool-smoke-latest.json")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.set_defaults(func=run_suite)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"tool eval failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
