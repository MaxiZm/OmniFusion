from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path("evals/coding/aider_config.json")
DEFAULT_SMOKE_TASKS = Path("evals/coding/smoke_tasks.json")
DEFAULT_FULL_TASKS = Path("evals/coding/full_tasks.json")
DEFAULT_RUNS_DIR = Path("evals/coding/runs")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def task_suite_checksum(tasks_path: Path) -> str:
    """A content checksum of the task manifest, pinned in baseline provenance so a
    promoted dated artifact records exactly which tasks were run."""
    return "sha256:" + hashlib.sha256(Path(tasks_path).read_bytes()).hexdigest()


def expected_passthrough_model(config: dict[str, Any]) -> str:
    """The preset a client model name must resolve to (e.g. openai/fusion/general
    -> fusion/general)."""
    model = config["model"]
    if model.startswith("openai/"):
        model = model[len("openai/") :]
    return model


def preflight_model_passthrough(config: dict[str, Any]) -> None:
    """The plan requires the smoke to verify model-name pass-through BEFORE any task
    runs. Posts a single chat request and asserts the configured client model name
    resolves to the intended fusion preset, aborting the run on mismatch."""
    base_url = config["base_url"].rstrip("/")
    api_key = os.environ.get(config["api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"{config['api_key_env']} is required for the smoke preflight")
    body = json.dumps(
        {
            "model": config["model"],
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"content-type": "application/json", "authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    expected = expected_passthrough_model(config)
    actual = payload.get("model")
    if actual != expected:
        raise RuntimeError(
            f"model-name pass-through failed: {config['model']} resolved to "
            f"{actual!r}, expected {expected!r}"
        )


def task_files(task: dict[str, Any]) -> list[str]:
    return [file_info["path"] for file_info in task.get("files", [])]


def write_task_files(workdir: Path, task: dict[str, Any]) -> None:
    for file_info in task.get("files", []):
        target = workdir / file_info["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file_info.get("content", ""))


def validate_task(workdir: Path, task: dict[str, Any]) -> dict[str, Any]:
    validations = []
    passed = True
    for expected in task.get("expected", []):
        path = workdir / expected["path"]
        content = path.read_text() if path.exists() else ""
        contains = expected.get("contains", "")
        ok = path.exists() and contains in content
        validations.append(
            {
                "path": expected["path"],
                "contains": contains,
                "passed": ok,
            }
        )
        passed = passed and ok
    return {"passed": passed, "checks": validations}


def parse_aider_cost(output: str) -> float:
    cost_lines = [line for line in output.splitlines() if "cost" in line.lower()]
    costs = re.findall(r"\$([0-9]+(?:\.[0-9]+)?)", "\n".join(cost_lines))
    if not costs:
        return 0.0
    return float(costs[-1])


def build_aider_command(config: dict[str, Any], task: dict[str, Any]) -> list[str]:
    version = config["aider_chat_version"]
    command = [
        "uvx",
        "--from",
        f"aider-chat=={version}",
        "aider",
        "--model",
        config["model"],
        "--yes-always",
        "--message",
        task["prompt"],
    ]
    command.extend(task_files(task))
    return command


def run_mock_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task["id"],
        "language": task["language"],
        "passed": bool(task.get("mock_passed", False)),
        "cost_usd": 0.0,
        "wall_time_s": 0.0,
        "driver": "mock-contract",
        "validation": {"passed": False, "checks": []},
    }


def run_aider_task(
    config: dict[str, Any],
    task: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault(config["base_url_env"], config["base_url"])
    if not env.get(config["api_key_env"]):
        raise RuntimeError(
            f"{config['api_key_env']} is required for non-mock coding evals"
        )

    with tempfile.TemporaryDirectory(prefix="omnifusion-eval-") as tmpdir:
        workdir = Path(tmpdir)
        write_task_files(workdir, task)
        subprocess.run(
            ["git", "init", "-q"],
            cwd=workdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        command = build_aider_command(config, task)
        start = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=workdir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
        wall_time_s = time.perf_counter() - start
        validation = validate_task(workdir, task)
        passed = result.returncode == 0 and validation["passed"]

    return {
        "id": task["id"],
        "language": task["language"],
        "passed": passed,
        "cost_usd": parse_aider_cost(result.stdout),
        "wall_time_s": round(wall_time_s, 3),
        "driver": "aider",
        "returncode": result.returncode,
        "command": command,
        "validation": validation,
        "stdout_tail": result.stdout[-4000:],
    }


def wilson_interval(successes: int, total: int) -> dict[str, float | str]:
    if total == 0:
        return {"method": "wilson", "lower": 0.0, "upper": 0.0}
    z = 1.96
    phat = successes / total
    denominator = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
    lower = max(0.0, (center - spread) / denominator)
    upper = min(1.0, (center + spread) / denominator)
    return {"method": "wilson", "lower": round(lower, 4), "upper": round(upper, 4)}


def build_payload(
    suite: str,
    config: dict[str, Any],
    task_results: list[dict[str, Any]],
    mock: bool,
) -> dict[str, Any]:
    total = len(task_results)
    passed = sum(1 for task in task_results if task["passed"])
    total_cost = sum(float(task["cost_usd"]) for task in task_results)
    total_wall = sum(float(task["wall_time_s"]) for task in task_results)
    raw = {
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "total_wall_time_s": round(total_wall, 3),
    }
    if suite == "coding-full":
        raw["confidence_interval"] = wilson_interval(passed, total)
        raw["cost_normalization"] = {
            "usd_per_task": round(total_cost / total, 6) if total else 0.0,
            "solve_per_usd": round(passed / total_cost, 6) if total_cost else 0.0,
            "solve_per_wall_s": round(passed / total_wall, 6) if total_wall else 0.0,
        }

    return {
        "suite": suite,
        "tier": "C" if suite == "coding-full" else "smoke",
        "driver": "mock-contract" if mock else "aider",
        "aider_chat_version": config["aider_chat_version"],
        "model": config["model"],
        "base_url_env": config["base_url_env"],
        "api_key_env": config["api_key_env"],
        "tasks": task_results,
        "raw": raw,
        "provenance": {
            "config": str(DEFAULT_CONFIG),
            "docs": config.get("docs", {}),
            "generated_by": "python -m omnifusion.evals.coding",
            "expected_passthrough_model": expected_passthrough_model(config),
        },
    }


def _report_paths(output_path: Path) -> dict[str, str]:
    return {
        "json": str(output_path),
        "jsonl": str(output_path.with_suffix(".jsonl")),
        "markdown": str(output_path.with_suffix(".md")),
    }


def write_jsonl_report(path: Path, payload: dict[str, Any]) -> None:
    rows = [
        {
            "type": "task_result",
            "suite": payload["suite"],
            "model": payload["model"],
            "task": task,
        }
        for task in payload["tasks"]
    ]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )


def render_markdown_report(payload: dict[str, Any]) -> str:
    raw = payload["raw"]
    ci = raw.get("confidence_interval", {})
    cost_norm = raw.get("cost_normalization", {})
    ci_text = (
        f"{ci.get('lower', 0.0):.4f}-{ci.get('upper', 0.0):.4f} "
        f"({ci.get('method', 'n/a')})"
    )
    lines = [
        "# OmniFusion Coding Full Report",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Suite | `{payload['suite']}` |",
        f"| Driver | `{payload['driver']}` |",
        f"| Model | `{payload['model']}` |",
        f"| Passed | {raw['passed']}/{raw['total']} |",
        f"| Pass rate | {raw['pass_rate']:.4f} |",
        f"| 95% CI | {ci_text} |",
        f"| Total cost USD | {raw['total_cost_usd']:.6f} |",
        f"| Total wall time s | {raw['total_wall_time_s']:.3f} |",
        f"| usd_per_task | {cost_norm.get('usd_per_task', 0.0):.6f} |",
        f"| solve_per_usd | {cost_norm.get('solve_per_usd', 0.0):.6f} |",
        f"| solve_per_wall_s | {cost_norm.get('solve_per_wall_s', 0.0):.6f} |",
        "",
    ]
    if payload["driver"] == "mock-contract":
        lines.extend(["Mock outputs are not benchmark evidence.", ""])

    lines.extend(
        [
            "| Task | Language | Passed | Cost USD | Wall s |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for task in payload["tasks"]:
        lines.append(
            "| {id} | {language} | {passed} | {cost:.6f} | {wall:.3f} |".format(
                id=str(task["id"]).replace("|", "\\|"),
                language=str(task["language"]).replace("|", "\\|"),
                passed="yes" if task["passed"] else "no",
                cost=float(task["cost_usd"]),
                wall=float(task["wall_time_s"]),
            )
        )
    return "\n".join(lines) + "\n"


def write_full_reports(output_path: Path, payload: dict[str, Any]) -> None:
    write_jsonl_report(output_path.with_suffix(".jsonl"), payload)
    output_path.with_suffix(".md").write_text(render_markdown_report(payload))


def run_suite(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    tasks = load_json(args.tasks)
    suite_name = f"coding-{args.suite}"
    if args.suite == "smoke" and len(tasks) > 20:
        raise RuntimeError("coding-smoke must contain at most 20 tasks")

    # The smoke verifies model-name pass-through before any task runs (skipped in
    # mock mode, which never reaches a live server).
    if args.suite == "smoke" and not args.mock:
        preflight_model_passthrough(config)

    task_results = []
    for task in tasks:
        if args.mock:
            task_results.append(run_mock_task(task))
        else:
            task_results.append(run_aider_task(config, task, args.timeout_s))

    payload = build_payload(suite_name, config, task_results, args.mock)
    if suite_name == "coding-full":
        payload["reports"] = _report_paths(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if suite_name == "coding-full":
        write_full_reports(args.output, payload)

    raw = payload["raw"]
    print(
        f"{suite_name}: {raw['passed']}/{raw['total']} passed, "
        f"${raw['total_cost_usd']:.6f}, {raw['total_wall_time_s']:.3f}s"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OmniFusion coding evals.")
    subparsers = parser.add_subparsers(dest="suite", required=True)

    defaults = {
        "smoke": (DEFAULT_SMOKE_TASKS, DEFAULT_RUNS_DIR / "smoke-latest.json"),
        "full": (DEFAULT_FULL_TASKS, DEFAULT_RUNS_DIR / "full-latest.json"),
    }
    for suite, (tasks_path, output_path) in defaults.items():
        subparser = subparsers.add_parser(suite)
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        subparser.add_argument("--tasks", type=Path, default=tasks_path)
        subparser.add_argument("--output", type=Path, default=output_path)
        subparser.add_argument("--mock", action="store_true")
        subparser.add_argument("--timeout-s", type=int, default=600)
        subparser.set_defaults(func=run_suite)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"coding eval failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
