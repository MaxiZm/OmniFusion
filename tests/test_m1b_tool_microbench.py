"""M1b tool-calling micro-bench (the thin internal runner where no client fits)."""
import json
import os
import subprocess
import sys
from pathlib import Path

from omnifusion.evals import tools as tool_eval


def test_score_task_requires_name_and_argument_match():
    task = {
        "expected_tool": "get_weather",
        "expected_arguments_contains": ["Paris"],
    }
    good = tool_eval.score_task(task, "get_weather", '{"location": "Paris"}')
    assert good["passed"] is True

    wrong_tool = tool_eval.score_task(task, "send_email", '{"location": "Paris"}')
    assert wrong_tool["passed"] is False

    missing_arg = tool_eval.score_task(task, "get_weather", '{"location": "Berlin"}')
    assert missing_arg["passed"] is False


def test_selected_tool_reads_first_tool_call():
    message = {
        "tool_calls": [
            {"function": {"name": "calculate", "arguments": '{"expression": "7*6"}'}}
        ]
    }
    name, args = tool_eval._selected_tool(message)
    assert name == "calculate"
    assert "7*6" in args

    assert tool_eval._selected_tool({"content": "plain text"}) == (None, "")


def test_tool_tasks_fixture_is_bounded_and_well_formed():
    tasks = json.loads(Path("evals/coding/tool_tasks.json").read_text())
    assert 0 < len(tasks) <= 20
    for task in tasks:
        assert task["expected_tool"]
        assert task["tools"]
        # Mock fixtures must be present so Tier A `--mock` stays deterministic.
        assert "mock_tool" in task


def test_tool_smoke_mock_outputs_raw_metrics_no_ci(tmp_path):
    output_path = tmp_path / "tool-smoke.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnifusion.evals.tools",
            "--mock",
            "--output",
            str(output_path),
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": "src"},
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text())
    assert payload["suite"] == "tool-smoke"
    assert payload["driver"] == "mock-contract"
    assert len(payload["tasks"]) <= 20
    required = {"id", "passed", "cost_usd", "wall_time_s"}
    assert all(required <= set(task) for task in payload["tasks"])
    assert "pass_rate" in payload["raw"]
    # No confidence intervals at micro-bench N.
    assert "95% CI" not in json.dumps(payload)
    assert "confidence_interval" not in json.dumps(payload)
    assert "tool-smoke" in result.stdout
