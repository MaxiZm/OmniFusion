import json
import os
import subprocess
import sys
from pathlib import Path


def test_eval_coding_full_emits_jsonl_and_markdown_reports(tmp_path):
    output_path = tmp_path / "full.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnifusion.evals.coding",
            "full",
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
    jsonl_path = output_path.with_suffix(".jsonl")
    markdown_path = output_path.with_suffix(".md")

    assert payload["suite"] == "coding-full"
    assert payload["reports"] == {
        "json": str(output_path),
        "jsonl": str(jsonl_path),
        "markdown": str(markdown_path),
    }
    assert {"usd_per_task", "solve_per_usd", "solve_per_wall_s"} <= set(
        payload["raw"]["cost_normalization"]
    )
    assert jsonl_path.exists()
    assert markdown_path.exists()

    jsonl_rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert len(jsonl_rows) == payload["raw"]["total"]
    assert jsonl_rows[0]["type"] == "task_result"
    assert {"id", "passed", "cost_usd", "wall_time_s"} <= set(jsonl_rows[0]["task"])

    markdown = markdown_path.read_text()
    assert "# OmniFusion Coding Full Report" in markdown
    assert "95% CI" in markdown
    assert "usd_per_task" in markdown
    assert "solve_per_usd" in markdown
    assert "solve_per_wall_s" in markdown
    assert "Mock outputs are not benchmark evidence." in markdown
