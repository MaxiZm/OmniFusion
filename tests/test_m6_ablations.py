import json


def valid_artifact():
    return {
        "strategy": "conductor",
        "component": "planner",
        "status": "candidate_default",
        "tier": "C",
        "date": "2026-06-26",
        "ci_method": "bootstrap",
        "commit_sha": "abcdef1",
        "failure_modes": {"syntax_error": {"strategy": 0, "best_single": 1}},
        "pricing": {"currency": "USD", "source": "provider-published"},
        "runs": [
            {"id": "run-1", "raw_artifact": "evals/coding/runs/run-1.json"},
            {"id": "run-2", "raw_artifact": "evals/coding/runs/run-2.json"},
            {"id": "run-3", "raw_artifact": "evals/coding/runs/run-3.json"},
        ],
        "comparisons": {
            "best_single": {
                "cost_budget_equal": True,
                "solve_per_usd_delta_mean": 0.1,
                "solve_per_usd_delta_ci95": [0.01, 0.2],
                "solve_per_wall_s_delta_mean": 0.05,
                "solve_per_wall_s_delta_ci95": [0.01, 0.09],
            },
            "judge_selected_best_of_n": {
                "cost_budget_equal": True,
                "solve_per_usd_delta_mean": 0.08,
                "solve_per_usd_delta_ci95": [0.02, 0.15],
                "solve_per_wall_s_delta_mean": 0.04,
                "solve_per_wall_s_delta_ci95": [0.01, 0.08],
            },
        },
    }


def test_ablation_template_is_not_benchmark_evidence():
    from omnifusion.evals.ablations import can_enable_default, validate_ablation_artifact

    with open("evals/coding/ablations/ablation_template.json") as f:
        template = json.load(f)

    errors = validate_ablation_artifact(template)

    assert errors
    assert can_enable_default(template) is False


def test_ablation_artifact_requires_tier_c_three_runs_and_two_baselines():
    from omnifusion.evals.ablations import can_enable_default, validate_ablation_artifact

    artifact = valid_artifact()

    assert validate_ablation_artifact(artifact) == []
    assert can_enable_default(artifact) is True

    artifact["runs"] = artifact["runs"][:2]
    errors = validate_ablation_artifact(artifact)
    assert "at least 3 real-provider runs" in "\n".join(errors)

    artifact = valid_artifact()
    artifact["comparisons"]["best_single"]["solve_per_usd_delta_ci95"] = [-0.01, 0.2]
    assert can_enable_default(artifact) is False


def test_ablation_artifact_requires_wall_clock_and_failure_breakdown():
    from omnifusion.evals.ablations import validate_ablation_artifact

    artifact = valid_artifact()
    del artifact["failure_modes"]
    del artifact["comparisons"]["best_single"]["solve_per_wall_s_delta_ci95"]

    errors = "\n".join(validate_ablation_artifact(artifact))

    assert "failure_modes" in errors
    assert "solve_per_wall_s_delta_ci95" in errors


def test_ablation_artifact_requires_bootstrap_ci_method():
    from omnifusion.evals.ablations import validate_ablation_artifact

    artifact = valid_artifact()
    artifact["ci_method"] = "wilson"

    assert "ci_method must be bootstrap" in "\n".join(
        validate_ablation_artifact(artifact)
    )
