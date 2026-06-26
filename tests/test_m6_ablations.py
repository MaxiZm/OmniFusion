import json


def valid_artifact():
    return {
        "strategy": "conductor",
        "component": "planner",
        "status": "candidate_default",
        "tier": "C",
        "date": "2026-06-26",
        "commit_sha": "abcdef1",
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
            },
            "judge_selected_best_of_n": {
                "cost_budget_equal": True,
                "solve_per_usd_delta_mean": 0.08,
                "solve_per_usd_delta_ci95": [0.02, 0.15],
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
