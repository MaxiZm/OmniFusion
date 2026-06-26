from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_BASELINES = ("best_single", "judge_selected_best_of_n")


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _ci95(comparison: dict[str, Any], field: str) -> list[Any]:
    value = comparison.get(field)
    return value if isinstance(value, list) else []


def validate_ablation_artifact(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if artifact.get("tier") != "C":
        errors.append("artifact tier must be C real-provider evidence")

    if artifact.get("ci_method") != "bootstrap":
        errors.append("ci_method must be bootstrap")

    for field in ("strategy", "component", "date", "commit_sha", "pricing"):
        if not artifact.get(field):
            errors.append(f"missing required field: {field}")

    if not isinstance(artifact.get("failure_modes"), dict) or not artifact["failure_modes"]:
        errors.append("artifact must include failure_modes breakdown")

    runs = artifact.get("runs")
    if not isinstance(runs, list) or len(runs) < 3:
        errors.append("artifact must include at least 3 real-provider runs")

    comparisons = artifact.get("comparisons")
    if not isinstance(comparisons, dict):
        errors.append("artifact must include comparisons")
        comparisons = {}

    for baseline in REQUIRED_BASELINES:
        comparison = comparisons.get(baseline)
        if not isinstance(comparison, dict):
            errors.append(f"missing comparison against {baseline}")
            continue
        if comparison.get("cost_budget_equal") is not True:
            errors.append(f"{baseline} comparison must use an equal cost/token budget")
        for metric in ("solve_per_usd", "solve_per_wall_s"):
            mean_field = f"{metric}_delta_mean"
            ci95_field = f"{metric}_delta_ci95"
            if not _is_number(comparison.get(mean_field)):
                errors.append(f"{baseline} comparison missing numeric {mean_field}")
            ci95 = _ci95(comparison, ci95_field)
            if len(ci95) != 2 or not all(_is_number(value) for value in ci95):
                errors.append(f"{baseline} comparison missing numeric {ci95_field}")

    if artifact.get("claims"):
        errors.append("ablation artifacts must not contain marketing claims")

    return errors


def can_enable_default(artifact: dict[str, Any]) -> bool:
    if validate_ablation_artifact(artifact):
        return False

    comparisons = artifact["comparisons"]
    for baseline in REQUIRED_BASELINES:
        solve_per_usd_low = _ci95(comparisons[baseline], "solve_per_usd_delta_ci95")[0]
        solve_per_wall_low = _ci95(comparisons[baseline], "solve_per_wall_s_delta_ci95")[0]
        if solve_per_usd_low <= 0 or solve_per_wall_low <= 0:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate OmniFusion ablation artifacts.")
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)

    artifact = json.loads(args.artifact.read_text())
    errors = validate_ablation_artifact(artifact)
    if errors:
        for error in errors:
            print(f"ablation artifact invalid: {error}")
        return 1
    print(
        "ablation artifact valid; "
        f"default_enabled={str(can_enable_default(artifact)).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
