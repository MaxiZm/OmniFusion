from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_BASELINES = ("best_single", "judge_selected_best_of_n")


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _ci95(comparison: dict[str, Any]) -> list[Any]:
    value = comparison.get("solve_per_usd_delta_ci95")
    return value if isinstance(value, list) else []


def validate_ablation_artifact(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if artifact.get("tier") != "C":
        errors.append("artifact tier must be C real-provider evidence")

    for field in ("strategy", "component", "date", "commit_sha", "pricing"):
        if not artifact.get(field):
            errors.append(f"missing required field: {field}")

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
        if not _is_number(comparison.get("solve_per_usd_delta_mean")):
            errors.append(f"{baseline} comparison missing numeric solve_per_usd_delta_mean")
        ci95 = _ci95(comparison)
        if len(ci95) != 2 or not all(_is_number(value) for value in ci95):
            errors.append(f"{baseline} comparison missing numeric solve_per_usd_delta_ci95")

    if artifact.get("claims"):
        errors.append("ablation artifacts must not contain marketing claims")

    return errors


def can_enable_default(artifact: dict[str, Any]) -> bool:
    if validate_ablation_artifact(artifact):
        return False

    comparisons = artifact["comparisons"]
    for baseline in REQUIRED_BASELINES:
        ci95_low = _ci95(comparisons[baseline])[0]
        if ci95_low <= 0:
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
