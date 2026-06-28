from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from omnifusion.fusion.types import PanelResult, Preset


@dataclass(frozen=True)
class ModelBanditStats:
    attempts: int
    successes: int
    cost_usd: float

    @property
    def reward_mean(self) -> float:
        if self.attempts <= 0:
            return 0.0
        avg_cost = max(self.cost_usd / self.attempts, 0.000001)
        return (self.successes / self.attempts) / avg_cost


def model_stats_from_panel_results(
    panel_results: list[PanelResult],
) -> dict[str, ModelBanditStats]:
    counters: dict[str, dict[str, float | int]] = {}
    for result in panel_results:
        item = counters.setdefault(
            result.model,
            {"attempts": 0, "successes": 0, "cost_usd": 0.0},
        )
        item["attempts"] = int(item["attempts"]) + 1
        item["successes"] = int(item["successes"]) + (1 if result.status == "ok" else 0)
        item["cost_usd"] = float(item["cost_usd"]) + float(result.cost_usd or 0.0)

    return {
        model: ModelBanditStats(
            attempts=int(values["attempts"]),
            successes=int(values["successes"]),
            cost_usd=float(values["cost_usd"]),
        )
        for model, values in counters.items()
    }


def _panel_candidates(preset: Preset) -> list[str]:
    models = [
        model.model
        for model in getattr(preset, "models", [])
        if getattr(model, "role", None) == "panel"
    ]
    return models or list(preset.panel_models)


def _weights(preset: Preset) -> dict[str, float]:
    return {
        model.model: float(model.weight)
        for model in getattr(preset, "models", [])
        if getattr(model, "role", None) == "panel"
    }


def select_panel_models(
    preset: Preset,
    *,
    stats: Mapping[str, ModelBanditStats] | None = None,
    max_count: int | None = None,
) -> list[str]:
    candidates = _panel_candidates(preset)
    limit = max_count if max_count is not None else len(candidates)
    bandit = getattr(preset, "bandit", None)
    if not bandit or not bandit.enabled:
        return list(preset.panel_models)[:limit]

    weights = _weights(preset)
    stats = stats or {}
    total_attempts = max(1, sum(item.attempts for item in stats.values()))
    original_index = {model: index for index, model in enumerate(candidates)}

    def score(model: str) -> tuple[float, int]:
        model_stats = stats.get(model)
        reward = model_stats.reward_mean if model_stats else 0.0
        attempts = model_stats.attempts if model_stats else 0
        exploration = float(bandit.exploration) * math.sqrt(
            math.log(total_attempts + 1) / max(1, attempts)
        )
        return (
            weights.get(model, 1.0) + reward + exploration,
            -original_index[model],
        )

    return sorted(candidates, key=score, reverse=True)[:limit]
