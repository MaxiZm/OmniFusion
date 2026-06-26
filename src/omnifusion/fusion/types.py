from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Any, Dict, Literal
from ..settings import settings


class PresetStage(BaseModel):
    max_tokens: int
    timeout: int

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int) -> int:
        if value < 1:
            raise ValueError("stage max_tokens must be >= 1")
        max_limit = settings.omnifusion_max_tokens_limit
        if value > max_limit:
            raise ValueError(f"stage max_tokens must be <= {max_limit}")
        return value

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, value: int) -> int:
        if value < 1:
            raise ValueError("stage timeout must be >= 1")
        max_limit = settings.omnifusion_max_stage_timeout
        if value > max_limit:
            raise ValueError(f"stage timeout must be <= {max_limit}")
        return value


class Preset(BaseModel):
    name: str
    strategy: Literal["B"] = "B"
    panel_models: List[str]
    panel: PresetStage
    judge_model: str
    judge: PresetStage
    final_model: str
    final: PresetStage
    usage_reporting: Literal["aggregate", "final"] = "aggregate"
    cost_ceiling: Optional[float] = None
    on_final_failure: Literal["error", "best_panel"] = "error"
    min_panel_success: int = 1

    @field_validator("name", "judge_model", "final_model")
    @classmethod
    def validate_nonempty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("panel_models")
    @classmethod
    def validate_panel_models(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("panel_models must contain at least one model")
        max_panel = settings.max_panel
        if len(value) > max_panel:
            raise ValueError(f"panel_models must contain at most {max_panel} models")
        if any(not model.strip() for model in value):
            raise ValueError("panel_models must not contain empty model names")
        return value

    @field_validator("min_panel_success")
    @classmethod
    def validate_min_panel_success_lower_bound(cls, value: int) -> int:
        if value < 1:
            raise ValueError("min_panel_success must be >= 1")
        return value

    @field_validator("cost_ceiling")
    @classmethod
    def validate_cost_ceiling(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("cost_ceiling must be > 0")
        max_budget = settings.global_daily_budget_usd
        if value > max_budget:
            raise ValueError(f"cost_ceiling must be <= {max_budget}")
        return value

    @model_validator(mode="after")
    def validate_cross_field_bounds(self):
        if self.min_panel_success > len(self.panel_models):
            raise ValueError("min_panel_success cannot exceed panel model count")
        return self


class PanelResult(BaseModel):
    model: str
    status: str  # ok, error, timeout, rate_limited
    content: Optional[str] = None
    cost_usd: float = 0.0
    usage: Optional[Any] = None


class JudgeAnalysis(BaseModel):
    consensus: str = ""
    disagreements: str = ""
    strongest_points_by_model: Dict[str, str] = Field(default_factory=dict)
    missing_information: str = ""
    likely_errors: str = ""
    recommended_final_answer_plan: str = ""
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class FusionTrace(BaseModel):
    run_id: str
    preset: str
    cost_usd: float
    wall_ms: int
    degraded: bool = False
    panel_results: List[PanelResult]
    judge_analysis: Optional[JudgeAnalysis] = None
    final_answer: Optional[str] = None
