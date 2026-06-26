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


class PresetModel(BaseModel):
    provider_id: str = "default"
    role: Literal["panel", "judge", "final"]
    model: str
    weight: float = Field(default=1.0, gt=0)

    @field_validator("provider_id", "model")
    @classmethod
    def validate_nonempty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class PresetPrompts(BaseModel):
    global_prompt: str = ""
    role_prompts: Dict[str, str] = Field(default_factory=dict)


class PresetBudgets(BaseModel):
    panel: PresetStage
    judge: PresetStage
    final: PresetStage
    cost_ceiling: Optional[float] = None
    min_panel_success: int = 1


class PresetBandit(BaseModel):
    enabled: bool = False
    exploration: float = Field(default=0.5, ge=0)


class PresetV2(BaseModel):
    name: str
    display_name: Optional[str] = None
    mode: Literal["fusion", "fugu_compat"] = "fusion"
    version: Literal[2] = 2
    models: List[PresetModel] = Field(default_factory=list)
    prompts: PresetPrompts = Field(default_factory=PresetPrompts)
    budgets: Optional[PresetBudgets] = None
    bandit: PresetBandit = Field(default_factory=PresetBandit)
    strategy: Literal["B", "conductor"] = "B"
    panel_models: List[str] = Field(default_factory=list)
    panel: Optional[PresetStage] = None
    judge_model: str = ""
    judge: Optional[PresetStage] = None
    final_model: str = ""
    final: Optional[PresetStage] = None
    usage_reporting: Literal["aggregate", "final"] = "aggregate"
    cost_ceiling: Optional[float] = None
    on_final_failure: Literal["error", "best_panel"] = "error"
    min_panel_success: int = 1
    compat_status: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_or_fill_v2(cls, data):
        if not isinstance(data, dict):
            return data

        data = dict(data)
        data["version"] = 2
        data.setdefault("display_name", data.get("name"))
        data.setdefault("mode", "fusion")
        data.setdefault("prompts", {})
        data.setdefault("bandit", {})

        budgets = data.get("budgets") or {}
        if budgets:
            data.setdefault("panel", budgets.get("panel"))
            data.setdefault("judge", budgets.get("judge"))
            data.setdefault("final", budgets.get("final"))
            data.setdefault("cost_ceiling", budgets.get("cost_ceiling"))
            data.setdefault("min_panel_success", budgets.get("min_panel_success", 1))

        models = data.get("models") or []
        if models:
            panel_models = [
                model.get("model")
                for model in models
                if isinstance(model, dict) and model.get("role") == "panel"
            ]
            judge_models = [
                model.get("model")
                for model in models
                if isinstance(model, dict) and model.get("role") == "judge"
            ]
            final_models = [
                model.get("model")
                for model in models
                if isinstance(model, dict) and model.get("role") == "final"
            ]
            data.setdefault("panel_models", panel_models)
            if judge_models:
                data.setdefault("judge_model", judge_models[0])
            if final_models:
                data.setdefault("final_model", final_models[0])

        if not data.get("models"):
            data["models"] = [
                *[
                    {
                        "provider_id": "default",
                        "role": "panel",
                        "model": model,
                        "weight": 1.0,
                    }
                    for model in data.get("panel_models", [])
                ],
                {
                    "provider_id": "default",
                    "role": "judge",
                    "model": data.get("judge_model", ""),
                    "weight": 1.0,
                },
                {
                    "provider_id": "default",
                    "role": "final",
                    "model": data.get("final_model", ""),
                    "weight": 1.0,
                },
            ]

        if not data.get("budgets"):
            data["budgets"] = {
                "panel": data.get("panel"),
                "judge": data.get("judge"),
                "final": data.get("final"),
                "cost_ceiling": data.get("cost_ceiling"),
                "min_panel_success": data.get("min_panel_success", 1),
            }

        return data

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


class Preset(PresetV2):
    pass


class PanelResult(BaseModel):
    model: str
    status: str  # ok, error, timeout, rate_limited
    content: Optional[str] = None
    cost_usd: float = 0.0
    usage: Optional[Any] = None


class JudgeAnalysis(BaseModel):
    consensus: str = ""
    disagreements: str = ""
    contradictions: str = ""
    partial_coverage: str = ""
    unique_insights: Dict[str, List[str]] = Field(default_factory=dict)
    blind_spots: str = ""
    model_strengths: Dict[str, str] = Field(default_factory=dict)
    synthesis_plan: str = ""
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
    metadata: Dict[str, Any] = Field(default_factory=dict)


def trace_metadata_for_preset(preset: Preset) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"preset_version": getattr(preset, "version", 1)}
    prompts = getattr(preset, "prompts", None)
    if prompts and (prompts.global_prompt or prompts.role_prompts):
        metadata["role_prompts_redacted"] = True
    if preset.compat_status:
        metadata["model_status"] = preset.compat_status
    return metadata


def role_prompt_content(preset: Preset, role: str) -> str:
    prompts = getattr(preset, "prompts", None)
    if not prompts:
        return ""
    parts = [
        prompts.global_prompt.strip(),
        prompts.role_prompts.get(role, "").strip(),
    ]
    return "\n\n".join(part for part in parts if part)
