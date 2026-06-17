from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class PresetStage(BaseModel):
    max_tokens: int
    timeout: int


class Preset(BaseModel):
    name: str
    strategy: str = "B"
    panel_models: List[str]
    panel: PresetStage
    judge_model: str
    judge: PresetStage
    final_model: str
    final: PresetStage
    usage_reporting: str = "aggregate"
    cost_ceiling: Optional[float] = None
    on_final_failure: str = "error"  # "error" or "best_panel"
    min_panel_success: int = 1


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
