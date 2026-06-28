from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
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


class SelfFusionConfig(BaseModel):
    n: int = Field(default=3, ge=1, le=16)
    temperature_spread: List[float] = Field(default_factory=lambda: [0.3, 0.7, 1.0])
    seed_offset: bool = True


class DebateConfig(BaseModel):
    rounds: int = Field(default=1, ge=1, le=3)


class RouteModelConfig(BaseModel):
    model: str
    tier: Literal["fast", "balanced", "strong"] = "balanced"
    provider_id: Optional[str] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("route model must not be empty")
        return value

    @field_validator("provider_id")
    @classmethod
    def validate_provider_id(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("route provider_id must not be empty")
        return value


class RouterConfig(BaseModel):
    enabled: bool = False
    mode: Literal["heuristic", "model", "always", "never"] = "heuristic"
    min_chars: int = Field(default=280, ge=0)
    fuse_keywords: List[str] = Field(
        default_factory=lambda: [
            "compare",
            "trade-off",
            "tradeoff",
            "analyze",
            "analyse",
            "evaluate",
            "design",
            "research",
            "pros and cons",
            "why",
            "explain",
            "critique",
            "recommend",
            "debug",
            "architecture",
        ]
    )
    classifier_model: Optional[str] = None
    classifier_provider_id: Optional[str] = None
    classifier_max_tokens: int = Field(default=4, ge=1, le=64)
    route_models: List[RouteModelConfig] = Field(default_factory=list)
    fuse_only_with_tools: bool = False

    @field_validator("classifier_model", "classifier_provider_id")
    @classmethod
    def validate_optional_string(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("router classifier fields must not be empty")
        return value


class AnalysisEmitConfig(BaseModel):
    enabled: bool = False


class ResponseCacheConfig(BaseModel):
    enabled: bool = False
    ttl_seconds: int = Field(default=300, ge=1)
    max_entries: int = Field(default=512, ge=1)


class PresetV2(BaseModel):
    name: str
    display_name: Optional[str] = None
    version: Literal[2] = 2
    models: List[PresetModel] = Field(default_factory=list)
    prompts: PresetPrompts = Field(default_factory=PresetPrompts)
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
    fusion_mode: Literal["panel", "self_fusion", "debate"] = "panel"
    aggregator: Literal["judge", "vote", "ranked"] = "judge"
    self_fusion: SelfFusionConfig = Field(default_factory=SelfFusionConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    analysis_emit: AnalysisEmitConfig = Field(default_factory=AnalysisEmitConfig)
    response_cache: ResponseCacheConfig = Field(default_factory=ResponseCacheConfig)
    # Server-side web grounding for the panel ("web on"). Opt-in per preset; a
    # request's `plugins.web` overrides this for a single request (M5).
    web_enabled: bool = False

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_or_fill_v2(cls, data):
        if not isinstance(data, dict):
            return data

        data = dict(data)
        data["version"] = 2
        data.setdefault("display_name", data.get("name"))
        data.setdefault("prompts", {})
        data.setdefault("bandit", {})

        # `budgets` is the v2 grouped shape; consume it to populate the flat stage
        # fields the runtime reads, then drop it (it is re-exposed as a computed
        # field so it round-trips without being a redundant stored field).
        # Guard against a corrupt/hand-edited row where `budgets` is present but
        # not an object: a non-dict value here would raise a raw AttributeError
        # (which pydantic does not wrap into a ValidationError), bypassing the
        # store's resilient loaders. Ignore it and fall back to the flat fields.
        budgets = data.pop("budgets", None)
        if isinstance(budgets, dict) and budgets:
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def budgets(self) -> Optional[PresetBudgets]:
        """The v2 grouped budget view, derived from the flat stage fields so it
        round-trips in serialization without being a redundant stored field."""
        if self.panel and self.judge and self.final:
            return PresetBudgets(
                panel=self.panel,
                judge=self.judge,
                final=self.final,
                cost_ceiling=self.cost_ceiling,
                min_panel_success=self.min_panel_success,
            )
        return None

    def provider_id_for(self, model: str, role: Optional[str] = None) -> str:
        """Resolve the configured provider for a pool model. Honors the models-pool
        provider_id (the roadmap-mandated provider_id/role/weight tuple) instead of
        always assuming 'default'."""
        for entry in self.models:
            if entry.model != model:
                continue
            if role is not None and entry.role != role:
                continue
            return entry.provider_id
        return "default"


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


class StageEvent(BaseModel):
    """One bounded record of a single stage call inside a fusion run.

    Additive and optional: old stored traces simply have an empty `stage_events`
    list. Carries no prompt/response bodies — only the per-stage accounting an
    operator needs to read a run end to end (who ran, status, tokens, cost,
    timing, and a short error code on failure).
    """

    stage: str  # web | panel | judge | synthesis | completion
    role: Optional[str] = None  # panel | judge | final
    provider_id: Optional[str] = None
    model: Optional[str] = None
    status: str = "ok"  # ok | error | timeout | rate_limited | degraded | skipped
    tokens: Optional[Dict[str, int]] = None  # {"prompt": int, "completion": int}
    cost_usd: float = 0.0
    wall_ms: Optional[int] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FusionTrace(BaseModel):
    run_id: str
    preset: str
    cost_usd: float
    wall_ms: int
    degraded: bool = False
    panel_results: List[PanelResult]
    judge_analysis: Optional[JudgeAnalysis] = None
    final_answer: Optional[str] = None
    # Additive, bounded, per-stage timeline. Defaults empty so traces stored before
    # this field existed still validate.
    stage_events: List[StageEvent] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def trace_metadata_for_preset(preset: Preset) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"preset_version": getattr(preset, "version", 1)}
    metadata["openfusion"] = {
        "fusion_mode": getattr(preset, "fusion_mode", "panel"),
        "aggregator": getattr(preset, "aggregator", "judge"),
        "router_enabled": bool(getattr(getattr(preset, "router", None), "enabled", False)),
        "response_cache_enabled": bool(
            getattr(getattr(preset, "response_cache", None), "enabled", False)
        ),
    }
    prompts = getattr(preset, "prompts", None)
    if prompts and (prompts.global_prompt or prompts.role_prompts):
        metadata["role_prompts_redacted"] = True
    return metadata


def _read_tokens(usage: Any) -> Optional[Dict[str, int]]:
    """Defensively read prompt/completion tokens from a provider usage object or
    dict. Returns None if nothing usable is present. Never raises."""
    if usage is None:
        return None

    def _get(name: str) -> int:
        try:
            if isinstance(usage, dict):
                value = usage.get(name)
            else:
                value = getattr(usage, name, None)
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    prompt = _get("prompt_tokens")
    completion = _get("completion_tokens")
    if prompt == 0 and completion == 0:
        return None
    return {"prompt": prompt, "completion": completion}


def build_stage_events(
    preset: Preset,
    panel_results: List["PanelResult"],
    judge_analysis: Optional["JudgeAnalysis"],
    final_answer: Optional[str],
    synth_cost: float = 0.0,
    synth_usage: Any = None,
    web_sources: Optional[List[Any]] = None,
    degraded: bool = False,
) -> List["StageEvent"]:
    """Derive a bounded per-stage timeline from the data already collected on a
    fusion run. Centralized so every FusionTrace construction site stays in sync.

    Carries no prompt/response bodies — only accounting fields. Defensive: it never
    raises, so trace persistence can never be broken by a malformed stage record.
    """
    events: List[StageEvent] = []

    try:
        if web_sources:
            domains: List[str] = []
            for src in web_sources:
                url = ""
                if isinstance(src, dict):
                    url = src.get("url") or src.get("domain") or ""
                else:
                    url = getattr(src, "url", "") or getattr(src, "domain", "") or ""
                if url:
                    domains.append(str(url)[:120])
            events.append(
                StageEvent(
                    stage="web",
                    status="ok",
                    metadata={
                        "source_count": len(web_sources),
                        "sources": domains[:8],
                    },
                )
            )

        for result in panel_results or []:
            model = getattr(result, "model", None)
            events.append(
                StageEvent(
                    stage="panel",
                    role="panel",
                    provider_id=preset.provider_id_for(model, "panel")
                    if model
                    else None,
                    model=model,
                    status=getattr(result, "status", "ok") or "ok",
                    tokens=_read_tokens(getattr(result, "usage", None)),
                    cost_usd=float(getattr(result, "cost_usd", 0.0) or 0.0),
                    error_code=None
                    if getattr(result, "status", "ok") == "ok"
                    else getattr(result, "status", None),
                )
            )

        if judge_analysis is not None:
            judge_tokens = {
                "prompt": int(getattr(judge_analysis, "prompt_tokens", 0) or 0),
                "completion": int(getattr(judge_analysis, "completion_tokens", 0) or 0),
            }
            if judge_tokens["prompt"] == 0 and judge_tokens["completion"] == 0:
                judge_tokens = None
            events.append(
                StageEvent(
                    stage="judge",
                    role="judge",
                    provider_id=preset.provider_id_for(preset.judge_model, "judge")
                    if preset.judge_model
                    else None,
                    model=preset.judge_model or None,
                    status="degraded" if degraded else "ok",
                    tokens=judge_tokens,
                    cost_usd=float(getattr(judge_analysis, "cost_usd", 0.0) or 0.0),
                )
            )

        # Synthesis ran whenever there is a final answer (the best-panel fallback also
        # produces a final answer but no synthesis cost — both are captured honestly).
        if final_answer is not None or synth_cost:
            events.append(
                StageEvent(
                    stage="synthesis",
                    role="final",
                    provider_id=preset.provider_id_for(preset.final_model, "final")
                    if preset.final_model
                    else None,
                    model=preset.final_model or None,
                    status="ok" if final_answer is not None else "error",
                    tokens=_read_tokens(synth_usage),
                    cost_usd=float(synth_cost or 0.0),
                )
            )
    except Exception:
        # Never let timeline construction break trace persistence.
        return events

    return events


def role_prompt_content(preset: Preset, role: str) -> str:
    prompts = getattr(preset, "prompts", None)
    if not prompts:
        return ""
    parts = [
        prompts.global_prompt.strip(),
        prompts.role_prompts.get(role, "").strip(),
    ]
    return "\n\n".join(part for part in parts if part)
