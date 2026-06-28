from typing import List, Optional, Any, Union, Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from .normalize import normalize_content
from ..settings import settings


FUNCTION_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
FUNCTION_DESCRIPTION_MAX_CHARS = 65536


class OpenAIShape(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def __getitem__(self, key: str):
        return getattr(self, key)


class FunctionToolSpec(OpenAIShape):
    name: str = Field(min_length=1, max_length=128, pattern=FUNCTION_NAME_PATTERN)
    description: Optional[str] = Field(default=None, max_length=FUNCTION_DESCRIPTION_MAX_CHARS)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ToolDefinition(OpenAIShape):
    type: Literal["function", "openrouter:web_search", "openrouter:web_fetch"]
    function: Optional[FunctionToolSpec] = None
    parameters: Optional[dict[str, JsonValue]] = None

    @model_validator(mode="after")
    def validate_tool_shape(self):
        if self.type == "function" and self.function is None:
            raise ValueError("function tools must include a function definition.")
        if self.type != "function" and self.function is not None:
            raise ValueError("OpenRouter server tools must not include function.")
        return self


class ToolCallFunction(OpenAIShape):
    name: str = Field(min_length=1, max_length=128, pattern=FUNCTION_NAME_PATTERN)
    arguments: str


class ToolCall(OpenAIShape):
    id: str = Field(min_length=1, max_length=256)
    type: Literal["function"]
    function: ToolCallFunction


class ToolChoiceFunctionRef(OpenAIShape):
    name: str = Field(min_length=1, max_length=128, pattern=FUNCTION_NAME_PATTERN)


class FunctionToolChoice(OpenAIShape):
    type: Literal["function"]
    function: ToolChoiceFunctionRef


ToolChoice = Union[Literal["none", "auto", "required"], FunctionToolChoice]
LegacyFunctionCall = Union[Literal["none", "auto"], ToolChoiceFunctionRef]


class ChatMessage(BaseModel):
    # "tool" is required for the agentic loop (tool-result messages). Assistant
    # messages that only carry tool_calls have null content, so content is optional.
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: Optional[str] = None
    # Tool-calling passthrough fields (OpenAI shape) — forwarded to the model.
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    @field_validator("content", mode="before")
    def validate_content(cls, v):
        v = normalize_content(v)
        # Fix #12: Enforce per-message content size limit
        max_chars = settings.omnifusion_max_content_chars
        if v is not None and len(v) > max_chars:
            raise ValueError(
                f"Message content exceeds maximum allowed length of {max_chars} characters."
            )
        return v


class StreamOptions(BaseModel):
    include_usage: bool = False


class FusionPlugins(OpenAIShape):
    analysis_models: Optional[List[str]] = None
    synthesis_model: Optional[str] = None
    judge_model: Optional[str] = None
    web: Optional[bool] = None
    max_panel: Optional[int] = None
    fusion_mode: Optional[Literal["panel", "self_fusion", "debate"]] = None
    aggregator: Optional[Literal["judge", "vote", "ranked"]] = None
    analysis_emit: Optional[bool] = None
    routing: Optional[bool] = None

    @field_validator("analysis_models")
    @classmethod
    def validate_analysis_models(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        if not value:
            raise ValueError("plugins.analysis_models must contain at least one model.")
        if len(value) > settings.max_panel:
            raise ValueError(
                f"plugins.analysis_models must contain at most {settings.max_panel} models."
            )
        if any(not model.strip() for model in value):
            raise ValueError("plugins.analysis_models must not contain empty model names.")
        return value

    @field_validator("synthesis_model", "judge_model")
    @classmethod
    def validate_synthesis_model(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("plugin model fields must not be empty.")
        return value

    @field_validator("max_panel")
    @classmethod
    def validate_max_panel(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if value < 1:
            raise ValueError("plugins.max_panel must be >= 1.")
        if value > settings.max_panel:
            raise ValueError(f"plugins.max_panel must be <= {settings.max_panel}.")
        return value


class OpenFusionRouteModel(OpenAIShape):
    model: str
    tier: Literal["fast", "balanced", "strong"] = "balanced"
    provider_id: Optional[str] = None

    @field_validator("model", "provider_id")
    @classmethod
    def validate_optional_strings(cls, value: Optional[str], info) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError(f"openfusion.route_models.{info.field_name} must not be empty.")
        return value


class OpenFusionRouter(OpenAIShape):
    enabled: Optional[bool] = None
    mode: Optional[Literal["heuristic", "model", "always", "never"]] = None
    min_chars: Optional[int] = Field(default=None, ge=0)
    fuse_keywords: Optional[List[str]] = None
    classifier_model: Optional[str] = None
    classifier_provider_id: Optional[str] = None
    classifier_max_tokens: Optional[int] = Field(default=None, ge=1, le=64)
    route_models: Optional[List[OpenFusionRouteModel]] = None
    fuse_only_with_tools: Optional[bool] = None

    @field_validator("classifier_model", "classifier_provider_id")
    @classmethod
    def validate_classifier_fields(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("openfusion.router classifier fields must not be empty.")
        return value

    @field_validator("fuse_keywords")
    @classmethod
    def validate_fuse_keywords(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        if any(not keyword.strip() for keyword in value):
            raise ValueError("openfusion.router.fuse_keywords must not contain empty strings.")
        return value


class OpenFusionSelfFusion(OpenAIShape):
    n: Optional[int] = Field(default=None, ge=1, le=16)
    temperature_spread: Optional[List[float]] = None
    seed_offset: Optional[bool] = None


class OpenFusionDebate(OpenAIShape):
    rounds: Optional[int] = Field(default=None, ge=1, le=3)


class OpenFusionResponseCache(OpenAIShape):
    enabled: Optional[bool] = None
    ttl_seconds: Optional[int] = Field(default=None, ge=1)
    max_entries: Optional[int] = Field(default=None, ge=1)


class OpenFusionOverrides(OpenAIShape):
    preset: Optional[Literal["quality", "budget"]] = None
    panel_models: Optional[List[str]] = None
    judge_model: Optional[str] = None
    final_model: Optional[str] = None
    fusion_mode: Optional[Literal["panel", "self_fusion", "debate"]] = None
    aggregator: Optional[Literal["judge", "vote", "ranked"]] = None
    self_fusion: Optional[OpenFusionSelfFusion] = None
    debate: Optional[OpenFusionDebate] = None
    router: Optional[OpenFusionRouter] = None
    analysis_emit: Optional[bool] = None
    response_cache: Optional[OpenFusionResponseCache] = None
    web: Optional[bool] = None
    max_panel: Optional[int] = Field(default=None, ge=1)

    @field_validator("panel_models")
    @classmethod
    def validate_panel_models(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        if not value:
            raise ValueError("openfusion.panel_models must contain at least one model.")
        if len(value) > settings.max_panel:
            raise ValueError(
                f"openfusion.panel_models must contain at most {settings.max_panel} models."
            )
        if any(not model.strip() for model in value):
            raise ValueError("openfusion.panel_models must not contain empty model names.")
        return value

    @field_validator("judge_model", "final_model")
    @classmethod
    def validate_models(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("openfusion model fields must not be empty.")
        return value

    @field_validator("max_panel")
    @classmethod
    def validate_max_panel(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value > settings.max_panel:
            raise ValueError(f"openfusion.max_panel must be <= {settings.max_panel}.")
        return value


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    store: bool = False
    user: Optional[str] = None
    metadata: Optional[dict[str, JsonValue]] = None
    seed: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    parallel_tool_calls: Optional[bool] = None
    service_tier: Optional[str] = None
    plugins: Optional[FusionPlugins] = None
    openfusion: Optional[OpenFusionOverrides] = None

    # We must explicitly reject these with an error
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[ToolChoice] = None
    functions: Optional[List[FunctionToolSpec]] = None
    function_call: Optional[LegacyFunctionCall] = None
    audio: Optional[Any] = None
    n: Optional[int] = None
    logprobs: Optional[Any] = None
    response_format: Optional[Any] = None
    reasoning_effort: Optional[Any] = None

    # NOTE: `tools` and `tool_choice` are intentionally accepted — when present,
    # the request is routed to a single tool-capable model (see api/chat.py) so
    # agentic clients like OpenCode work. Legacy `functions`/`function_call`
    # are normalized to the equivalent tool shape in M2.
    @field_validator(
        "audio",
        "logprobs",
        "response_format",
        "reasoning_effort",
    )
    def reject_unsupported(cls, v, info):
        if v is not None:
            raise ValueError(
                f"Field '{info.field_name}' is not supported in OmniFusion API."
            )
        return v

    @field_validator("n")
    def reject_n_greater_than_1(cls, v):
        if v is not None and v > 1:
            raise ValueError("n > 1 is not supported in OmniFusion API.")
        return v

    @field_validator("max_tokens", "max_completion_tokens")
    def validate_max_tokens(cls, v):
        # Fix #12: Reject zero or negative max_tokens; enforce upper bound
        if v is not None:
            if v < 1:
                raise ValueError("max_tokens must be >= 1.")
            max_limit = settings.omnifusion_max_tokens_limit
            if v > max_limit:
                raise ValueError(
                    f"max_tokens {v} exceeds the server maximum of {max_limit}."
                )
        return v

    @field_validator("presence_penalty", "frequency_penalty")
    def validate_penalties(cls, v, info):
        if v is not None and not -2 <= v <= 2:
            raise ValueError(f"{info.field_name} must be between -2 and 2.")
        return v

    @model_validator(mode="after")
    def validate_and_normalize(self):
        # Fix #12: Enforce message list size limit
        max_msgs = settings.omnifusion_max_messages
        if len(self.messages) > max_msgs:
            raise ValueError(
                f"messages list length {len(self.messages)} exceeds maximum of {max_msgs}."
            )

        if (
            self.max_tokens is not None
            and self.max_completion_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            raise ValueError("max_tokens and max_completion_tokens must match if both are set.")
        if self.max_tokens is None and self.max_completion_tokens is not None:
            self.max_tokens = self.max_completion_tokens

        self.messages = [
            message.model_copy(update={"role": "system"})
            if message.role == "developer"
            else message
            for message in self.messages
        ]

        if self.functions and not self.tools:
            self.tools = [
                ToolDefinition(type="function", function=function)
                for function in self.functions
            ]

        if self.function_call is not None and self.tool_choice is None:
            if isinstance(self.function_call, str):
                self.tool_choice = self.function_call
            else:
                self.tool_choice = FunctionToolChoice(
                    type="function",
                    function=self.function_call,
                )
        return self
