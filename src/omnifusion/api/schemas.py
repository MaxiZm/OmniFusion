from typing import List, Optional, Any, Union, Literal
from pydantic import BaseModel, field_validator, model_validator
from ..settings import settings


class ChatMessage(BaseModel):
    # "tool" is required for the agentic loop (tool-result messages). Assistant
    # messages that only carry tool_calls have null content, so content is optional.
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    # Tool-calling passthrough fields (OpenAI shape) — forwarded to the model.
    tool_calls: Optional[Any] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    @field_validator("content", mode="before")
    def validate_content(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError(
                "content must be a string. Multimodal/array parts are not supported."
            )
        # Fix #12: Enforce per-message content size limit
        max_chars = settings.omnifusion_max_content_chars
        if len(v) > max_chars:
            raise ValueError(
                f"Message content exceeds maximum allowed length of {max_chars} characters."
            )
        return v


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    store: bool = False
    user: Optional[str] = None

    # We must explicitly reject these with an error
    tools: Optional[Any] = None
    tool_choice: Optional[Any] = None
    functions: Optional[Any] = None
    function_call: Optional[Any] = None
    audio: Optional[Any] = None
    n: Optional[int] = None
    logprobs: Optional[Any] = None
    response_format: Optional[Any] = None
    reasoning_effort: Optional[Any] = None

    # NOTE: `tools` and `tool_choice` are intentionally NOT rejected — when present,
    # the request is routed to a single tool-capable model (see api/chat.py) so
    # agentic clients like OpenCode work. The legacy `functions`/`function_call`
    # pair is still rejected (use `tools`).
    @field_validator(
        "functions",
        "function_call",
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

    @field_validator("max_tokens")
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

    @model_validator(mode="after")
    def validate_messages_count(self):
        # Fix #12: Enforce message list size limit
        max_msgs = settings.omnifusion_max_messages
        if len(self.messages) > max_msgs:
            raise ValueError(
                f"messages list length {len(self.messages)} exceeds maximum of {max_msgs}."
            )
        return self
