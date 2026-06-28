from __future__ import annotations

from typing import Any

# Generation-affecting OpenAI params that the plan's M2 normalization names and that
# must actually reach the provider call (not be accepted then silently dropped).
# `parallel_tool_calls` only makes sense when tools are present.
_GENERATION_FIELDS = ("seed", "presence_penalty", "frequency_penalty", "service_tier")


def generation_passthrough_kwargs(body: Any, *, include_tool_params: bool = False) -> dict:
    """Collect the non-None generation params from a request so callers can forward
    them to the provider. A client that sets `seed` for reproducibility must have it
    take effect rather than getting silently non-deterministic output."""
    out: dict[str, Any] = {}
    for field in _GENERATION_FIELDS:
        value = getattr(body, field, None)
        if value is not None:
            out[field] = value
    if include_tool_params:
        parallel = getattr(body, "parallel_tool_calls", None)
        if parallel is not None:
            out["parallel_tool_calls"] = parallel
    return out


def normalize_content(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value

    if not isinstance(value, list):
        raise ValueError(
            "content must be a string or an array of text content parts."
        )

    parts = []
    for part in value:
        if not isinstance(part, dict):
            raise ValueError("content parts must be objects.")
        part_type = part.get("type")
        if part_type not in {"text", "input_text"}:
            raise ValueError("Only text content parts are supported.")
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("text content parts must include string text.")
        parts.append(text)
    return "\n".join(parts)
