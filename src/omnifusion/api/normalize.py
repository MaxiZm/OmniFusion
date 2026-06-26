from __future__ import annotations

from typing import Any


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
