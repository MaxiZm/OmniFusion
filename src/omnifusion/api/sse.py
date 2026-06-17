"""Helpers for OpenAI-compatible streaming (Server-Sent Events)."""
import json
import time
import uuid


def wants_usage(body) -> bool:
    """True when the request opted into a streaming usage chunk via stream_options."""
    so = getattr(body, "stream_options", None)
    return bool(so and getattr(so, "include_usage", False))


def usage_chunk_sse(model: str, prompt_tokens: int, completion_tokens: int) -> str:
    """An OpenAI-style terminal usage chunk: empty choices + a usage block.

    Emitted just before `data: [DONE]` when the client set
    stream_options.include_usage. Mirrors the shape OpenAI sends so SDKs can read
    token usage off a streamed response.
    """
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return f"data: {json.dumps(chunk)}\n\n"
