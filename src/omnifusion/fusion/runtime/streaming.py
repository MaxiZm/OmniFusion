"""The single canonical home for OpenAI-compatible SSE emission (M3b).

Every streaming path (classic fusion, tool-step, conductor, passthrough) shapes
its Server-Sent Events through StreamingAdapter so chunk framing, the terminal
usage chunk, the `[DONE]` sentinel, finish-reason normalization, and tool-call
deltas are defined exactly once.
"""
import json
import time
import uuid

DONE_SSE = "data: [DONE]\n\n"


def normalize_finish_reason(reason) -> str:
    """OpenAI requires a non-null finish_reason on the terminal chunk/choice."""
    return reason or "stop"


def usage_chunk(model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    """An OpenAI-style terminal usage chunk: empty choices + a usage block."""
    return {
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


class StreamingAdapter:
    def __init__(self, model: str):
        self.model = model

    def chunk_sse(self, chunk) -> str:
        """Frame a litellm/pydantic chunk object, forcing our model name."""
        data = json.loads(chunk.model_dump_json())
        data["model"] = self.model
        return f"data: {json.dumps(data)}\n\n"

    def raw_chunk_sse(self, chunk: dict) -> str:
        """Frame an already-OpenAI-shaped chunk dict, forcing our model name."""
        data = dict(chunk)
        data["model"] = self.model
        return f"data: {json.dumps(data)}\n\n"

    def usage_sse(self, prompt_tokens: int, completion_tokens: int) -> str:
        """The terminal usage chunk, emitted before [DONE] when requested."""
        return f"data: {json.dumps(usage_chunk(self.model, prompt_tokens, completion_tokens))}\n\n"

    def done_sse(self) -> str:
        return DONE_SSE

    def tool_call_sse(self, tool_calls):
        """Yield the SSE sequence for an assistant tool-call response."""
        cid = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())

        def chunk(delta, finish=None):
            return {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        yield f"data: {json.dumps(chunk({'role': 'assistant', 'content': None}))}\n\n"
        delta_tcs = [
            {
                "index": i,
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                },
            }
            for i, tc in enumerate(tool_calls)
        ]
        yield f"data: {json.dumps(chunk({'tool_calls': delta_tcs}))}\n\n"
        yield f"data: {json.dumps(chunk({}, finish='tool_calls'))}\n\n"
        yield self.done_sse()
