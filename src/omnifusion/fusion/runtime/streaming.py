import json

from ...api.sse import usage_chunk_sse


class StreamingAdapter:
    def __init__(self, model: str):
        self.model = model

    def chunk_sse(self, chunk) -> str:
        data = json.loads(chunk.model_dump_json())
        data["model"] = self.model
        return f"data: {json.dumps(data)}\n\n"

    def usage_sse(self, prompt_tokens: int, completion_tokens: int) -> str:
        return usage_chunk_sse(self.model, prompt_tokens, completion_tokens)

    def done_sse(self) -> str:
        return "data: [DONE]\n\n"
