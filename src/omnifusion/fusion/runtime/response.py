import time
import uuid


class ResponseShaper:
    @staticmethod
    def chat_completion(
        *,
        model: str,
        content: str | None,
        usage: dict,
        finish_reason: str = "stop",
        created: int | None = None,
        response_id: str | None = None,
    ) -> dict:
        return {
            "id": response_id or f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": created or int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason or "stop",
                }
            ],
            "usage": usage,
        }
