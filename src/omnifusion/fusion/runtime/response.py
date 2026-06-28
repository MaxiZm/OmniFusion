import time
import uuid


def _read_usage(usage) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from a usage object/dict, default 0."""
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return (
            int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0),
        )
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


class ResponseShaper:
    @staticmethod
    def usage_block(prompt_tokens: int, completion_tokens: int) -> dict:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    @staticmethod
    def read_usage(usage) -> tuple[int, int]:
        return _read_usage(usage)

    @staticmethod
    def aggregate_usage(preset, panel_results, judge_analysis, final_result) -> dict:
        """The canonical panel+judge+final usage aggregation (absorbed from the
        orchestrator). When preset.usage_reporting == 'final', only the final call's
        usage is reported."""
        final_prompt, final_completion = _read_usage(getattr(final_result, "usage", None))

        if getattr(preset, "usage_reporting", "aggregate") == "final":
            prompt_tokens, completion_tokens = final_prompt, final_completion
        else:
            prompt_tokens, completion_tokens = final_prompt, final_completion
            for r in panel_results:
                p, c = _read_usage(getattr(r, "usage", None))
                prompt_tokens += p
                completion_tokens += c
            if judge_analysis is not None:
                prompt_tokens += int(getattr(judge_analysis, "prompt_tokens", 0) or 0)
                completion_tokens += int(getattr(judge_analysis, "completion_tokens", 0) or 0)

        return ResponseShaper.usage_block(prompt_tokens, completion_tokens)

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

    @staticmethod
    def tool_call_completion(*, model: str, tool_calls: list, usage: dict) -> dict:
        """The non-stream response for an assistant tool-call turn."""
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": usage,
        }
