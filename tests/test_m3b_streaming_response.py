import json


class FakeDelta:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.delta = FakeDelta(content)


class FakeChunk:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]

    def model_dump_json(self):
        return json.dumps(
            {
                "id": "chunk",
                "object": "chat.completion.chunk",
                "model": "raw-model",
                "choices": [{"index": 0, "delta": {"content": "hello"}}],
            }
        )


def test_streaming_adapter_overrides_model_and_done_bytes():
    from omnifusion.fusion.runtime.streaming import StreamingAdapter

    adapter = StreamingAdapter("fusion/general")

    assert adapter.chunk_sse(FakeChunk("hello")) == (
        'data: {"id": "chunk", "object": "chat.completion.chunk", '
        '"model": "fusion/general", "choices": [{"index": 0, '
        '"delta": {"content": "hello"}}]}\n\n'
    )
    assert adapter.done_sse() == "data: [DONE]\n\n"


def test_response_shaper_chat_completion_shape():
    from omnifusion.fusion.runtime.response import ResponseShaper

    payload = ResponseShaper.chat_completion(
        model="fusion/general",
        content="answer",
        usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        finish_reason="stop",
        created=123,
        response_id="chatcmpl-fixed",
    )

    assert payload == {
        "id": "chatcmpl-fixed",
        "object": "chat.completion",
        "created": 123,
        "model": "fusion/general",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }
