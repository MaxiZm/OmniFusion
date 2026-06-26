from omnifusion.api.schemas import ChatCompletionRequest


def test_openai_compat_request_fields_normalize_to_internal_shape():
    req = ChatCompletionRequest(
        model="fusion/general",
        messages=[
            {"role": "developer", "content": "answer tersely"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "world"},
                ],
            },
        ],
        max_completion_tokens=42,
        metadata={"ticket": "M2"},
        seed=7,
        presence_penalty=0.25,
        frequency_penalty=-0.5,
        parallel_tool_calls=False,
        service_tier="default",
    )

    assert [message.role for message in req.messages] == ["system", "user"]
    assert req.messages[1].content == "hello\nworld"
    assert req.max_tokens == 42
    assert req.metadata == {"ticket": "M2"}
    assert req.seed == 7
    assert req.presence_penalty == 0.25
    assert req.frequency_penalty == -0.5
    assert req.parallel_tool_calls is False
    assert req.service_tier == "default"


def test_legacy_functions_translate_to_tools_and_tool_choice():
    req = ChatCompletionRequest(
        model="fusion/general",
        messages=[{"role": "user", "content": "lookup weather"}],
        functions=[
            {
                "name": "get_weather",
                "description": "Read weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ],
        function_call={"name": "get_weather"},
    )

    assert req.tools[0].type == "function"
    assert req.tools[0].function.name == "get_weather"
    assert req.tool_choice.type == "function"
    assert req.tool_choice.function.name == "get_weather"


def test_text_content_parts_reject_non_text_parts():
    try:
        ChatCompletionRequest(
            model="fusion/general",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/a.png"},
                        }
                    ],
                }
            ],
        )
    except ValueError as exc:
        assert "Only text content parts are supported" in str(exc)
    else:
        raise AssertionError("image content part should have been rejected")
