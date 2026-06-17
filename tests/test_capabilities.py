from omnifusion.providers.capabilities import (
    get_provider_type_from_model,
    filter_params,
)


def test_get_provider_type_from_model():
    assert get_provider_type_from_model("openai/gpt-4o") == "openai"
    assert get_provider_type_from_model("gpt-4o") == "openai"
    assert get_provider_type_from_model("anthropic/claude-3") == "anthropic"
    assert get_provider_type_from_model("claude-3-5-sonnet") == "anthropic"
    assert get_provider_type_from_model("ollama/llama3") == "ollama"
    assert get_provider_type_from_model("groq/mixtral") == "groq"
    assert get_provider_type_from_model("openrouter/meta/llama-3") == "openrouter"
    assert get_provider_type_from_model("lmstudio/local-model") == "lmstudio"
    assert get_provider_type_from_model("some-unknown-model") == "custom"


def test_filter_params():
    openai_params = {"temperature": 0.7, "response_format": {"type": "json_object"}}
    filtered_openai = filter_params("openai", openai_params)
    assert "response_format" in filtered_openai
    assert filtered_openai["temperature"] == 0.7

    anthropic_params = {
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
        "stop": "\n",
    }
    filtered_anthropic = filter_params("anthropic", anthropic_params)
    assert "response_format" not in filtered_anthropic
    assert filtered_anthropic["temperature"] == 0.7
    assert filtered_anthropic["stop"] == "\n"
