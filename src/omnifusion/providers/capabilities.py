# Simple capability registry
# True means the provider supports it, False means it should be dropped
PROVIDER_CAPABILITIES = {
    "openai": {
        "temperature": True,
        "top_p": True,
        "stop": True,
        "response_format": True,
    },
    "anthropic": {
        "temperature": True,
        "top_p": True,
        "stop": True,
        "response_format": False,
    },
    "gemini": {
        "temperature": True,
        "top_p": True,
        "stop": True,
        "response_format": False,
    },
    "ollama": {
        "temperature": True,
        "top_p": True,
        "stop": True,
        "response_format": True,
    },
    "openrouter": {
        "temperature": True,
        "top_p": True,
        "stop": True,
        "response_format": True,
    },
    "groq": {
        "temperature": True,
        "top_p": True,
        "stop": True,
        "response_format": False,
    },
    "lmstudio": {
        "temperature": True,
        "top_p": True,
        "stop": True,
        "response_format": True,
    },
}


def get_provider_type_from_model(model: str) -> str:
    m = model.lower()
    if m.startswith("openai/") or m.startswith("gpt-"):
        return "openai"
    if m.startswith("anthropic/") or m.startswith("claude-"):
        return "anthropic"
    if m.startswith("gemini/"):
        return "gemini"
    if m.startswith("ollama/"):
        return "ollama"
    if m.startswith("groq/"):
        return "groq"
    if m.startswith("openrouter/"):
        return "openrouter"
    if m.startswith("lm_studio/") or m.startswith("lmstudio/"):
        return "lmstudio"
    return "custom"


def filter_params(provider_type: str, params: dict) -> dict:
    """Filters optional params based on provider capabilities."""
    if provider_type not in PROVIDER_CAPABILITIES:
        return params  # default allow all

    caps = PROVIDER_CAPABILITIES[provider_type]
    filtered = {}
    for k, v in params.items():
        if caps.get(k, True):
            filtered[k] = v
    return filtered
