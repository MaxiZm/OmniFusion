def normalize_requested_model(model: str) -> str:
    """Normalize adapter-specific model aliases at the API boundary."""
    aider_openai_prefix = "openai/fusion/"
    if model.startswith(aider_openai_prefix):
        return f"fusion/{model[len(aider_openai_prefix):]}"
    return model
