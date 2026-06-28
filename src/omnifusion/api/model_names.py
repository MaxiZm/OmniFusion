from ..settings import settings


def normalize_requested_model(model: str) -> str:
    """Normalize adapter-specific model aliases at the API boundary."""
    aider_openai_prefix = "openai/fusion/"
    if model.startswith(aider_openai_prefix):
        return f"fusion/{model[len(aider_openai_prefix):]}"
    if model in {"openrouter/fusion", "openfusion"}:
        return f"fusion/{settings.omnifusion_default_fusion_preset}"
    return model


def is_fusion_model_reference(model: str) -> bool:
    normalized = model.strip().lower()
    if normalized in {"openrouter/fusion", "openrouter:fusion", "openfusion"}:
        return True
    return normalized.startswith(("fusion/", "openai/fusion/"))


def model_alias_entries(created: int) -> list[dict]:
    return [
        {
            "id": "openrouter/fusion",
            "object": "model",
            "created": created,
            "owned_by": "omnifusion",
            "alias_of": f"fusion/{settings.omnifusion_default_fusion_preset}",
        },
        {
            "id": "openfusion",
            "object": "model",
            "created": created,
            "owned_by": "omnifusion",
            "alias_of": f"fusion/{settings.omnifusion_default_fusion_preset}",
        },
    ]
