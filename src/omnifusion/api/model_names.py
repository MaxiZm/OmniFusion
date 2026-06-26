from ..settings import settings


COMPAT_PLACEHOLDER_STATUS = "compat_placeholder - not conductor-backed yet"
COMPAT_PLACEHOLDER_PRESET_NAMES = {"fugu", "fugu-ultra"}
COMPAT_PLACEHOLDER_ALIASES = {
    "fugu": "fusion/fugu",
    "fugu-ultra": "fusion/fugu-ultra",
}


def normalize_requested_model(model: str) -> str:
    """Normalize adapter-specific model aliases at the API boundary."""
    aider_openai_prefix = "openai/fusion/"
    if model.startswith(aider_openai_prefix):
        return f"fusion/{model[len(aider_openai_prefix):]}"
    if model == "openrouter/fusion":
        return f"fusion/{settings.omnifusion_default_fusion_preset}"
    if model in COMPAT_PLACEHOLDER_ALIASES:
        return COMPAT_PLACEHOLDER_ALIASES[model]
    return model


def is_fusion_model_reference(model: str) -> bool:
    normalized = model.strip().lower()
    if normalized in {"openrouter/fusion", "openrouter:fusion"}:
        return True
    if normalized in COMPAT_PLACEHOLDER_ALIASES:
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
            "id": "fugu",
            "object": "model",
            "created": created,
            "owned_by": "omnifusion",
            "alias_of": "fusion/fugu",
            "status": COMPAT_PLACEHOLDER_STATUS,
        },
        {
            "id": "fugu-ultra",
            "object": "model",
            "created": created,
            "owned_by": "omnifusion",
            "alias_of": "fusion/fugu-ultra",
            "status": COMPAT_PLACEHOLDER_STATUS,
        },
    ]
