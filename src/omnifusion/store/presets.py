import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from pydantic import ValidationError

from .db import get_db_connection
from ..api.model_names import COMPAT_PLACEHOLDER_PRESET_NAMES, COMPAT_PLACEHOLDER_STATUS
from ..fusion.types import Preset, PresetPrompts, PresetStage
from ..settings import settings

logger = logging.getLogger("omnifusion.presets")


@dataclass
class InvalidPreset:
    """A stored preset row that no longer validates against the current ``Preset``
    model — typically because an operator tightened an operational limit
    (max_tokens, stage timeout, cost ceiling, panel size) after the preset was
    saved. We surface these instead of letting a single bad row raise and take
    down every listing path (admin UI, ``/v1/models``, ``/v1/presets``, CLI)."""

    name: str
    error: str


def _summarize_load_error(exc: Exception) -> str:
    """Render a preset-load failure as a compact, operator-readable string.

    Most failures are pydantic ValidationErrors (e.g. a stored value now exceeds
    a tightened limit). But a corrupt or hand-edited row can also fail in the
    model's ``mode="before"`` validator with a non-ValidationError (e.g. an
    ``AttributeError`` from a non-dict ``budgets``, or a ``TypeError`` from a
    non-iterable ``models``), so we degrade gracefully for any exception type."""
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            msg = (err.get("msg") or "").removeprefix("Value error, ")
            parts.append(f"{loc}: {msg}" if loc else msg)
        return "; ".join(p for p in parts if p) or "invalid preset spec"
    return f"{type(exc).__name__}: {exc}".strip() or "invalid preset spec"


async def _fetch_spec_json(name: str) -> Optional[str]:
    """Return the raw stored spec_json for a preset, or None if it doesn't exist.
    Kept separate from parsing so callers can decide how to treat an un-loadable
    row (raise vs. regenerate vs. isolate)."""
    async with get_db_connection() as db:
        cursor = await db.execute("SELECT spec_json FROM presets WHERE name=?", (name,))
        row = await cursor.fetchone()
        return row[0] if row else None


async def get_preset(name: str) -> Optional[Preset]:
    spec_json = await _fetch_spec_json(name)
    if spec_json is None:
        return None
    try:
        return Preset.model_validate_json(spec_json)
    except Exception as exc:  # noqa: BLE001 - any un-loadable row maps to a 422
        # The row exists but cannot be loaded — usually a stored value that now
        # violates a tightened limit, occasionally a corrupt row. Returning None
        # would masquerade as "not found" (404); raise a clear 422 instead so
        # callers (run path, API) explain why and point at the fix. We catch
        # broadly because Preset's mode="before" validator can raise non-
        # ValidationError errors that pydantic does not wrap.
        from ..api.errors import OmniFusionError

        raise OmniFusionError(
            f"Preset '{name}' is stored in a state that cannot be loaded "
            f"({_summarize_load_error(exc)}). Update or delete it in the admin console.",
            status_code=422,
            code="preset_invalid",
        ) from exc


async def _partition_presets() -> Tuple[List[Preset], List[InvalidPreset]]:
    """Load every stored preset, separating rows that load from rows that don't.
    A single un-loadable row is logged and isolated rather than propagated, so
    listing endpoints degrade gracefully."""
    valid: List[Preset] = []
    invalid: List[InvalidPreset] = []
    async with get_db_connection() as db:
        cursor = await db.execute("SELECT name, spec_json FROM presets")
        async for name, spec_json in cursor:
            try:
                valid.append(Preset.model_validate_json(spec_json))
            except Exception as exc:  # noqa: BLE001 - one bad row must not 500 the list
                summary = _summarize_load_error(exc)
                logger.warning(
                    "Skipping preset %r: stored spec could not be loaded and was "
                    "excluded from listings. It likely predates a tightened limit; "
                    "update or delete it. Error: %s",
                    name,
                    summary,
                )
                invalid.append(InvalidPreset(name=name, error=summary))
    return valid, invalid


async def list_presets() -> List[Preset]:
    valid, _invalid = await _partition_presets()
    return valid


async def list_presets_with_invalid() -> Tuple[List[Preset], List[InvalidPreset]]:
    """Like :func:`list_presets`, but also returns the rows that could not be
    loaded so the admin UI can display and offer to delete them."""
    return await _partition_presets()


async def save_preset(preset: Preset):
    async with get_db_connection() as db:
        await db.execute(
            """
            INSERT INTO presets (name, strategy, spec_json)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                strategy=excluded.strategy,
                spec_json=excluded.spec_json
        """,
            (preset.name, preset.strategy, preset.model_dump_json()),
        )
        await db.commit()


async def delete_preset(name: str):
    async with get_db_connection() as db:
        await db.execute("DELETE FROM presets WHERE name=?", (name,))
        await db.commit()


def compat_placeholder_preset(name: str) -> Preset:
    max_tokens = min(1024, settings.omnifusion_max_tokens_limit)
    timeout = min(30, settings.omnifusion_max_stage_timeout)
    stage = PresetStage(max_tokens=max_tokens, timeout=timeout)
    model = settings.omnifusion_compat_placeholder_model
    display_names = {"fugu": "Fugu", "fugu-ultra": "Fugu Ultra"}
    return Preset(
        name=name,
        display_name=display_names.get(name, name),
        mode="fugu_compat",
        version=2,
        strategy="B",
        panel_models=[model],
        panel=stage,
        judge_model=model,
        judge=stage,
        final_model=model,
        final=stage,
        cost_ceiling=min(settings.request_budget_usd, settings.global_daily_budget_usd),
        min_panel_success=1,
        compat_status=COMPAT_PLACEHOLDER_STATUS,
        prompts=PresetPrompts(
            role_prompts={
                "panel": "Transparent Fugu-compatibility placeholder. This is not conductor-backed yet.",
                "judge": "Transparent Fugu-compatibility placeholder. This is not conductor-backed yet.",
                "final": "Transparent Fugu-compatibility placeholder. This is not conductor-backed yet.",
            }
        ),
    )


async def get_or_create_compat_placeholder_preset(name: str) -> Optional[Preset]:
    if name not in COMPAT_PLACEHOLDER_PRESET_NAMES:
        return None

    # Compat placeholders are app-managed and safe to regenerate. If a stored row
    # predates tightened limits or was hand-edited into an un-loadable state,
    # regenerate it instead of taking down startup or /v1/models.
    existing: Optional[Preset] = None
    spec_json = await _fetch_spec_json(name)
    if spec_json is not None:
        try:
            existing = Preset.model_validate_json(spec_json)
        except Exception as exc:  # noqa: BLE001 - regenerate any un-loadable row
            logger.warning(
                "Compat preset %r is stored in an un-loadable state (%s); "
                "regenerating from the current placeholder definition.",
                name,
                _summarize_load_error(exc),
            )

    if existing:
        if (
            existing.compat_status != COMPAT_PLACEHOLDER_STATUS
            or existing.version != 2
            or existing.mode != "fugu_compat"
        ):
            replacement = compat_placeholder_preset(name)
            existing.compat_status = replacement.compat_status
            existing.display_name = replacement.display_name
            existing.mode = replacement.mode
            existing.prompts = replacement.prompts
            await save_preset(existing)
        return existing

    preset = compat_placeholder_preset(name)
    await save_preset(preset)
    return preset


async def ensure_compat_placeholder_presets() -> None:
    for name in sorted(COMPAT_PLACEHOLDER_PRESET_NAMES):
        await get_or_create_compat_placeholder_preset(name)
