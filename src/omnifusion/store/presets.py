from typing import List, Optional
from .db import get_db_connection
from ..api.model_names import COMPAT_PLACEHOLDER_PRESET_NAMES, COMPAT_PLACEHOLDER_STATUS
from ..fusion.types import Preset, PresetPrompts, PresetStage
from ..settings import settings


async def get_preset(name: str) -> Optional[Preset]:
    async with get_db_connection() as db:
        cursor = await db.execute("SELECT spec_json FROM presets WHERE name=?", (name,))
        row = await cursor.fetchone()
        if not row:
            return None
        return Preset.model_validate_json(row[0])


async def list_presets() -> List[Preset]:
    presets = []
    async with get_db_connection() as db:
        cursor = await db.execute("SELECT spec_json FROM presets")
        async for row in cursor:
            presets.append(Preset.model_validate_json(row[0]))
    return presets


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

    existing = await get_preset(name)
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
