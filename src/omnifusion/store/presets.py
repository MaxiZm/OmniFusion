from typing import List, Optional
from .db import get_db_connection
from ..fusion.types import Preset


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
