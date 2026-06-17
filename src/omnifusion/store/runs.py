import time
from typing import Optional
from .db import get_db_connection
from ..fusion.types import FusionTrace


async def save_trace(
    trace: FusionTrace, store_flag: bool, key_hash: str, retention_days: int = 30
):
    if not store_flag:
        return  # Honors store:false

    expires_at = int(time.time()) + (retention_days * 86400)

    async with get_db_connection() as db:
        await db.execute(
            """
            INSERT INTO runs (run_id, preset, created_by_key_hash, wall_ms, cost_usd, store_flag, expires_at, trace_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                trace.run_id,
                trace.preset,
                key_hash,
                trace.wall_ms,
                trace.cost_usd,
                store_flag,
                expires_at,
                trace.model_dump_json(),
            ),
        )
        await db.commit()


async def get_trace(
    run_id: str, key_hash: Optional[str] = None
) -> Optional[FusionTrace]:
    """
    If key_hash is provided, enforces owner scoping. Expired runs are treated as
    absent (defense-in-depth alongside the background purge).
    """
    now = int(time.time())
    async with get_db_connection() as db:
        if key_hash:
            cursor = await db.execute(
                "SELECT trace_json FROM runs WHERE run_id=? AND created_by_key_hash=? "
                "AND (expires_at IS NULL OR expires_at >= ?)",
                (run_id, key_hash, now),
            )
        else:
            cursor = await db.execute(
                "SELECT trace_json FROM runs WHERE run_id=? "
                "AND (expires_at IS NULL OR expires_at >= ?)",
                (run_id, now),
            )

        row = await cursor.fetchone()
        if not row:
            return None

        return FusionTrace.model_validate_json(row[0])


async def purge_expired_runs() -> int:
    """Delete runs whose retention window has elapsed. Returns rows removed."""
    now = int(time.time())
    async with get_db_connection() as db:
        cursor = await db.execute("DELETE FROM runs WHERE expires_at < ?", (now,))
        await db.commit()
        return cursor.rowcount or 0
