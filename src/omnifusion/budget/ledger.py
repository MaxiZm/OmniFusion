import uuid
import datetime
import logging
from typing import Optional
from ..store.db import get_db_connection
from ..api.errors import BudgetExceededError
from ..settings import settings

logger = logging.getLogger("omnifusion.budget")

# Stale reservation age threshold in seconds (1 hour)
STALE_RESERVATION_AGE_SECONDS = 3600


async def initialize_request_budget(run_id: str, ceiling_micro_usd: Optional[int]):
    today_str = datetime.date.today().isoformat()
    async with get_db_connection() as db:
        await db.execute("BEGIN IMMEDIATE")
        try:
            # 1. Ensure global daily budget row exists
            global_ceiling = int(
                getattr(settings, "global_daily_budget_usd", 100.0) * 1_000_000
            )
            await db.execute(
                "INSERT OR IGNORE INTO budget_ledger (scope, window_key, reserved_micro_usd, spent_micro_usd, ceiling_micro_usd) VALUES ('global', ?, 0, 0, ?)",
                (today_str, global_ceiling),
            )

            # 2. Ensure request budget row exists
            if ceiling_micro_usd is None:
                ceiling_micro_usd = int(
                    getattr(settings, "request_budget_usd", 10.0) * 1_000_000
                )
            await db.execute(
                "INSERT OR IGNORE INTO budget_ledger (scope, window_key, reserved_micro_usd, spent_micro_usd, ceiling_micro_usd) VALUES ('request', ?, 0, 0, ?)",
                (run_id, ceiling_micro_usd),
            )
            await db.commit()
        except Exception:
            await db.execute("ROLLBACK")
            raise


async def reserve_budget(run_id: str, stage: str, reserve_micro_usd: int) -> str:
    """Reserves budget for a specific run and stage. Returns a reservation_id."""
    reservation_id = str(uuid.uuid4())
    today_str = datetime.date.today().isoformat()

    async with get_db_connection() as db:
        await db.execute("BEGIN IMMEDIATE")
        try:
            # Fix #4: Ensure the global row for TODAY always exists inside the transaction.
            # This handles the midnight case where the server started on a previous day.
            global_ceiling = int(
                getattr(settings, "global_daily_budget_usd", 100.0) * 1_000_000
            )
            await db.execute(
                "INSERT OR IGNORE INTO budget_ledger (scope, window_key, reserved_micro_usd, spent_micro_usd, ceiling_micro_usd) VALUES ('global', ?, 0, 0, ?)",
                (today_str, global_ceiling),
            )

            # Check global budget
            cursor = await db.execute(
                "SELECT ceiling_micro_usd, reserved_micro_usd, spent_micro_usd FROM budget_ledger WHERE scope='global' AND window_key=?",
                (today_str,),
            )
            row = await cursor.fetchone()
            # Row is now guaranteed to exist (created above with INSERT OR IGNORE)
            if row:
                ceiling, reserved, spent = row
                if ceiling is not None and (
                    reserved + spent + reserve_micro_usd > ceiling
                ):
                    raise BudgetExceededError(
                        f"Global daily budget exceeded (needs {reserve_micro_usd} microUSD, remaining {ceiling - (reserved + spent)} microUSD)"
                    )

            # Check request budget
            cursor = await db.execute(
                "SELECT ceiling_micro_usd, reserved_micro_usd, spent_micro_usd FROM budget_ledger WHERE scope='request' AND window_key=?",
                (run_id,),
            )
            row = await cursor.fetchone()
            if row:
                ceiling, reserved, spent = row
                if ceiling is not None and (
                    reserved + spent + reserve_micro_usd > ceiling
                ):
                    raise BudgetExceededError(
                        f"Request budget exceeded (needs {reserve_micro_usd} microUSD, remaining {ceiling - (reserved + spent)} microUSD)"
                    )

            # Update both global and request ledger
            await db.execute(
                "UPDATE budget_ledger SET reserved_micro_usd = reserved_micro_usd + ? WHERE scope='global' AND window_key=?",
                (reserve_micro_usd, today_str),
            )
            await db.execute(
                "UPDATE budget_ledger SET reserved_micro_usd = reserved_micro_usd + ? WHERE scope='request' AND window_key=?",
                (reserve_micro_usd, run_id),
            )

            # Insert reservation with created_at timestamp for age-based sweeping
            import time as _time
            await db.execute(
                "INSERT INTO budget_reservations (reservation_id, run_id, stage, reserved_micro_usd, state, global_window_key, created_at) VALUES (?, ?, ?, ?, 'reserved', ?, ?)",
                (reservation_id, run_id, stage, reserve_micro_usd, today_str, int(_time.time())),
            )

            await db.commit()
            return reservation_id

        except Exception:
            await db.execute("ROLLBACK")
            raise


async def reconcile_budget(reservation_id: str, actual_micro_usd: int):
    """Reconciles a completed call's actual cost against its reservation.

    Records the TRUE actual spend, even when it pushes the ledger past its ceiling.
    The provider has already billed us for the real amount, so clamping recorded
    spend down to the ceiling would under-report and let subsequent requests run
    after the budget is genuinely exhausted.

    Enforcement lives at reservation time: reserve_budget() rejects new work once
    reserved + spent + new > ceiling. So an over-ceiling `spent` here simply means
    every future reservation for this window is denied — which is the correct
    fail-closed behavior after an under-reservation.
    """
    today_str = datetime.date.today().isoformat()
    actual_micro_usd = max(0, actual_micro_usd)
    async with get_db_connection() as db:
        await db.execute("BEGIN IMMEDIATE")
        try:
            cursor = await db.execute(
                "SELECT run_id, reserved_micro_usd, state, global_window_key FROM budget_reservations WHERE reservation_id=?",
                (reservation_id,),
            )
            row = await cursor.fetchone()
            if not row:
                await db.execute("ROLLBACK")
                return  # Not found

            run_id, reserved, state, global_window_key = row
            if state != "reserved":
                await db.execute("ROLLBACK")
                return  # Already reconciled/cancelled

            # Observability: surface under-reservations so mispriced/aliased models
            # can be corrected, but still record the real spend below.
            if actual_micro_usd > reserved:
                logger.warning(
                    f"Reconcile overspend: reservation={reservation_id} reserved={reserved} "
                    f"actual={actual_micro_usd} (model under-reserved; recording true spend)"
                )

            # Mark reservation reconciled
            await db.execute(
                "UPDATE budget_reservations SET state='reconciled' WHERE reservation_id=?",
                (reservation_id,),
            )

            # Request ledger: release the reservation, record the true actual spend.
            await db.execute(
                "UPDATE budget_ledger SET reserved_micro_usd = reserved_micro_usd - ?, spent_micro_usd = spent_micro_usd + ? WHERE scope='request' AND window_key=?",
                (reserved, actual_micro_usd, run_id),
            )

            # Global daily ledger: same — true spend, no clamp.
            g_window = global_window_key if global_window_key else today_str
            await db.execute(
                "UPDATE budget_ledger SET reserved_micro_usd = reserved_micro_usd - ?, spent_micro_usd = spent_micro_usd + ? WHERE scope='global' AND window_key=?",
                (reserved, actual_micro_usd, g_window),
            )

            await db.commit()

        except Exception:
            await db.execute("ROLLBACK")
            raise


async def cancel_reservation(reservation_id: str):
    """Cancels a reservation by releasing it with zero actual cost."""
    await reconcile_budget(reservation_id, 0)


async def sweep_stale_reservations():
    """Sweeps reservations still in 'reserved' state older than STALE_RESERVATION_AGE_SECONDS.

    Fix A: Uses the created_at timestamp for purely age-based eviction.
    The previous implementation used run-table membership, which would
    incorrectly sweep live in-flight reservations for store:false runs
    (which are never written to the runs table until after reconcile).
    Age-based eviction is safe because any legitimate call either:
      (a) reconciles within its timeout (typically ≪ 1 hour), or
      (b) is genuinely orphaned and should be released.
    """
    import time
    cutoff = int(time.time()) - STALE_RESERVATION_AGE_SECONDS

    # Step 1: collect candidates — do not hold the connection while cancelling
    stale_ids = []
    try:
        async with get_db_connection() as db:
            cursor = await db.execute(
                """
                SELECT reservation_id
                FROM budget_reservations
                WHERE state = 'reserved'
                  AND created_at > 0          -- skip un-migrated rows with DEFAULT 0
                  AND created_at < ?           -- older than cutoff
                LIMIT 100
                """,
                (cutoff,),
            )
            stale_ids = [row[0] for row in await cursor.fetchall()]
    except Exception as e:
        logger.warning(f"sweep_stale_reservations: failed to query candidates: {e}")
        return

    # Step 2: cancel each one individually so a single failure doesn't abort the batch
    for rid in stale_ids:
        try:
            await cancel_reservation(rid)
            logger.info(f"Swept stale reservation {rid} (older than {STALE_RESERVATION_AGE_SECONDS}s)")
        except Exception as e:
            logger.warning(f"Failed to sweep stale reservation {rid}: {e}")

