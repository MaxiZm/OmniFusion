"""Read-only budget-ledger summary for the admin UI and its JSON route.

Surfaces the reserve-and-reconcile state the orchestrator maintains: the global
daily window, recent per-request windows, and recent reservations. Values are
integer microdollars in the DB; they are converted to USD here for display.
Contains no secret material.
"""

import datetime
import time
from typing import Any, Dict, List

from ..settings import settings
from ..store.db import get_db_connection

_MICRO = 1_000_000


def _usd(micro: Any) -> float:
    return round((micro or 0) / _MICRO, 6)


async def collect_budget_summary(
    request_limit: int = 20, reservation_limit: int = 20
) -> Dict[str, Any]:
    today_str = datetime.date.today().isoformat()
    summary: Dict[str, Any] = {
        "today": today_str,
        "global_daily": None,
        "requests": [],
        "reservations": [],
    }

    async with get_db_connection() as db:
        # Global daily window for today.
        cursor = await db.execute(
            "SELECT reserved_micro_usd, spent_micro_usd, ceiling_micro_usd "
            "FROM budget_ledger WHERE scope='global' AND window_key=?",
            (today_str,),
        )
        row = await cursor.fetchone()
        if row:
            reserved, spent, ceiling = row
        else:
            reserved, spent, ceiling = 0, 0, int(settings.global_daily_budget_usd * _MICRO)
        remaining = (ceiling or 0) - (reserved or 0) - (spent or 0)
        summary["global_daily"] = {
            "reserved_usd": _usd(reserved),
            "spent_usd": _usd(spent),
            "ceiling_usd": _usd(ceiling),
            "remaining_usd": _usd(remaining),
        }

        # Recent per-request windows.
        cursor = await db.execute(
            "SELECT window_key, reserved_micro_usd, spent_micro_usd, ceiling_micro_usd "
            "FROM budget_ledger WHERE scope='request' ORDER BY rowid DESC LIMIT ?",
            (request_limit,),
        )
        requests: List[Dict[str, Any]] = []
        async for r in cursor:
            window_key, r_reserved, r_spent, r_ceiling = r
            requests.append(
                {
                    "run_id": window_key,
                    "reserved_usd": _usd(r_reserved),
                    "spent_usd": _usd(r_spent),
                    "ceiling_usd": _usd(r_ceiling),
                }
            )
        summary["requests"] = requests

        # Recent reservations (newest first by created_at, then rowid as a tiebreak).
        cursor = await db.execute(
            "SELECT reservation_id, run_id, stage, reserved_micro_usd, state, created_at "
            "FROM budget_reservations ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (reservation_limit,),
        )
        reservations: List[Dict[str, Any]] = []
        async for r in cursor:
            reservation_id, run_id, stage, reserved_micro, state, created_at = r
            created_str = (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
                if created_at
                else None
            )
            reservations.append(
                {
                    "reservation_id": reservation_id,
                    "run_id": run_id,
                    "stage": stage,
                    "reserved_usd": _usd(reserved_micro),
                    "state": state,
                    "created_at": created_str,
                }
            )
        summary["reservations"] = reservations

    # Reconciliation health: count reservations still open.
    summary["open_reservations"] = sum(
        1 for r in summary["reservations"] if r["state"] == "reserved"
    )
    return summary
