"""Admin budget summary: ledger state, reservation reconciliation, no secret leakage."""

import json

import pytest

from omnifusion.admin.budget_view import collect_budget_summary
from omnifusion.budget.ledger import (
    cancel_reservation,
    initialize_request_budget,
    reconcile_budget,
    reserve_budget,
)
from omnifusion.settings import settings
from omnifusion.store.db import init_db


@pytest.fixture
def budget_db(tmp_path):
    old_db = settings.db_path
    settings.db_path = str(tmp_path / "admin-budget.db")
    try:
        yield
    finally:
        settings.db_path = old_db


def _assert_no_secrets(summary):
    blob = json.dumps(summary)
    assert "test-password-123" not in blob
    assert settings.omnifusion_secret_key.get_secret_value() not in blob


@pytest.mark.asyncio
async def test_reservation_reconciliation_reflected_in_summary(budget_db):
    await init_db()

    await initialize_request_budget("run-x", 5_000_000)  # $5 request ceiling
    res = await reserve_budget("run-x", "panel", 1_000_000)  # reserve $1
    await reconcile_budget(res, 800_000)  # actual $0.80

    summary = await collect_budget_summary()

    # Global daily window records the true spend; reservation released.
    assert summary["global_daily"]["spent_usd"] == pytest.approx(0.8)
    assert summary["global_daily"]["reserved_usd"] == pytest.approx(0.0)

    # The request window shows the reconciled spend.
    req = next(r for r in summary["requests"] if r["run_id"] == "run-x")
    assert req["spent_usd"] == pytest.approx(0.8)
    assert req["ceiling_usd"] == pytest.approx(5.0)

    # The reservation is reconciled and no longer open.
    reservation = next(r for r in summary["reservations"] if r["reservation_id"] == res)
    assert reservation["state"] == "reconciled"
    assert reservation["stage"] == "panel"
    assert summary["open_reservations"] == 0

    _assert_no_secrets(summary)


@pytest.mark.asyncio
async def test_open_reservation_is_surfaced(budget_db):
    await init_db()
    await initialize_request_budget("run-open", 5_000_000)
    res = await reserve_budget("run-open", "judge", 500_000)

    summary = await collect_budget_summary()
    reservation = next(r for r in summary["reservations"] if r["reservation_id"] == res)
    assert reservation["state"] == "reserved"
    assert summary["open_reservations"] >= 1
    # Reserved-but-not-spent shows up as reserved on the global window.
    assert summary["global_daily"]["reserved_usd"] >= 0.5

    # Cancelling releases it.
    await cancel_reservation(res)
    after = await collect_budget_summary()
    assert after["open_reservations"] == 0


@pytest.mark.asyncio
async def test_empty_ledger_summary_is_well_formed(budget_db):
    await init_db()
    summary = await collect_budget_summary()
    assert summary["requests"] == []
    assert summary["reservations"] == []
    assert summary["global_daily"]["spent_usd"] == 0.0
    assert summary["open_reservations"] == 0
