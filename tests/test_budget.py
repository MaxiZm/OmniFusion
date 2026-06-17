import pytest
import os
from omnifusion.budget.ledger import (
    initialize_request_budget,
    reserve_budget,
    reconcile_budget,
)
from omnifusion.api.errors import BudgetExceededError
from omnifusion.settings import settings
from omnifusion.store.db import init_db, get_db_connection


@pytest.fixture(autouse=True)
def setup_db():
    old_db = settings.db_path
    settings.db_path = "test_budget.db"
    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)
    yield
    if os.path.exists(settings.db_path):
        try:
            os.remove(settings.db_path)
        except Exception:
            pass
    settings.db_path = old_db


@pytest.mark.asyncio
async def test_budget_reserve_and_reconcile():
    await init_db()

    run_id = "test-run-123"
    await initialize_request_budget(run_id, 500000)

    resid = await reserve_budget(run_id, "panel/test", 100000)
    assert resid is not None

    async with get_db_connection() as db:
        cursor = await db.execute(
            "SELECT reserved_micro_usd, spent_micro_usd FROM budget_ledger WHERE scope='request' AND window_key=?",
            (run_id,),
        )
        reserved, spent = await cursor.fetchone()
        assert reserved == 100000
        assert spent == 0

    await reconcile_budget(resid, 80000)

    async with get_db_connection() as db:
        cursor = await db.execute(
            "SELECT reserved_micro_usd, spent_micro_usd FROM budget_ledger WHERE scope='request' AND window_key=?",
            (run_id,),
        )
        reserved, spent = await cursor.fetchone()
        assert reserved == 0
        assert spent == 80000


@pytest.mark.asyncio
async def test_budget_exceeded():
    await init_db()

    run_id = "test-run-456"
    await initialize_request_budget(run_id, 10000)

    with pytest.raises(BudgetExceededError):
        await reserve_budget(run_id, "panel/test", 15000)

    resid = await reserve_budget(run_id, "panel/test", 5000)
    assert resid is not None

    with pytest.raises(BudgetExceededError):
        await reserve_budget(run_id, "judge", 6000)
