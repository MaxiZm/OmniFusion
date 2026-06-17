import aiosqlite
import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from ..settings import settings

logger = logging.getLogger("omnifusion.db")

# Fix #16: Track whether WAL mode has been set for this DB path.
# WAL mode is persistent across connections for the same database file,
# so we only need to issue PRAGMA journal_mode=WAL once at init time.
_wal_initialized: set = set()
_wal_init_lock = asyncio.Lock()


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[aiosqlite.Connection, None]:
    """
    Fix #16: Reduced per-connection overhead.
    - PRAGMA journal_mode=WAL is issued lazily only once (WAL is persistent).
    - busy_timeout and foreign_keys are still set per connection (they are
      connection-scoped pragmas that must be re-applied each connection).
    """
    async with aiosqlite.connect(settings.db_path, isolation_level=None) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("PRAGMA busy_timeout = 5000")  # 5 seconds retry on locked

        # Only issue WAL pragma if we haven't for this db_path yet. Verify it actually
        # took — SQLite silently returns the *current* mode (e.g. 'delete') instead of
        # erroring when it can't switch to WAL (network FS, etc.). Only mark initialized
        # on success so a transient failure is retried on the next connection.
        if settings.db_path not in _wal_initialized:
            async with _wal_init_lock:
                if settings.db_path not in _wal_initialized:
                    cursor = await db.execute("PRAGMA journal_mode = WAL")
                    row = await cursor.fetchone()
                    mode = (row[0] if row else "").lower()
                    if mode == "wal":
                        _wal_initialized.add(settings.db_path)
                    else:
                        logger.warning(
                            f"SQLite WAL mode not enabled (journal_mode={mode!r}); "
                            f"concurrency may be degraded. Will retry on next connection."
                        )

        yield db


async def sweep_expired_sessions():
    """Fix (medium): Remove expired sessions from the DB to prevent unbounded growth."""
    now = int(time.time())
    try:
        async with get_db_connection() as db:
            await db.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
            await db.commit()
    except Exception as e:
        logger.warning(f"Failed to sweep expired sessions: {e}")


async def init_db():
    import os

    db_dir = os.path.dirname(settings.db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # Ensure WAL mode is initialized fresh for a new DB
    _wal_initialized.discard(settings.db_path)

    async with get_db_connection() as db:
        # Create tables
        await db.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key_verify_token TEXT
        )
        """)

        await db.execute("""
        CREATE TABLE IF NOT EXISTS providers (
            id TEXT PRIMARY KEY,
            type TEXT,
            base_url TEXT,
            enc_key BLOB,
            api_key_ref TEXT,
            models_json TEXT
        )
        """)

        await db.execute("""
        CREATE TABLE IF NOT EXISTS presets (
            name TEXT PRIMARY KEY,
            strategy TEXT,
            spec_json TEXT
        )
        """)

        await db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            username TEXT,
            csrf_token TEXT,
            expires_at INTEGER
        )
        """)

        await db.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            preset TEXT,
            created_by_key_hash TEXT,
            wall_ms INTEGER,
            cost_usd REAL,
            tokens_json TEXT,
            store_flag BOOLEAN,
            expires_at INTEGER,
            trace_json TEXT
        )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_owner ON runs(created_by_key_hash)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_expires ON runs(expires_at)"
        )

        # Budget ledger
        await db.execute("""
        CREATE TABLE IF NOT EXISTS budget_ledger (
            scope TEXT, -- 'global' or 'request'
            window_key TEXT, -- e.g., run_id or date
            reserved_micro_usd INTEGER DEFAULT 0,
            spent_micro_usd INTEGER DEFAULT 0,
            ceiling_micro_usd INTEGER,
            PRIMARY KEY (scope, window_key)
        )
        """)

        await db.execute("""
        CREATE TABLE IF NOT EXISTS budget_reservations (
            reservation_id TEXT PRIMARY KEY,
            run_id TEXT,
            stage TEXT,
            reserved_micro_usd INTEGER,
            state TEXT, -- 'reserved', 'reconciled', 'cancelled'
            global_window_key TEXT,
            created_at INTEGER  -- Unix epoch; used by stale-reservation sweeper
        )
        """)

        # Migrate budget_reservations: add columns that may be missing on existing DBs
        cursor = await db.execute("PRAGMA table_info(budget_reservations)")
        columns = [row[1] for row in await cursor.fetchall()]
        if columns and "global_window_key" not in columns:
            await db.execute(
                "ALTER TABLE budget_reservations ADD COLUMN global_window_key TEXT"
            )
        if columns and "created_at" not in columns:
            # Fix A: add created_at; backfill with 0 so existing rows are immediately
            # eligible for a sweep (they have no meaningful age anyway).
            await db.execute(
                "ALTER TABLE budget_reservations ADD COLUMN created_at INTEGER DEFAULT 0"
            )

        await db.execute("""
        CREATE TABLE IF NOT EXISTS workers (
            pid INTEGER PRIMARY KEY,
            last_seen INTEGER
        )
        """)
        await db.commit()

        # Fix (medium): Sweep any expired sessions on startup
        await db.execute(
            "DELETE FROM sessions WHERE expires_at < ?", (int(time.time()),)
        )
        await db.commit()

        # Verify the secret key Fernet token
        from ..secrets.crypto import check_secret_key

        await check_secret_key(db)
