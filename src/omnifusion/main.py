import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from .settings import settings, validate_startup_security
from .api.chat import router as chat_router
from .api.traces import router as traces_router
from .api.models import router as models_router
from .admin.routes import router as admin_router
from .api.errors import (
    OmniFusionError,
    omnifusion_error_handler,
    generic_exception_handler,
)
from .secrets.redact import setup_logging_redaction
from .logging_config import configure_logging, set_run_id
from .ratelimit.circuit_breaker import circuit_breaker, configure_from_settings

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import os
import time
import asyncio
from .store.db import init_db, get_db_connection, sweep_expired_sessions

# Configure logging to redact secrets if necessary
configure_logging(settings.omnifusion_log_level, settings.omnifusion_log_format)
setup_logging_redaction()
logger = logging.getLogger("omnifusion")

# Fix (medium): Store strong references to background tasks to prevent GC-vanishing.
_background_tasks: list = []
_background_task_names: dict = {}


def _create_background_task(name: str, coro) -> asyncio.Task:
    task = asyncio.create_task(coro, name=f"omnifusion:{name}")
    _background_tasks.append(task)
    _background_task_names[task] = name
    return task


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting OmniFusion...")
    validate_startup_security()
    configure_from_settings(settings)

    # 1. Initialize DB and tables
    await init_db()
    # 2. Check and enforce single worker constraint
    await check_single_worker()

    # Fix (medium): Store task references so they can't be GC'd and can be
    # cancelled cleanly on shutdown.
    _create_background_task("heartbeat", worker_heartbeat_loop())
    _create_background_task("jobs_sweep", jobs_sweep_loop())
    _create_background_task("session_sweep", session_sweep_loop())
    _create_background_task("reservation_sweep", reservation_sweep_loop())
    _create_background_task("runs_sweep", runs_sweep_loop())

    if settings.omnifusion_unsafe_allow_multiworker:
        logger.warning(
            "OMNIFUSION_UNSAFE_ALLOW_MULTIWORKER=1 is set. "
            "WARNING: Limiters, circuit breakers, and the playground job registry are per-process. "
            "Durable budget ledgers in SQLite will remain correct, but rate limiting will not be strictly global."
        )

    try:
        yield
    finally:
        # Fix (medium): Cancel all background tasks on shutdown
        logger.info("Shutting down OmniFusion background tasks...")
        for task in _background_tasks:
            if not task.done():
                task.cancel()
        if _background_tasks:
            await asyncio.gather(*_background_tasks, return_exceptions=True)
        _background_tasks.clear()
        _background_task_names.clear()


app = FastAPI(
    title="OmniFusion",
    version="1.0.0",
    docs_url=None,  # Disable Swagger UI for production by default
    lifespan=lifespan,
)

app.add_exception_handler(OmniFusionError, omnifusion_error_handler)
app.add_exception_handler(Exception, generic_exception_handler)


@app.middleware("http")
async def run_id_logging_middleware(request: Request, call_next):
    set_run_id(getattr(request.state, "run_id", None))
    try:
        return await call_next(request)
    finally:
        set_run_id(None)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    errors = exc.errors()
    msg = "; ".join(
        [f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in errors]
    )
    # Correlate with the trace and keep the envelope consistent with the other
    # handlers (run-id header if a run was already assigned).
    run_id = getattr(request.state, "run_id", None)
    headers = {"X-OmniFusion-Run-Id": run_id} if run_id else {}
    return JSONResponse(
        status_code=400,
        headers=headers,
        content={
            "error": {
                "message": f"Validation error: {msg}",
                "type": "invalid_request_error",
                "param": None,
                "code": "validation_error",
            }
        },
    )


# CORS disabled by default (same-origin UI only)

app.include_router(chat_router, prefix="/v1")
app.include_router(traces_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")
app.include_router(admin_router, prefix="/admin")


@app.get("/health")
async def health():
    status_code = 200
    payload = {"status": "ok", "db": {"status": "ok"}, "tasks": {}}

    try:
        async with get_db_connection() as db:
            await db.execute("SELECT 1")
    except Exception as exc:
        status_code = 503
        payload["status"] = "unhealthy"
        payload["db"] = {"status": "unhealthy", "error": str(exc)}

    for task in _background_tasks:
        name = _background_task_names.get(task, task.get_name())
        task_status = "running"
        if task.done():
            task_status = "failed" if task.exception() else "stopped"
            if task_status == "failed":
                status_code = 503
                payload["status"] = "unhealthy"
        payload["tasks"][name] = {"status": task_status}

    payload["circuit_breaker"] = circuit_breaker.get_all_states()
    return JSONResponse(status_code=status_code, content=payload)


async def worker_heartbeat_loop():
    pid = os.getpid()
    while True:
        try:
            async with get_db_connection() as db:
                await db.execute(
                    "INSERT OR REPLACE INTO workers (pid, last_seen) VALUES (?, ?)",
                    (pid, int(time.time())),
                )
                await db.commit()
        except Exception:
            pass
        await asyncio.sleep(3)


async def check_single_worker():
    pid = os.getpid()
    now = int(time.time())
    async with get_db_connection() as db:
        await db.execute("BEGIN IMMEDIATE")
        try:
            # Clean up heartbeats older than 8 seconds
            await db.execute("DELETE FROM workers WHERE last_seen < ?", (now - 8,))
            
            # Fetch all other active heartbeats
            cursor = await db.execute(
                "SELECT pid FROM workers WHERE pid != ?", (pid,)
            )
            other_worker_pids = [r[0] for r in await cursor.fetchall()]

            active_other_workers = 0
            for opid in other_worker_pids:
                try:
                    os.kill(opid, 0)
                    active_other_workers += 1
                except OSError:
                    # Clean up defunct worker immediately
                    await db.execute("DELETE FROM workers WHERE pid = ?", (opid,))

            if active_other_workers > 0:
                if not settings.omnifusion_unsafe_allow_multiworker:
                    try:
                        await db.execute("ROLLBACK")
                    except Exception:
                        pass
                    logger.error(
                        "Multiple worker processes detected! Refusing to start because settings.omnifusion_unsafe_allow_multiworker is False."
                    )
                    raise RuntimeError(
                        "Multiple worker processes detected. Single worker process required."
                    )
                else:
                    logger.warning(
                        "Multiple worker processes detected, but OMNIFUSION_UNSAFE_ALLOW_MULTIWORKER is enabled. Concurrency and playground task registries may not be shared."
                    )

            await db.execute(
                "INSERT OR REPLACE INTO workers (pid, last_seen) VALUES (?, ?)",
                (pid, now),
            )
            await db.commit()
        except Exception:
            try:
                await db.execute("ROLLBACK")
            except Exception:
                pass
            raise


async def jobs_sweep_loop():
    from .admin.jobs import job_registry

    while True:
        try:
            await job_registry.sweep()
        except Exception:
            pass
        await asyncio.sleep(5)


async def session_sweep_loop():
    """Fix (medium): Periodically sweep expired sessions from the DB."""
    while True:
        try:
            await sweep_expired_sessions()
        except Exception:
            pass
        await asyncio.sleep(300)  # Every 5 minutes


async def reservation_sweep_loop():
    """Fix #5: Periodically sweep stale budget reservations."""
    while True:
        try:
            from .budget.ledger import sweep_stale_reservations
            await sweep_stale_reservations()
        except Exception:
            pass
        await asyncio.sleep(600)  # Every 10 minutes


async def runs_sweep_loop():
    """Purge runs past their retention window so the runs table can't grow unbounded."""
    while True:
        try:
            from .store.runs import purge_expired_runs

            removed = await purge_expired_runs()
            if removed:
                logger.info(f"Purged {removed} expired run trace(s)")
        except Exception:
            pass
        await asyncio.sleep(3600)  # Every hour
