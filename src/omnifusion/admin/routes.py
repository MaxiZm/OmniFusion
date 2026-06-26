from fastapi import APIRouter, Request, Depends, Form, HTTPException, status, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse
from argon2 import PasswordHasher
from typing import List
from importlib import resources
from pathlib import Path
import secrets
import time
import uuid
import asyncio
import logging

from ..settings import settings
from ..store.db import get_db_connection
from ..store.providers import (
    get_provider,
    list_providers,
    save_provider,
    delete_provider,
)
from ..store.presets import list_presets, save_preset, delete_preset
from ..fusion.types import Preset, PresetPrompts, PresetStage
from ..fusion.runtime.registry import registry as strategy_registry
from .jobs import job_registry, run_playground_job
from .csrf import generate_csrf_token

router = APIRouter()


def template_directory() -> Path:
    return Path(str(resources.files("omnifusion").joinpath("web", "templates")))


templates = Jinja2Templates(directory=str(template_directory()))
ph = PasswordHasher()
logger = logging.getLogger("omnifusion.admin")

# Ensure admin password is set
if not settings.omnifusion_admin_password:
    from ..api.errors import ConfigurationError

    raise ConfigurationError(
        "OMNIFUSION_ADMIN_PASSWORD environment variable must be set."
    )

# Fix (medium — ADMIN_HASH frozen at import): Compute hash lazily in the login
# handler rather than at module import time. This allows tests to override the
# password via env before importing. We store the hash in a module-level variable
# that is computed once on first login attempt rather than at import time.
_admin_hash: str | None = None


def _get_admin_hash() -> str:
    global _admin_hash
    if _admin_hash is None:
        _admin_hash = ph.hash(settings.omnifusion_admin_password.get_secret_value())
    return _admin_hash


# Fix #8: Per-IP login attempt tracking for brute-force protection.
# Structure: {ip: {"count": int, "locked_until": float, "ts": float}}
_login_attempts: dict = {}
_login_attempts_lock = asyncio.Lock()


def _client_ip(request: Request) -> str:
    """Resolve the client IP for lockout/logging. Honors the first X-Forwarded-For
    hop only when explicitly trusted (behind a known reverse proxy); otherwise uses
    the socket peer. Without this, all clients behind a proxy share one bucket and a
    single attacker locks out everyone."""
    if settings.omnifusion_trust_proxy_headers:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            first = xff.split(",")[0].strip()
            if first:
                return first
    return request.client.host if request.client else "unknown"


def _prune_login_attempts(now: float) -> None:
    """Drop stale entries so the dict can't grow unbounded (called under the lock)."""
    horizon = settings.omnifusion_login_lockout_seconds
    stale = [
        ip
        for ip, r in _login_attempts.items()
        if r.get("locked_until", 0) < now and (now - r.get("ts", 0)) > horizon
    ]
    for ip in stale:
        _login_attempts.pop(ip, None)


async def _check_login_rate_limit(client_ip: str) -> None:
    """Raises HTTPException 429 if this IP is currently locked out."""
    async with _login_attempts_lock:
        record = _login_attempts.get(client_ip)
        if record:
            if record.get("locked_until", 0) > time.time():
                retry_after = int(record["locked_until"] - time.time())
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Too many failed login attempts. Try again in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)},
                )


async def _record_failed_login(client_ip: str) -> None:
    """Increment failure count; lock out IP if threshold exceeded."""
    async with _login_attempts_lock:
        now = time.time()
        _prune_login_attempts(now)
        record = _login_attempts.setdefault(
            client_ip, {"count": 0, "locked_until": 0, "ts": now}
        )
        record["count"] += 1
        record["ts"] = now
        if record["count"] >= settings.omnifusion_max_login_attempts:
            record["locked_until"] = time.time() + settings.omnifusion_login_lockout_seconds
            logger.warning(
                f"Login brute-force lockout applied for IP {client_ip} "
                f"after {record['count']} failed attempts"
            )


async def _clear_login_attempts(client_ip: str) -> None:
    """Clear failed attempts after successful login."""
    async with _login_attempts_lock:
        _login_attempts.pop(client_ip, None)


async def verify_admin_session(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/admin/login"},
        )

    async with get_db_connection() as db:
        cursor = await db.execute(
            "SELECT username, csrf_token, expires_at FROM sessions WHERE session_id=?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                headers={"Location": "/admin/login"},
            )

        username, csrf_token, expires_at = row
        if time.time() > expires_at:
            await db.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                headers={"Location": "/admin/login"},
            )

        # Validate CSRF on mutations
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            from .csrf import validate_csrf

            await validate_csrf(request, session_id)

        return {
            "session_id": session_id,
            "username": username,
            "csrf_token": csrf_token,
        }


# Context processor for CSRF token
def render_template(
    template_name: str, request: Request, context: dict = {}, session: dict = None
):
    csrf_token = session["csrf_token"] if session else ""
    full_context = {"csrf_token": csrf_token, **context}
    return templates.TemplateResponse(request, template_name, full_context)


# Auth Routes
@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
async def login(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
):
    client_ip = _client_ip(request)

    # Fix #8: Check if this IP is locked out before processing
    await _check_login_rate_limit(client_ip)

    if username != "admin":
        await _record_failed_login(client_ip)
        return templates.TemplateResponse(
            request, "login.html", {"error": "Invalid username or password"}
        )

    try:
        ph.verify(_get_admin_hash(), password)
    except Exception:
        await _record_failed_login(client_ip)
        return templates.TemplateResponse(
            request, "login.html", {"error": "Invalid username or password"}
        )

    # Success: clear failed attempts
    await _clear_login_attempts(client_ip)

    old_session_id = request.cookies.get("session_id")
    if old_session_id:
        async with get_db_connection() as db:
            await db.execute("DELETE FROM sessions WHERE session_id=?", (old_session_id,))
            await db.commit()

    # Create session
    session_id = secrets.token_hex(32)
    csrf_token = generate_csrf_token()
    expires_at = int(time.time()) + 3600  # 1 hour session

    async with get_db_connection() as db:
        await db.execute(
            "INSERT INTO sessions (session_id, username, csrf_token, expires_at) VALUES (?, ?, ?, ?)",
            (session_id, "admin", csrf_token, expires_at),
        )
        await db.commit()

    # Fix #7: Honour OMNIFUSION_SECURE_COOKIE setting (must be True in production)
    response = RedirectResponse("/admin/", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="strict",
        secure=settings.omnifusion_secure_cookie,  # True in production (HTTPS)
        max_age=3600,
    )
    return response


@router.post("/logout")
async def logout(response: Response, session=Depends(verify_admin_session)):
    session_id = session["session_id"]
    async with get_db_connection() as db:
        await db.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        await db.commit()

    response = RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("session_id")
    return response


# Pages
@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, session=Depends(verify_admin_session)):
    # Load counts
    async with get_db_connection() as db:
        cursor = await db.execute("SELECT COUNT(*) FROM runs")
        total_runs = (await cursor.fetchone())[0]

        cursor = await db.execute("SELECT IFNULL(SUM(cost_usd), 0.0) FROM runs")
        total_spent = (await cursor.fetchone())[0]

        cursor = await db.execute("SELECT COUNT(*) FROM providers")
        total_providers = (await cursor.fetchone())[0]

        cursor = await db.execute("SELECT COUNT(*) FROM presets")
        total_presets = (await cursor.fetchone())[0]

        # Load daily spent
        today_str = time.strftime("%Y-%m-%d")
        cursor = await db.execute(
            "SELECT spent_micro_usd, ceiling_micro_usd FROM budget_ledger WHERE scope='global' AND window_key=?",
            (today_str,),
        )
        row = await cursor.fetchone()
        daily_spent = row[0] / 1_000_000 if row else 0.0
        daily_ceiling = row[1] / 1_000_000 if row else settings.global_daily_budget_usd

    context = {
        "total_runs": total_runs,
        "total_spent": round(total_spent, 5),
        "total_providers": total_providers,
        "total_presets": total_presets,
        "daily_spent": round(daily_spent, 5),
        "daily_ceiling": daily_ceiling,
    }
    return render_template("dashboard.html", request, context, session)


@router.get("/providers", response_class=HTMLResponse)
async def providers_page(request: Request, session=Depends(verify_admin_session)):
    providers = await list_providers()
    return render_template("providers.html", request, {"providers": providers}, session)


@router.post("/providers/save")
async def save_provider_route(
    id: str = Form(...),
    type: str = Form(...),
    api_key: str = Form(""),
    api_key_ref: str = Form(""),
    base_url: str = Form(""),
    models_raw: str = Form(""),
    session=Depends(verify_admin_session),
):
    models = [m.strip() for m in models_raw.split(",") if m.strip()]
    try:
        await save_provider(
            provider_id=id,
            p_type=type,
            plain_key=api_key,
            base_url=base_url if base_url else None,
            api_key_ref=api_key_ref if api_key_ref else None,
            models=models,
        )
    except Exception as e:
        return HTMLResponse(
            f"<div class='alert alert-danger'>Error: {str(e)}</div>", status_code=400
        )

    return RedirectResponse("/admin/providers", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/providers/{provider_id}/delete")
async def delete_provider_route(
    provider_id: str, session=Depends(verify_admin_session)
):
    await delete_provider(provider_id)
    return RedirectResponse("/admin/providers", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/providers/{provider_id}/test")
async def test_provider_route(provider_id: str, session=Depends(verify_admin_session)):
    start = time.time()
    from ..llm.client import llm_client
    from ..secrets.redact import redactor

    provider = await get_provider(provider_id)
    if not provider:
        return HTMLResponse("<span class='badge bg-danger'>Not Found</span>")

    models = provider.get("models", [])
    if not models:
        return HTMLResponse("<span class='badge bg-warning'>No Models</span>")

    model = models[0]
    try:
        # Resolve test completion call
        await llm_client.acompletion(
            provider_id=provider_id,
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            timeout=5,
            max_tokens=1,
        )
        latency = int((time.time() - start) * 1000)
        return HTMLResponse(
            f"<span class='badge bg-success'>Success ({latency}ms)</span>"
        )
    except Exception as e:
        # Fix (medium): Redact the error message before embedding in HTML
        # to prevent decrypted API keys from leaking in the title attribute.
        raw_err = str(e)
        safe_err = redactor.redact(raw_err).replace("'", "&#39;").replace('"', "&quot;")
        return HTMLResponse(
            f"<span class='badge bg-danger' title='{safe_err}'>Failed</span>"
        )


@router.get("/presets", response_class=HTMLResponse)
async def presets_page(request: Request, session=Depends(verify_admin_session)):
    presets = await list_presets()
    providers = await list_providers()

    # Gather all models from providers
    all_models = []
    for p in providers:
        all_models.extend(p.get("models", []))

    return render_template(
        "presets.html", request, {"presets": presets, "all_models": all_models}, session
    )


@router.post("/presets/save")
async def save_preset_route(
    name: str = Form(...),
    strategy: str = Form(...),
    panel_models_raw: List[str] = Form(...),
    panel_max_tokens: int = Form(...),
    panel_timeout: int = Form(...),
    judge_model: str = Form(...),
    judge_max_tokens: int = Form(...),
    judge_timeout: int = Form(...),
    final_model: str = Form(...),
    final_max_tokens: int = Form(...),
    final_timeout: int = Form(...),
    cost_ceiling: float = Form(...),
    on_final_failure: str = Form("error"),
    min_panel_success: int = Form(1),
    display_name: str = Form(""),
    mode: str = Form("fusion"),
    web_enabled: bool = Form(False),
    prompt_global: str = Form(""),
    prompt_panel: str = Form(""),
    prompt_judge: str = Form(""),
    prompt_final: str = Form(""),
    session=Depends(verify_admin_session),
):
    public_strategies = [key for key in strategy_registry.keys() if not key.startswith("_")]
    if strategy not in public_strategies:
        available = ", ".join(public_strategies)
        return HTMLResponse(
            f"<div class='alert alert-danger'>Unknown strategy '{strategy}'. "
            f"Available strategies: {available}.</div>",
            status_code=400,
        )

    role_prompts = {
        role: text
        for role, text in (
            ("panel", prompt_panel),
            ("judge", prompt_judge),
            ("final", prompt_final),
        )
        if text.strip()
    }
    preset = Preset(
        name=name,
        display_name=display_name or name,
        mode=mode if mode in ("fusion", "fugu_compat") else "fusion",
        strategy=strategy,
        web_enabled=web_enabled,
        prompts=PresetPrompts(global_prompt=prompt_global, role_prompts=role_prompts),
        panel_models=panel_models_raw,
        panel=PresetStage(max_tokens=panel_max_tokens, timeout=panel_timeout),
        judge_model=judge_model,
        judge=PresetStage(max_tokens=judge_max_tokens, timeout=judge_timeout),
        final_model=final_model,
        final=PresetStage(max_tokens=final_max_tokens, timeout=final_timeout),
        cost_ceiling=cost_ceiling,
        on_final_failure=on_final_failure,
        min_panel_success=min_panel_success,
    )

    await save_preset(preset)
    return RedirectResponse("/admin/presets", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/presets/{name}/delete")
async def delete_preset_route(name: str, session=Depends(verify_admin_session)):
    await delete_preset(name)
    return RedirectResponse("/admin/presets", status_code=status.HTTP_303_SEE_OTHER)


# Playground
@router.get("/playground", response_class=HTMLResponse)
async def playground_page(request: Request, session=Depends(verify_admin_session)):
    presets = await list_presets()
    return render_template("playground.html", request, {"presets": presets}, session)


@router.post("/playground/run")
async def run_playground(
    preset_name: str = Form(...),
    prompt: str = Form(...),
    session=Depends(verify_admin_session),
):
    run_id = str(uuid.uuid4())

    # Spawn background task
    task = asyncio.create_task(
        run_playground_job(run_id, preset_name, prompt, session["session_id"])
    )

    # Register in registry
    await job_registry.register(run_id, task)

    return {"run_id": run_id}


@router.get("/playground/runs/{run_id}/events")
async def playground_events(
    run_id: str, request: Request, session=Depends(verify_admin_session)
):
    # Retrieve job
    job = await job_registry.get(run_id)
    if not job:
        raise HTTPException(status_code=404, detail="Run not found")

    # Mark connected
    job.is_connected = True
    job.last_connected = time.time()

    async def event_generator():
        queue = job.queue
        try:
            while True:
                if await request.is_disconnected():
                    job.is_connected = False
                    job.last_connected = time.time()
                    break

                # Dequeue event with timeout. Yield dict-shaped events so
                # EventSourceResponse formats them as proper named SSE events
                # ("event: <name>\r\ndata: <payload>"). Yielding a preformatted
                # SSE string here would be re-wrapped as a `data:` field, mangling
                # the event type so the browser's named listeners never fire.
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # No event yet; loop to re-check disconnect. EventSourceResponse
                    # emits its own keepalive pings, so no manual heartbeat needed.
                    continue

                yield {"event": event["event"], "data": event["data"]}
                if event["event"] in ("done", "error", "cancelled"):
                    break
        except asyncio.CancelledError:
            pass

    return EventSourceResponse(event_generator())


@router.post("/playground/runs/{run_id}/cancel")
async def cancel_playground(run_id: str, session=Depends(verify_admin_session)):
    await job_registry.cancel(run_id)
    return {"status": "ok"}


# History
@router.get("/runs", response_class=HTMLResponse)
async def runs_history(request: Request, session=Depends(verify_admin_session)):
    runs = []
    async with get_db_connection() as db:
        cursor = await db.execute(
            "SELECT run_id, preset, created_by_key_hash, wall_ms, cost_usd, store_flag, expires_at FROM runs ORDER BY expires_at DESC LIMIT 100"
        )
        async for row in cursor:
            runs.append(
                {
                    "run_id": row[0],
                    "preset": row[1],
                    "key_hash": row[2][:8] if row[2] else "admin",
                    "wall_ms": row[3],
                    "cost_usd": round(row[4], 5),
                    "store_flag": bool(row[5]),
                    "expires_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(row[6])
                    )
                    if row[6]
                    else None,
                }
            )
    return render_template("runs.html", request, {"runs": runs}, session)


@router.get("/runs/{run_id}/trace")
async def view_run_trace(run_id: str, session=Depends(verify_admin_session)):
    from ..store.runs import get_trace

    trace = await get_trace(run_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    import html

    raw_json = trace.model_dump_json(indent=2)
    escaped_json = html.escape(raw_json)

    return HTMLResponse(
        f"<pre class='text-light bg-dark p-3 rounded'><code>{escaped_json}</code></pre>"
    )
