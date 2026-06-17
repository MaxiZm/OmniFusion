import secrets
from fastapi import Request, HTTPException, status
from ..store.db import get_db_connection


def generate_csrf_token() -> str:
    return secrets.token_hex(32)


async def validate_csrf(request: Request, session_id: str):
    # Safe methods do not require CSRF
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return

    token = request.headers.get("x-csrf-token")
    if not token:
        try:
            form = await request.form()
            token = form.get("csrf_token")
        except Exception:
            pass

    if not token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token missing"
        )

    async with get_db_connection() as db:
        cursor = await db.execute(
            "SELECT csrf_token FROM sessions WHERE session_id=?", (session_id,)
        )
        row = await cursor.fetchone()
        # Byte-safe constant-time compare: a non-ASCII token must yield a clean 403,
        # not a 500 (secrets.compare_digest raises on non-ASCII str). Guard None row[0].
        stored = row[0] if row else None
        valid = False
        if stored:
            try:
                valid = secrets.compare_digest(stored.encode("utf-8"), token.encode("utf-8"))
            except Exception:
                valid = False
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token invalid"
            )
