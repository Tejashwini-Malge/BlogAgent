"""
Three auth tiers for three genuinely different callers of this app:

  Tier A — require_user_api / require_user_page — the operator's own
           browser (the SPA, scheduler control, run history, profile).
           Real accounts now (src/accounts.py, SQLite-backed), session
           identified by an opaque, database-backed `session` cookie — the
           same capability-token pattern as Tier B's review_token, not a
           signed cookie, so no extra signing-secret env var is needed.
           Cookies are sent automatically by the browser on every request
           to the origin, including `EventSource` connections (which can't
           set custom JS headers) — so this satisfies /api/generate's SSE
           constraint the same way the old Basic Auth did, for free.
           _api 401s (JSON) on a missing/invalid/expired session; _page
           redirects (302) to /login instead, for actual page loads.

  Tier B — require_review_token — review-email magic links, clicked from a
           phone with no prior login. A per-post token (src/pending.py's
           `review_token` field) travels in the URL's `?t=` query param
           instead of requiring a session.

  Tier C — require_api_key      — src/hermes_routes.py's routes, documented
           as "for Hermes Agent or a curl call" — a static key in a header,
           no browser involved.

Tier A fails CLOSED on HERMES_API_KEY being unset (Tiers B/C don't need any
extra secret — B is per-post, C's key IS the secret). `check_secrets_configured()`
is called once at app startup (see app.py's lifespan) so a misconfigured
deploy fails loudly before serving a single request, rather than quietly
running open.

`rate_limit()` is a light, single-process, in-memory sliding-window
limiter — defense-in-depth against a leaked credential, not anti-abuse
infrastructure for a multi-tenant service.
"""
import os
import secrets
import threading
import time
from collections import defaultdict

from fastapi import Header, HTTPException, Query, Request

SESSION_COOKIE = "session"


def check_secrets_configured() -> None:
    """Call once at startup. Raises if any required secret is unset/empty —
    fail closed instead of silently serving every route unauthenticated."""
    missing = [
        name for name in ("HERMES_API_KEY",)
        if not os.getenv(name, "").strip()
    ]
    if missing:
        raise RuntimeError(
            "Refusing to start: missing required auth env var(s) "
            f"{', '.join(missing)}. Set them in .env (see .env.example)."
        )


# ── Tier A: browser SPA, real accounts ────────────────────────────────────

def require_user_api(request: Request) -> dict:
    """For JSON API routes — 401s on a missing/invalid/expired session."""
    from src import accounts

    token = request.cookies.get(SESSION_COOKIE, "")
    user = accounts.get_session_user(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Not signed in.")
    return user


class RedirectToLogin(Exception):
    """Raised by require_user_page and turned into a 302 by app.py's
    exception handler — a dependency can't return a RedirectResponse
    directly and still let the route function run."""


def require_user_page(request: Request) -> dict:
    """For full-page routes (/, /profile) — redirects to /login instead of
    401ing, since a JSON error body is useless when the browser navigated
    here directly."""
    from src import accounts

    token = request.cookies.get(SESSION_COOKIE, "")
    user = accounts.get_session_user(token)
    if user is None:
        raise RedirectToLogin()
    return user


# ── Tier B: review-email magic links ─────────────────────────────────────

def require_review_token(post_id: str, t: str = Query(default="")) -> dict:
    """Loads the post as a side effect so route handlers don't re-fetch it —
    depend on this and take the returned dict instead of calling
    pending.get_post() again."""
    from src import pending as pending_store

    post = pending_store.get_post(post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found.")

    expected = post.get("review_token") or ""
    if not expected or not secrets.compare_digest(t, expected):
        raise HTTPException(status_code=403, detail="Invalid or missing review token.")
    return post


# ── Tier C: Hermes / programmatic routes ─────────────────────────────────

def require_api_key(x_api_key: str = Header(default="")) -> None:
    expected = os.getenv("HERMES_API_KEY", "")
    if not expected or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ── rate limiting ─────────────────────────────────────────────────────────

_buckets: dict = defaultdict(list)
_buckets_lock = threading.Lock()


def rate_limit(key_prefix: str, max_calls: int = 10, window_seconds: int = 60):
    """Returns a FastAPI dependency enforcing `max_calls` per `window_seconds`
    per (route, client IP). In-memory, single-process — fine for a
    single-operator deployment; not meant to survive a restart or scale
    across workers."""
    def _dependency(request: Request) -> None:
        client_ip = request.client.host if request.client else "unknown"
        key = f"{key_prefix}:{client_ip}"
        now = time.monotonic()
        cutoff = now - window_seconds
        with _buckets_lock:
            bucket = _buckets[key]
            while bucket and bucket[0] < cutoff:
                bucket.pop(0)
            if len(bucket) >= max_calls:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded — slow down and try again shortly.",
                    headers={"Retry-After": str(window_seconds)},
                )
            bucket.append(now)
    return _dependency
