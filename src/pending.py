"""
Post lifecycle store.

Persists posts to data/pending.json with thread-safe reads/writes.

Post shape:
  {
    "id":            str  (uuid4),
    "topic":         str,
    "tone":          str,
    "length":        str,
    "audience":      str,
    "content":       str  (final markdown),
    "status":        "pending" | "approved" | "skipped" | "published",
    "created_at":    ISO-8601,
    "approved_at":   ISO-8601 | null,
    "approved_by":   str | null  ("review-link" or "hermes-api" — audit
                     provenance for a single-operator app, not authorization;
                     Tier B review links have no login, so identity is only
                     ever "possessed the token", not a named user),
    "review_token":  str  (secrets.token_urlsafe — gates access to the
                     /review/{id} page and its approve/skip/revise actions;
                     see src/auth.py's require_review_token),
    "user_id":       str | null  (owner's account id from src/accounts.py;
                     null for posts created before accounts existed, and for
                     scheduler/Hermes-triggered drafts — see project memory,
                     the daily automated cycle is still global/unowned),
    "corrections": [{"before": str, "after": str, "reason": str, "ts": str}]
  }

Hard rule: only posts with status=="approved" are ever published.
"""

import json
import secrets
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.paths import data_file

_STORE = data_file("pending.json")
_lock = threading.Lock()


# ── helpers ───────────────────────────────────────────────────────────────────

def _load() -> list:
    if not _STORE.exists():
        return []
    try:
        return json.loads(_STORE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save(posts: list) -> None:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    _STORE.write_text(json.dumps(posts, indent=2, ensure_ascii=False), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── public API ─────────────────────────────────────────────────────────────────

def create_post(
    topic: str,
    content: str,
    tone: str = "professional",
    length: str = "medium",
    audience: str = "general",
    notes: str = "",
    grounding: dict | None = None,
    run_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    """
    Create a new pending post. Returns the post dict.

    `grounding` and `run_id` are optional so posts written before run records
    existed still load: readers use post.get("grounding") and treat a missing
    value as unknown rather than as grounded. No migration of pending.json.

    `user_id` is None for scheduler/Hermes-triggered drafts (that cycle is
    still global — see project memory) and set to the logged-in account's id
    when a post is created from the authenticated web UI.
    """
    post = {
        "id": str(uuid.uuid4()),
        "topic": topic,
        "tone": tone,
        "length": length,
        "audience": audience,
        "notes": notes,
        "content": content,
        "status": "pending",
        "created_at": _now(),
        "grounding": grounding,
        "run_id": run_id,
        "approved_at": None,
        "approved_by": None,
        "published_at": None,
        "review_token": secrets.token_urlsafe(24),
        "user_id": user_id,
        "corrections": [],
    }
    with _lock:
        posts = _load()
        posts.append(post)
        _save(posts)
    return post


def get_post(post_id: str) -> Optional[dict]:
    """Return the post or None if not found."""
    with _lock:
        posts = _load()
    for p in posts:
        if p["id"] == post_id:
            return p
    return None


def update_post(post_id: str, **fields) -> Optional[dict]:
    """Update arbitrary fields on a post. Returns updated post or None."""
    with _lock:
        posts = _load()
        for p in posts:
            if p["id"] == post_id:
                p.update(fields)
                _save(posts)
                return p
    return None


def add_correction(post_id: str, before: str, after: str, reason: str = "") -> Optional[dict]:
    """Append a correction to a post's corrections list. Returns the post."""
    with _lock:
        posts = _load()
        for p in posts:
            if p["id"] == post_id:
                p.setdefault("corrections", []).append({
                    "ts": _now(),
                    "before": before[:500],
                    "after": after[:500],
                    "reason": reason[:300],
                })
                _save(posts)
                return p
    return None


def approved_posts() -> list:
    """Posts ready to publish (status == 'approved')."""
    with _lock:
        return [p for p in _load() if p.get("status") == "approved"]


def pending_posts() -> list:
    """Posts still awaiting review (status == 'pending')."""
    with _lock:
        return [p for p in _load() if p.get("status") == "pending"]


def all_posts() -> list:
    with _lock:
        return _load()


def posts_for_user(user_id: str) -> list:
    """Posts owned by one account — the web UI's history drawer and profile
    page use this, not all_posts(), so one user never sees another's drafts."""
    with _lock:
        return [p for p in _load() if p.get("user_id") == user_id]
