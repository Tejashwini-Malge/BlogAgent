"""
User accounts and sessions, backed by src/db.py.

Password hashing is stdlib-only (hashlib.pbkdf2_hmac + a random salt per
user) rather than pulling in passlib/bcrypt — one function, no new
dependency, same "don't add a library until it earns its place" call made
in src/auth.py. Sessions are opaque, database-backed tokens (secrets.
token_urlsafe), the same capability-token pattern already used for
pending.py's review_token — not signed cookies, so no signing-secret env
var is needed either.
"""
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from src.db import get_connection

_PBKDF2_ITERATIONS = 260_000
_SESSION_LIFETIME = timedelta(days=30)


class AccountError(Exception):
    """User-facing account errors (duplicate email, bad credentials)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, hex_digest = stored_hash.split("$", 1)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS)
    return secrets.compare_digest(digest.hex(), hex_digest)


def create_user(email: str, password: str) -> dict:
    email = email.strip().lower()
    if not email or "@" not in email:
        raise AccountError("Enter a valid email address.")
    if len(password) < 8:
        raise AccountError("Password must be at least 8 characters.")

    user = {
        "id": str(uuid.uuid4()),
        "email": email,
        "password_hash": hash_password(password),
        "created_at": _now(),
    }
    with get_connection() as conn:
        try:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (user["id"], user["email"], user["password_hash"], user["created_at"]),
            )
        except Exception as exc:
            if "UNIQUE" in str(exc):
                raise AccountError("An account with that email already exists.")
            raise
    return user


def get_user_by_email(email: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def authenticate(email: str, password: str) -> dict:
    """Raises AccountError on any failure — deliberately the same message
    for 'no such user' and 'wrong password' so a login form doesn't leak
    which emails have accounts."""
    user = get_user_by_email(email)
    if user is None or not verify_password(password, user["password_hash"]):
        raise AccountError("Incorrect email or password.")
    return user


def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now.isoformat(), (now + _SESSION_LIFETIME).isoformat()),
        )
    return token


def get_session_user(token: str) -> dict | None:
    if not token:
        return None
    with get_connection() as conn:
        row = conn.execute(
            "SELECT s.expires_at, u.* FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token = ?",
            (token,),
        ).fetchone()
    if row is None:
        return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
        delete_session(token)
        return None
    user = dict(row)
    user.pop("expires_at", None)
    return user


def delete_session(token: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
