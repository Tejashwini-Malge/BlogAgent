"""
SQLite storage for user accounts and sessions.

Everything else in this app (pending posts, topics, run records) stays on
JSON files for now — accounts are the one thing that needs real relational
integrity (unique emails, session expiry) and were already flagged as the
natural place to introduce SQLite first. Lives on the same DATA_DIR as
everything else (src/paths.py) so it inherits the existing ephemeral-storage
warning on platforms that rebuild the code directory each deploy.
"""
import sqlite3
from contextlib import contextmanager

from src.paths import data_file

_DB_PATH = data_file("blogagent.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
"""


def init_db() -> None:
    """Create tables if missing. Call once at startup — safe to call
    repeatedly, every statement is CREATE ... IF NOT EXISTS."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(_SCHEMA)


@contextmanager
def get_connection():
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
