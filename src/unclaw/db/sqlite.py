"""SQLite helpers for Unclaw local persistence."""

from __future__ import annotations

import os
import sqlite3
import warnings
from pathlib import Path

_DATABASE_FILE_MODE = 0o600


def open_connection(database_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection configured for dictionary-like row access."""

    path = Path(database_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    ensure_database_permissions(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def ensure_database_permissions(path: Path) -> None:
    """Enforce owner-only permissions for the local SQLite database on POSIX."""

    if os.name != "posix" or not path.exists():
        return

    try:
        os.chmod(path, _DATABASE_FILE_MODE)
    except OSError:
        warnings.warn(
            (
                "Unclaw could not enforce owner-only permissions on the local "
                f"SQLite database: {path}"
            ),
            stacklevel=2,
        )


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create the base SQLite schema used by the runtime."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            summary_text TEXT,
            is_active INTEGER NOT NULL CHECK (is_active IN (0, 1))
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
        ON sessions (updated_at DESC);

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session_created_at
        ON messages (session_id, created_at);

        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            event_type TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            payload_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_events_session_created_at
        ON events (session_id, created_at DESC);
        """
    )
    _ensure_session_summary_column(connection)
    connection.commit()


def _ensure_session_summary_column(connection: sqlite3.Connection) -> None:
    column_names = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in connection.execute("PRAGMA table_info(sessions);").fetchall()
    }
    if "summary_text" in column_names:
        return

    connection.execute("ALTER TABLE sessions ADD COLUMN summary_text TEXT;")
