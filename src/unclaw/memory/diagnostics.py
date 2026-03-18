"""Read-only memory layer diagnostics for the /memory-status command.

Collects a snapshot of all three memory layers without mutating any file or DB:
  1. Short-term / session DB   : data/memory/app.db
  2. Conversation memory       : data/memory/chats/{session_id}.jsonl
  3. Long-term memory DB       : data/memory/long_term.db

Design rules:
- Read-only only: no writes, no schema changes, no side effects.
- Paths derived from data_dir at call time — never hardcoded.
- Graceful: missing files/tables are reported clearly, never raised.
- stdlib only: sqlite3 + pathlib, no unclaw imports.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MemoryDiagnostics:
    """Read-only snapshot of the three memory layers."""

    # Canonical paths
    session_db_path: Path
    long_term_db_path: Path
    chats_dir_path: Path
    legacy_db_path: Path

    # Existence / health
    session_db_exists: bool
    long_term_db_exists: bool
    chats_dir_exists: bool
    legacy_db_exists: bool

    # Current session
    current_session_id: str | None
    session_jsonl_exists: bool

    # Counts
    session_db_message_count: int | None   # None = unavailable (table missing, etc.)
    session_jsonl_message_count: int | None
    total_chat_jsonl_files: int
    total_long_term_records: int | None    # None = unavailable

    # Sizes in bytes (None = file does not exist)
    session_db_size_bytes: int | None
    long_term_db_size_bytes: int | None
    session_jsonl_size_bytes: int | None


def collect_memory_diagnostics(
    data_dir: Path,
    session_db_path: Path,
    current_session_id: str | None,
) -> MemoryDiagnostics:
    """Return a read-only snapshot of the three memory layers.

    All path lookups and DB queries are best-effort: failures produce
    None counts rather than exceptions.
    """
    long_term_db_path = data_dir / "memory" / "long_term.db"
    chats_dir_path = data_dir / "memory" / "chats"
    legacy_db_path = data_dir / "app.db"

    session_db_exists = session_db_path.is_file()
    long_term_db_exists = long_term_db_path.is_file()
    chats_dir_exists = chats_dir_path.is_dir()
    legacy_db_exists = legacy_db_path.is_file()

    # Locate the current session's JSONL file.
    if current_session_id is not None and chats_dir_exists:
        session_jsonl_path: Path | None = (
            chats_dir_path / f"{current_session_id}.jsonl"
        )
        session_jsonl_exists = session_jsonl_path.is_file()
    else:
        session_jsonl_path = None
        session_jsonl_exists = False

    # Count messages for current session in SQLite (read-only connection).
    session_db_message_count: int | None = None
    if session_db_exists and current_session_id is not None:
        try:
            conn = sqlite3.connect(
                f"file:{session_db_path}?mode=ro", uri=True, check_same_thread=False
            )
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                    (current_session_id,),
                ).fetchone()
                session_db_message_count = int(row[0]) if row else 0
            finally:
                conn.close()
        except sqlite3.Error:
            session_db_message_count = None

    # Count messages for current session in JSONL.
    session_jsonl_message_count: int | None = None
    if session_jsonl_exists and session_jsonl_path is not None:
        try:
            text = session_jsonl_path.read_text(encoding="utf-8")
            session_jsonl_message_count = sum(
                1 for line in text.splitlines() if line.strip()
            )
        except OSError:
            session_jsonl_message_count = None

    # Count total JSONL files across all sessions.
    total_chat_jsonl_files = 0
    if chats_dir_exists:
        try:
            total_chat_jsonl_files = sum(
                1 for p in chats_dir_path.iterdir() if p.suffix == ".jsonl"
            )
        except OSError:
            total_chat_jsonl_files = 0

    # Count total long-term memory records (read-only connection).
    total_long_term_records: int | None = None
    if long_term_db_exists:
        try:
            conn = sqlite3.connect(
                f"file:{long_term_db_path}?mode=ro", uri=True, check_same_thread=False
            )
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM long_term_memories"
                ).fetchone()
                total_long_term_records = int(row[0]) if row else 0
            finally:
                conn.close()
        except sqlite3.Error:
            total_long_term_records = None

    # File sizes.
    session_db_size_bytes: int | None = None
    if session_db_exists:
        try:
            session_db_size_bytes = session_db_path.stat().st_size
        except OSError:
            pass

    long_term_db_size_bytes: int | None = None
    if long_term_db_exists:
        try:
            long_term_db_size_bytes = long_term_db_path.stat().st_size
        except OSError:
            pass

    session_jsonl_size_bytes: int | None = None
    if session_jsonl_exists and session_jsonl_path is not None:
        try:
            session_jsonl_size_bytes = session_jsonl_path.stat().st_size
        except OSError:
            pass

    return MemoryDiagnostics(
        session_db_path=session_db_path,
        long_term_db_path=long_term_db_path,
        chats_dir_path=chats_dir_path,
        legacy_db_path=legacy_db_path,
        session_db_exists=session_db_exists,
        long_term_db_exists=long_term_db_exists,
        chats_dir_exists=chats_dir_exists,
        legacy_db_exists=legacy_db_exists,
        current_session_id=current_session_id,
        session_jsonl_exists=session_jsonl_exists,
        session_db_message_count=session_db_message_count,
        session_jsonl_message_count=session_jsonl_message_count,
        total_chat_jsonl_files=total_chat_jsonl_files,
        total_long_term_records=total_long_term_records,
        session_db_size_bytes=session_db_size_bytes,
        long_term_db_size_bytes=long_term_db_size_bytes,
        session_jsonl_size_bytes=session_jsonl_size_bytes,
    )


def _fmt_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "n/a"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes // 1024} KB"
    return f"{size_bytes // (1024 * 1024)} MB"


def _exists_label(exists: bool) -> str:
    return "exists" if exists else "missing"


def render_memory_diagnostics(d: MemoryDiagnostics) -> tuple[str, ...]:
    """Render a compact human-readable memory diagnostics summary.

    Output is deterministic: same MemoryDiagnostics input → same lines output.
    Intended for terminal display; never emits JSON.
    """
    lines: list[str] = []

    lines.append("Memory layer diagnostics")
    lines.append("")

    lines.append("Paths:")
    lines.append(f"  session DB    : {d.session_db_path}")
    lines.append(f"  long-term DB  : {d.long_term_db_path}")
    lines.append(f"  chats dir     : {d.chats_dir_path}")
    lines.append(f"  legacy DB     : {d.legacy_db_path}")
    lines.append("")

    lines.append("Existence:")
    lines.append(f"  session DB    : {_exists_label(d.session_db_exists)}")
    lines.append(f"  long-term DB  : {_exists_label(d.long_term_db_exists)}")
    lines.append(f"  chats dir     : {_exists_label(d.chats_dir_exists)}")
    lines.append(f"  legacy DB     : {_exists_label(d.legacy_db_exists)}")
    lines.append("")

    if d.current_session_id is not None:
        lines.append(f"Current session: {d.current_session_id}")
        lines.append(f"  session JSONL : {_exists_label(d.session_jsonl_exists)}")
        lines.append("")

        lines.append("Session counts:")
        db_count = (
            str(d.session_db_message_count)
            if d.session_db_message_count is not None
            else "unavailable"
        )
        jsonl_count = (
            str(d.session_jsonl_message_count)
            if d.session_jsonl_message_count is not None
            else "unavailable"
        )
        lines.append(f"  messages in session DB   : {db_count}")
        lines.append(f"  messages in session JSONL: {jsonl_count}")
        lines.append("")
    else:
        lines.append("Current session: none")
        lines.append("")

    lines.append("Global counts:")
    ltm_count = (
        str(d.total_long_term_records)
        if d.total_long_term_records is not None
        else "unavailable"
    )
    lines.append(f"  total chat JSONL files   : {d.total_chat_jsonl_files}")
    lines.append(f"  long-term memory records : {ltm_count}")
    lines.append("")

    lines.append("Sizes:")
    lines.append(f"  session DB    : {_fmt_size(d.session_db_size_bytes)}")
    lines.append(f"  long-term DB  : {_fmt_size(d.long_term_db_size_bytes)}")
    if d.current_session_id is not None:
        lines.append(f"  session JSONL : {_fmt_size(d.session_jsonl_size_bytes)}")

    return tuple(lines)


__all__ = [
    "MemoryDiagnostics",
    "collect_memory_diagnostics",
    "render_memory_diagnostics",
]
